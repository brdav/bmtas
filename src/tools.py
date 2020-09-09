from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

from src import resources
from src import utils


def train_search(device,
                 start_epoch,
                 max_epochs,
                 tasks,
                 trainloader_weight,
                 trainloader_arch,
                 model,
                 loss,
                 optimizer_weight,
                 optimizer_arch,
                 exp_dir):

    writer = SummaryWriter(log_dir=exp_dir)

    iter_per_epoch = len(
        trainloader_weight.dataset) // trainloader_weight.batch_size
    total_iter = iter_per_epoch * max_epochs
    delay_epochs = max_epochs // 20

    model.train()
    for epoch in range(start_epoch, max_epochs + 1):

        model.warmup_flag = (epoch <= delay_epochs)
        # set the gumbel temperature according to a linear schedule
        model.gumbel_temp = min(5.0 - (epoch - delay_epochs - 1)
                                / (max_epochs - delay_epochs - 1) * (5.0 - 0.1), 5.0)

        arch_loss = 0
        arch_counter = 0

        if epoch > delay_epochs:
            print('modifying architecture...')

            # we reset the arch optimizer state
            optimizer_arch.state = defaultdict(dict)

            # we use current batch statistics in search period
            model.freeze_encoder_bn_running_stats()

            for samples_search in trainloader_arch:

                inputs_search = samples_search['image'].to(
                    device, non_blocking=True)
                target_search = {task: samples_search[task].to(
                    device, non_blocking=True) for task in tasks}

                optimizer_arch.zero_grad()

                for task in tasks:
                    # many images don't have human parts annotations, skip those
                    uniq = torch.unique(target_search[task])
                    if len(uniq) == 1 and uniq[0] == 255:
                        continue

                    output = model(inputs_search, task=task)
                    tot_loss = loss(output, target_search, task=task)
                    tot_loss.backward()

                    arch_loss += tot_loss.item()
                    arch_counter += 1

                optimizer_arch.step()

            # we reset the main optimizer state because arch has changed
            optimizer_weight.state = defaultdict(dict)

            # we should reset bn running stats
            model.unfreeze_encoder_bn_running_stats()
            model.reset_encoder_bn_running_stats()

        for batch_idx, samples in enumerate(trainloader_weight):

            inputs = samples['image'].to(device, non_blocking=True)
            target = {task: samples[task].to(
                device, non_blocking=True) for task in tasks}

            current_loss = 0
            counter = 0

            for task in tasks:
                # many images don't have human parts annotations, skip those
                uniq = torch.unique(target[task])
                if len(uniq) == 1 and uniq[0] == 255:
                    continue

                optimizer_weight.zero_grad()

                output = model(inputs, task=task)
                tot_loss = loss(
                    output, target, task=task, omit_resource=True)
                tot_loss.backward()

                optimizer_weight.step()

                current_loss += tot_loss.item()
                counter += 1

            if (batch_idx + 1) % 100 == 0:
                n_iter = (epoch - 1) * iter_per_epoch + batch_idx + 1
                print('Train Iterations: {}, Loss: {:.4f}'.format(utils.progress(n_iter,
                                                                                 total_iter),
                                                                  current_loss / counter))
                writer.add_scalar(
                    'loss_current', current_loss / counter, n_iter)
                writer.add_scalar(
                    'arch_loss', arch_loss / max(1, arch_counter), n_iter)
                writer.add_scalar('gumbel_temp', model.gumbel_temp, n_iter)
                for name, param in model.named_arch_parameters():
                    writer.add_image(name, torch.nn.functional.softmax(
                        param.data, dim=-1), n_iter, dataformats='HW')

        # save model
        state = {
            'state_dict': model.state_dict(),
            'tasks': tasks,
            'epoch': epoch,
            'optimizer_weight': optimizer_weight.state_dict(),
            'optimizer_arch': optimizer_arch.state_dict(),
        }
        torch.save(state, Path(exp_dir) / 'checkpoint.pth')

    branch_config = model.get_branch_config()
    utils.write_json({'config': branch_config},
                     Path(exp_dir) / 'branch_config.json')


def train_branched(local_rank,
                   world_size,
                   device,
                   start_epoch,
                   max_epochs,
                   tasks,
                   trainloader,
                   model,
                   loss,
                   optimizer,
                   scheduler,
                   exp_dir):

    writer = SummaryWriter(log_dir=exp_dir) if local_rank == 0 else None

    iter_per_epoch = len(
        trainloader.dataset) // (trainloader.batch_size * max(1, world_size))
    total_iter = iter_per_epoch * max_epochs

    model.train()
    for epoch in range(start_epoch, max_epochs + 1):

        if world_size > 1:
            trainloader.sampler.set_epoch(epoch)

        for batch_idx, samples in enumerate(trainloader):

            inputs = samples['image'].to(device, non_blocking=True)
            target = {task: samples[task].to(
                device, non_blocking=True) for task in tasks}

            optimizer.zero_grad()

            output = model(inputs)
            tot_loss = loss(output, target)
            tot_loss.backward()

            optimizer.step()

            if (batch_idx + 1) % 100 == 0 and local_rank == 0:
                current_loss = tot_loss.item()
                n_iter = (epoch - 1) * iter_per_epoch + batch_idx + 1
                print('Train Iterations: {}, Loss: {}'.format(utils.progress(n_iter, total_iter),
                                                              current_loss))
                writer.add_scalar('loss_current', current_loss, n_iter)
                writer.add_scalar(
                    'learning_rate', optimizer.param_groups[0]['lr'], n_iter)

        scheduler.step()

        if local_rank == 0:
            # save model
            state = {
                'state_dict': model.state_dict(),
                'tasks': tasks,
                'branch_config': model.branch_config,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, Path(exp_dir) / 'checkpoint.pth')


@torch.no_grad()
def test_branched(device, tasks, testloader, model, metrics_dict, exp_dir):

    model.eval()

    # get resources
    sample = next(iter(testloader))
    height, width = sample['image'].shape[-2:]
    gflops = resources.compute_gflops(model, device=device,
                                      in_shape=(1, 3, height, width))
    params = resources.count_parameters(model)
    results = {
        'gmadds': gflops / 2.0,
        'mparams': params / 1e6
    }

    for idx, samples in enumerate(testloader):

        inputs = samples['image'].to(device, non_blocking=True)
        target = {task: samples[task].to(
            device, non_blocking=True) for task in tasks}
        im_size = tuple(x.item() for x in samples['meta']['im_size'])
        im_name = samples['meta']['image'][0]

        output = model(inputs)

        for task in tasks:

            uniq = torch.unique(target[task])
            if len(uniq) == 1 and uniq[0] == 255:
                continue

            ground_truth = torch.squeeze(target[task], dim=0).cpu().numpy()
            prediction = torch.squeeze(output[task], dim=0).cpu().numpy()

            # metrics want numpy array of format (H x W x C)
            ground_truth = ground_truth.transpose(1, 2, 0)
            prediction = prediction.transpose(1, 2, 0)

            metrics_dict[task].update(
                prediction, ground_truth, im_size, im_name)

        if (idx + 1) % 100 == 0:
            print('{} / {} images done.'.format(idx + 1, len(testloader)))

    for task in tasks:
        results['_'.join([task, metrics_dict[task].__class__.__name__])
                ] = metrics_dict[task].get_score()
    utils.write_json(results, Path(exp_dir) / 'eval.json')
