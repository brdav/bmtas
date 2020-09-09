import os
import argparse
from pathlib import Path
import torch

from src import dataset
from src import models
from src import modules
from src import criterion
from src import tools
from src import utils


parser = argparse.ArgumentParser()
parser.add_argument('--configuration', type=str, required=True,
                    help='path to the branching configuration file')
parser.add_argument('--data_root', default='./PASCAL_MT',
                    type=str, help='PASCAL-Context dataset root dir')
parser.add_argument('--tasks', default='semseg,human_parts,sal,normals,edge', type=str,
                    help='tasks to train, comma-separated, order matters!')
parser.add_argument('--resume_path', type=str,
                    help='path to model to resume')

torch.backends.cudnn.benchmark = True


def main(local_rank, world_size, opt):

    configuration = opt.configuration
    data_root = opt.data_root
    tasks = opt.tasks.split(',')
    resume_path = opt.resume_path

    printf = utils.distributed_print(local_rank)
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=local_rank
        )
    device = torch.device('cuda:{}'.format(local_rank)
                          if world_size > 0 else 'cpu')

    # set up dataloader
    printf('setting up dataloader...')
    trainset = dataset.PASCALContext(
        data_dir=data_root, split='train', transforms=True, tasks=tasks, download=False)
    if world_size > 1:
        assert (16 % world_size) == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
    else:
        train_sampler = None
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=16 // max(1, world_size),
        num_workers=int((4 + max(1, world_size) - 1) / max(1, world_size)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True
    )

    # build model architecture
    printf('building the model and loss...')
    branch_config = utils.read_json(configuration)['config']
    model = models.BranchMobileNetV2(
        tasks, branch_config=branch_config)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    loss = criterion.WeightedSumLoss(tasks)
    model = model.to(device)
    loss = loss.to(device)
    if world_size > 1:
        model = modules.MyDataParallel(model,
                                       device_ids=[
                                           local_rank],
                                       output_device=local_rank)

    # build optimization tools
    printf('building optimization tools...')
    max_epochs = 130  # around 40000 iterations with batchsize 16
    optimizer = torch.optim.SGD(
        lr=0.005, momentum=0.9, weight_decay=1e-4, params=model.parameters())
    # poly learning rate schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda ep: (1 - float(ep) / max_epochs) ** 0.9)

    # in case we resume...
    start_epoch = 1
    if resume_path is not None:
        printf('resuming saved model...')
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    printf('setup complete, start training...')
    exp_dir = Path(configuration).parent.parent / 'branched'
    if local_rank == 0:
        exp_dir.mkdir(parents=True, exist_ok=True)

    tools.train_branched(local_rank,
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
                         exp_dir)

    printf('training finished!')


if __name__ == '__main__':
    opt = parser.parse_args()
    world_size = torch.cuda.device_count()  # only support training on one node
    if world_size > 1:
        torch.multiprocessing.spawn(
            main, nprocs=world_size, args=(world_size, opt))
    else:
        main(0, world_size, opt)
