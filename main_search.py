import argparse
import random
from pathlib import Path
from datetime import datetime
import torch

from src import dataset
from src import criterion
from src import models
from src import tools


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='./PASCAL_MT',
                    type=str, help='PASCAL-Context dataset root dir')
parser.add_argument('--tasks', default='semseg,human_parts,sal,normals,edge', type=str,
                    help='tasks to train, comma-separated')
parser.add_argument('--resource_loss_weight', default=0.05, type=float,
                    help='weight of resource loss')
parser.add_argument('--resume_path', type=str,
                    help='path to model to resume')

torch.backends.cudnn.benchmark = True


def main(opt):

    data_root = opt.data_root
    tasks = opt.tasks.split(',')
    resource_loss_weight = opt.resource_loss_weight
    resume_path = opt.resume_path

    # if available, we train on one GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set up dataloader
    print('setting up dataloaders...')
    trainset = dataset.PASCALContext(
        data_dir=data_root, split='train', transforms=True, tasks=tasks, use_resized=True)
    indices = list(range(len(trainset)))
    random.shuffle(indices)
    # split the dataset into 80% for training the weights and 20% for training the arch-params
    trainset_weight = torch.utils.data.Subset(
        trainset, indices[:int(0.8 * len(indices))])
    trainset_arch = torch.utils.data.Subset(
        trainset, indices[int(0.8 * len(indices)):])
    trainloader_weight = torch.utils.data.DataLoader(dataset=trainset_weight, batch_size=16,
                                                     shuffle=True, pin_memory=True,
                                                     drop_last=True, num_workers=4)
    trainloader_arch = torch.utils.data.DataLoader(dataset=trainset_arch, batch_size=16,
                                                   shuffle=True, pin_memory=True,
                                                   drop_last=True, num_workers=4)

    # build model architecture
    print('building the model and loss...')
    model = models.SuperMobileNetV2(tasks)
    loss = criterion.WeightedSumLoss(
        tasks, resource_loss_weight=resource_loss_weight, model=model)
    model = model.to(device)
    loss = loss.to(device)

    # build optimization tools
    print('building optimization tools...')
    max_epochs = 160  # around 40000 weight update iterations at batchsize 16
    optimizer_weight = torch.optim.Adam(
        lr=0.001, weight_decay=1e-4, params=model.weight_parameters())
    optimizer_arch = torch.optim.Adam(
        lr=0.01, weight_decay=5e-5, params=model.arch_parameters())

    # in case we resume...
    start_epoch = 1
    if resume_path is not None:
        print('resuming model...')
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer_weight.load_state_dict(checkpoint['optimizer_weight'])
        optimizer_arch.load_state_dict(checkpoint['optimizer_arch'])
        start_epoch = checkpoint['epoch'] + 1

    # start training!
    print('setup complete, start training...')
    exp_dir = Path('./exp_{}_{}_{}'.format('-'.join(tasks), resource_loss_weight,
                                           datetime.now().strftime(r'%m-%d-%H-%M-%S'))) / 'search'
    exp_dir.mkdir(parents=True, exist_ok=True)
    tools.train_search(device,
                       start_epoch,
                       max_epochs,
                       tasks,
                       trainloader_weight,
                       trainloader_arch,
                       model,
                       loss,
                       optimizer_weight,
                       optimizer_arch,
                       exp_dir)
    print('search finished!')


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)
