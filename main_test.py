import argparse
from pathlib import Path
import torch

from src import dataset
from src import models
from src import tools
from src import metrics


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True,
                    help='path to the model to test')
parser.add_argument('--data_root', default='./PASCAL_MT',
                    type=str, help='PASCAL-Context dataset root dir')


def main(opt):

    model_path = opt.model_path
    data_root = opt.data_root

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    # set up dataset
    print('setting up dataloader...')
    testset = dataset.PASCALContext(
        data_dir=data_root, split='val', transforms=True, tasks=checkpoint['tasks'], download=False)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=1, shuffle=False, pin_memory=True)

    # build model architecture and load weights
    print('building the model...')
    model = models.BranchMobileNetV2(
        checkpoint['tasks'], branch_config=checkpoint['branch_config'])
    model = model.to(device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        state_dict = {k.replace('module.', ''): v
                      for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

    # get metrics
    print('build metrics...')
    exp_dir = Path(model_path).parent.parent / 'test'
    exp_dir.mkdir(parents=True, exist_ok=True)
    edge_save_dir = Path(exp_dir) / 'edge'
    edge_save_dir.mkdir(parents=True, exist_ok=True)
    metrics_dict = {
        'semseg': metrics.MeanIoU(task='semseg', n_classes=21),
        'human_parts': metrics.MeanIoU(task='human_parts', n_classes=7),
        'sal': metrics.ThresholdedMeanIoU(task='sal', thresholds=[x / 20. for x in range(4, 19)]),
        'normals': metrics.MeanErrorInAngle(task='normals'),
        'edge': metrics.SavePrediction(task='edge', save_dir=edge_save_dir)
    }

    print('testing the network...')
    tools.test_branched(device,
                        checkpoint['tasks'],
                        testloader,
                        model,
                        metrics_dict,
                        exp_dir)

    print('testing finished!')


if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)
