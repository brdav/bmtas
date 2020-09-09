from pathlib import Path
import torch
import torch.nn as nn

from src import modules
from src import utils
from src import resources


TASK_CHANNEL_MAPPING = {
    'semseg': 21,
    'human_parts': 7,
    'sal': 1,
    'normals': 3,
    'edge': 1
}


MODEL_URL = 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'


class BranchMobileNetV2(nn.Module):
    """
    Branched multi-task network based on a MobileNetV2 backbone, R-ASPP module and
    DeepLabv3+ head.
    branch_config: Array (or nested list) of shape (n_layers X n_tasks) expressing which
    blocks are sampled for each task at each layer. Example (n_layers=6, n_tasks=4):
    branch_config = [
        [0, 0, 0, 0],
        [2, 2, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 2, 3],
        [0, 1, 2, 3],
        [1, 1, 3, 3]
    ]
    This array determines the branching configuration.
    """

    def __init__(self, tasks, branch_config):

        super().__init__()
        self.tasks = tasks
        self.branch_config = branch_config

        self.stem = modules.ConvBNReLU(in_channels=3,
                                       out_channels=32,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       activation='relu6')

        mappings = self._get_branch_mappings()
        blocks = [
            modules.InvertedResidual(
                32, 16, 1, 1, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                16, 24, 2, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                24, 24, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                24, 32, 2, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                32, 32, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                32, 32, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                32, 64, 2, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                64, 64, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                64, 64, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                64, 64, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                64, 96, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                96, 96, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                96, 96, 1, 6, dilation=1, activation='relu6'),
            modules.InvertedResidual(
                96, 160, 2, 6, dilation=2, activation='relu6'),
            modules.InvertedResidual(
                160, 160, 1, 6, dilation=2, activation='relu6'),
            modules.InvertedResidual(
                160, 160, 1, 6, dilation=2, activation='relu6'),
            modules.InvertedResidual(
                160, 320, 1, 6, dilation=2, activation='relu6'),
            modules.RASPP(320, 128, activation='relu6', drop_rate=0.1)
        ]
        self.encoder = nn.Sequential(*[modules.BranchedLayer(
            bl, ma) for bl, ma in zip(blocks, mappings)])

        self.decoder = nn.ModuleDict({
            task: modules.DeepLabV3PlusDecoder(
                in_channels_low=24,
                out_f_classifier=128,
                use_separable=True,
                activation='relu6',
                num_outputs=TASK_CHANNEL_MAPPING[task])
            for task in self.tasks})

        self._initialize_weights()
        self._load_imagenet_weights()

    def _get_branch_mappings(self):
        """
        Calculates branch mappings from the branch_config. `mappings` is a list of dicts mapping
        the index of an input branch to indices of output branches. For example:
        mappings = [
            {0: [0]},
            {0: [0, 1]},
            {0: [0, 2], 1: [1, 3]},
            {0: [1], 1: [2], 2: [3], 3: [0]}
        ]
        """

        def get_partition(layer_config):
            partition = []
            for t in range(len(self.tasks)):
                s = {i for i, x in enumerate(layer_config) if x == t}
                if not len(s) == 0:
                    partition.append(s)
            return partition

        def make_refinement(partition, ancestor):
            """ make `partition` a refinement of `ancestor` """
            refinement = []
            for part_1 in partition:
                for part_2 in ancestor:
                    inter = part_1.intersection(part_2)
                    if not len(inter) == 0:
                        refinement.append(inter)
            return refinement

        task_grouping = [set(range(len(self.tasks))), ]

        mappings = []
        for layer_idx, layer_config in enumerate(self.branch_config):

            partition = get_partition(layer_config)
            refinement = make_refinement(partition, task_grouping)

            out_dict = {}
            for prev_idx, prev in enumerate(task_grouping):
                out_dict[prev_idx] = []
                for curr_idx, curr in enumerate(refinement):
                    if curr.issubset(prev):
                        out_dict[prev_idx].append(curr_idx)

            task_grouping = refinement
            mappings.append(out_dict)

            if layer_idx == 2:
                self.x_low_decoder_mapping = {
                    task: [t_idx in gr for gr in task_grouping].index(True)
                    for t_idx, task in enumerate(self.tasks)
                }

        self.x_decoder_mapping = {
            task: [t_idx in gr for gr in task_grouping].index(True)
            for t_idx, task in enumerate(self.tasks)
        }
        return mappings

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.stem(x)

        out = {0: x}
        for i, layer in enumerate(self.encoder):
            out = layer(out)
            if i == 2:
                out_low = {b: out[b].clone() for b in out.keys()}

        output = {task: self.decoder[task](out[self.x_decoder_mapping[task]],
                                           out_low[self.x_low_decoder_mapping[task]],
                                           input_shape)
                  for task in self.tasks}
        return output

    def _initialize_weights(self):
        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'logits' in name:
                    # initialize final prediction layer with fixed std
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def _load_imagenet_weights(self):
        # we are using pretrained weights from torchvision.models:
        source_state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URL, progress=False)
        target_state_dict = self.state_dict()

        mapped_state_dict = {}
        for name_trg in target_state_dict:
            if name_trg.startswith('decoder') or name_trg.startswith('encoder.17'):
                continue  # can't load decoder and ASPP
            if name_trg.startswith('stem'):
                name_src = name_trg.replace('stem.op', 'features.0')
            # this is highly specific to the current naming
            elif name_trg.startswith('encoder'):
                parsed = name_trg.split('.')
                layer_nr = int(parsed[1])
                chain_nr = int(parsed[5])
                op_nr = int(parsed[7])
                del parsed[2:4]  # remove the path and its index
                parsed[0] = 'features'
                parsed[1] = str(layer_nr + 1)
                parsed[2] = 'conv'
                if chain_nr == 0 or (chain_nr == 1 and layer_nr != 0):
                    parsed[3] = str(chain_nr)
                    del parsed[4]
                else:
                    parsed[3] = str(chain_nr + op_nr)
                    del parsed[4:6]
                name_src = '.'.join(parsed)
            else:
                raise ValueError
            mapped_state_dict[name_trg] = source_state_dict[name_src]
        self.load_state_dict(mapped_state_dict, strict=False)


class SuperMobileNetV2(nn.Module):
    """
    Supergraph encompassing all possible branched multi-task networks, based on a MobileNetV2
    encoder, R-ASPP module and DeepLabv3+ head. The branch configuration distribution is
    parameterized with architecture parameters self.alphas and relaxed for optimization using
    a Gumbel-Softmax.
    """

    def __init__(self, tasks):

        super().__init__()
        self.tasks = tasks

        # First conv
        self.stem = modules.ConvBNReLU(in_channels=3,
                                       out_channels=32,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       activation='relu6')

        # Build encoder supergraph
        blocks = [
            modules.InvertedResidual(
                32, 16, 1, 1, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                16, 24, 2, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                24, 24, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                24, 32, 2, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                32, 32, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                32, 32, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                32, 64, 2, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                64, 64, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                64, 64, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                64, 64, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                64, 96, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                96, 96, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                96, 96, 1, 6, dilation=1, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                96, 160, 2, 6, dilation=2, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                160, 160, 1, 6, dilation=2, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                160, 160, 1, 6, dilation=2, activation='relu6', final_affine=False),
            modules.InvertedResidual(
                160, 320, 1, 6, dilation=2, activation='relu6', final_affine=False),
            modules.RASPP(320, 128, activation='relu6',
                          final_affine=False, drop_rate=0)
        ]
        self.encoder = nn.ModuleList(modules.SupernetLayer(
            bl, len(self.tasks)) for bl in blocks)

        # Build decoder
        self.decoder = nn.ModuleDict({
            task: modules.DeepLabV3PlusDecoder(
                in_channels_low=24,
                out_f_classifier=128,
                use_separable=True,
                activation='relu6',
                num_outputs=TASK_CHANNEL_MAPPING[task])
            for task in self.tasks})

        self.warmup_flag = False
        self.gumbel_temp = None
        self.gumbel_func = modules.GumbelSoftmax(dim=-1, hard=False)

        self._create_alphas()
        self._initialize_weights()
        self._load_imagenet_weights()

    def forward(self, x, task):
        input_shape = x.shape[-2:]
        t_idx = self.tasks.index(task)

        t_feat = self.stem(x)
        for idx, op in enumerate(self.encoder):
            if self.warmup_flag:
                op_weights = torch.eye(len(self.tasks), out=torch.empty_like(
                    x), requires_grad=False)[t_idx, :]
            else:
                op_weights = self.gumbel_func(
                    self.alphas[idx][t_idx, :], temperature=self.gumbel_temp)
            t_feat = op(t_feat, op_weights)
            if idx == 2:
                t_interm = t_feat.clone()

        res = {task: self.decoder[task](t_feat, t_interm, input_shape)}
        return res

    def _initialize_weights(self):
        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'logits' in name:
                    # initialize final prediction layer with fixed std
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def freeze_encoder_bn_running_stats(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

    def unfreeze_encoder_bn_running_stats(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = True

    def reset_encoder_bn_running_stats(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

    def _create_alphas(self):
        """ Create the architecture parameters of the supergraph. """
        n_tasks = len(self.tasks)
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.zeros(n_tasks, n_tasks)) for _ in self.encoder
        ])

    def _load_imagenet_weights(self):
        # we are using pretrained weights from torchvision.models:
        source_state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URL, progress=False)
        target_state_dict = self.state_dict()

        mapped_state_dict = {}
        for name_trg in target_state_dict:
            if name_trg.startswith('decoder') or name_trg.startswith('encoder.17'):
                continue  # can't load decoder and ASPP
            if name_trg.startswith('alphas'):
                continue
            if name_trg.startswith('stem'):
                name_src = name_trg.replace('stem.op', 'features.0')
            # this is highly specific to the current naming
            elif name_trg.startswith('encoder'):
                parsed = name_trg.split('.')
                layer_nr = int(parsed[1])
                chain_nr = int(parsed[5])
                op_nr = int(parsed[7])
                del parsed[2:4]  # remove the path and its index
                parsed[0] = 'features'
                parsed[1] = str(layer_nr + 1)
                parsed[2] = 'conv'
                if chain_nr == 0 or (chain_nr == 1 and layer_nr != 0):
                    parsed[3] = str(chain_nr)
                    del parsed[4]
                else:
                    parsed[3] = str(chain_nr + op_nr)
                    del parsed[4:6]
                name_src = '.'.join(parsed)
            else:
                raise ValueError
            mapped_state_dict[name_trg] = source_state_dict[name_src]
        self.load_state_dict(mapped_state_dict, strict=False)

    def weight_parameters(self):
        for name, param in self.named_weight_parameters():
            yield param

    def named_weight_parameters(self):
        return filter(lambda x: not x[0].startswith('alphas'),
                      self.named_parameters())

    def arch_parameters(self):
        for name, param in self.named_arch_parameters():
            yield param

    def named_arch_parameters(self):
        return filter(lambda x: x[0].startswith('alphas'),
                      self.named_parameters())

    def get_branch_config(self):
        n_blocks = len(self.alphas)
        n_tasks = len(self.tasks)
        branch_config = torch.empty(n_blocks, n_tasks, device='cpu')
        for b in range(n_blocks):
            alpha_probs = nn.functional.softmax(
                self.alphas[b], dim=-1).to('cpu').detach()
            for t in range(n_tasks):
                branch_config[b, t] = torch.argmax(alpha_probs[t, :])
        return branch_config.numpy().tolist()

    def calculate_flops_lut(self, file_name, input_size):
        shared_config = torch.zeros(18, len(self.tasks))
        model = BranchMobileNetV2(tasks=['semseg'],
                                  branch_config=shared_config)
        in_shape = (1, 3, input_size[0], input_size[1])
        n_blocks = len(model.encoder)
        flops = torch.zeros(n_blocks, device='cpu')

        model.eval()
        with torch.no_grad():
            for idx, m in enumerate(model.encoder):
                m = resources.add_flops_counting_methods(m)
                m.start_flops_count()
                cache_inputs = torch.rand(in_shape)
                _ = model(cache_inputs)
                block_flops = m.compute_average_flops_cost()
                m.stop_flops_count()
                flops[idx] = block_flops
        flops_dict = {'per_block_flops': flops.numpy().tolist()}
        del model

        # save the FLOPS to LUT
        utils.write_json(flops_dict, file_name)

        return flops_dict

    def get_flops(self):
        input_size = [512, 512]  # for PASCAL-Context

        filename = Path('flops_MobileNetV2_{}_{}.json'.format(
            input_size[0], input_size[1]))
        if filename.is_file():
            flops_dict = utils.read_json(filename)
        else:
            print('no LUT found, calculating FLOPS...')
            flops_dict = self.calculate_flops_lut(filename, input_size)
        return flops_dict['per_block_flops']
