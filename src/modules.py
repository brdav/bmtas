import copy
import torch
import torch.nn as nn


def activation_func(activation):
    return nn.ModuleDict({
        'relu': nn.ReLU(inplace=True),
        'relu6': nn.ReLU6(inplace=True),
        'none': nn.Identity()
    })[activation]


class MyDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name == 'module':
            return nn.parallel.DistributedDataParallel.__getattr__(self, name)
        return getattr(self.module, name)


class BranchedLayer(nn.Module):

    def __init__(self, operation, mapping):
        super().__init__()
        self.mapping = mapping
        self.path = nn.ModuleDict()
        for ind_list in self.mapping.values():
            for ind in ind_list:
                self.path[str(ind)] = copy.deepcopy(operation)

    def forward(self, x):
        out = {}
        for branch_k in self.mapping.keys():
            for out_branch in self.mapping[branch_k]:
                out[out_branch] = self.path[str(out_branch)](x[branch_k])
        return out


class SupernetLayer(nn.Module):

    def __init__(self, operation, n_ops):
        super().__init__()
        self.path = nn.ModuleList()
        for _ in range(n_ops):
            self.path.append(copy.deepcopy(operation))

    def forward(self, x, op_weights):
        out = sum(op_weights[i] * op(x) for i, op in enumerate(self.path))
        return out


class GumbelSoftmax(nn.Module):

    def __init__(self, dim=None, hard=False):
        super().__init__()
        self.hard = hard
        self.dim = dim

    def forward(self, logits, temperature):
        # known issues with gumbel_softmax for older pytorch versions:
        # https://github.com/pytorch/pytorch/issues/22442
        # https://github.com/pytorch/pytorch/pull/20179
        eps = 1e-10
        gumbels = -(torch.empty_like(logits).exponential_() +
                    eps).log()  # ~Gumbel(0,1)
        # ~Gumbel(logits,temperature)
        gumbels = (logits + gumbels) / temperature
        y_soft = gumbels.softmax(self.dim)

        if self.hard:
            # Straight through.
            index = y_soft.max(self.dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(self.dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparameterization trick.
            ret = y_soft
        return ret


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, affine=True, activation='relu'):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_channels, affine=affine),
            activation_func(activation)
        )

    def forward(self, x):
        return self.op(x)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion, kernel_size=3, groups=1,
                 dilation=1, skip_connect=True, final_affine=True, activation='relu'):
        super().__init__()
        assert kernel_size in [1, 3, 5, 7]
        assert stride in [1, 2]
        if stride == 2 and dilation > 1:
            stride = 1
            dilation = dilation // 2
        padding = int((kernel_size - 1) * dilation / 2)
        hidden_dim = round(in_channels * expansion)

        self.chain = []
        if expansion != 1:
            self.chain.append(ConvBNReLU(in_channels,
                                         hidden_dim,
                                         1,
                                         stride=1,
                                         padding=0,
                                         groups=groups,
                                         activation=activation))
        self.chain.extend([
            ConvBNReLU(hidden_dim,
                       hidden_dim,
                       kernel_size,
                       stride=stride,
                       padding=padding,
                       groups=hidden_dim,
                       dilation=dilation,
                       activation=activation),
            ConvBNReLU(hidden_dim,
                       out_channels,
                       1,
                       stride=1,
                       padding=0,
                       groups=groups,
                       affine=final_affine,
                       activation='none')])
        self.chain = nn.Sequential(*self.chain)

        if skip_connect and in_channels == out_channels and stride == 1:
            self.res_flag = True
        else:
            self.res_flag = False

    def forward(self, x):
        identity = x
        out = self.chain(x)
        if self.res_flag:
            out += identity
        return out


class RASPP(nn.Module):

    def __init__(self, in_channels, out_channels, activation='relu6',
                 drop_rate=0, final_affine=True):
        super().__init__()

        self.drop_rate = drop_rate

        # 1x1 convolution
        self.aspp_branch_1 = ConvBNReLU(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        activation=activation)
        # image pooling feature
        self.aspp_branch_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1,
                       activation=activation))

        self.aspp_projection = ConvBNReLU(2 * out_channels, out_channels, kernel_size=1, stride=1,
                                          activation=activation, affine=final_affine)

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        branch_1 = self.aspp_branch_1(x)
        branch_2 = self.aspp_branch_2(x)
        branch_2 = nn.functional.interpolate(input=branch_2, size=(h, w),
                                             mode='bilinear', align_corners=False)

        # Concatenate the parallel streams
        out = torch.cat([branch_1, branch_2], dim=1)

        if self.drop_rate > 0:
            out = nn.functional.dropout(
                out, p=self.drop_rate, training=self.training)
        out = self.aspp_projection(out)

        return out


class DeepLabV3PlusDecoder(nn.Module):

    def __init__(self,
                 num_outputs,
                 in_channels_low=256,
                 out_f_classifier=256,
                 use_separable=False,
                 activation='relu'):

        super().__init__()

        projected_filters = 48

        self.low_level_reduce = ConvBNReLU(in_channels_low,
                                           projected_filters,
                                           kernel_size=1,
                                           stride=1,
                                           activation=activation)

        if use_separable:
            self.conv_1 = nn.Sequential(ConvBNReLU(out_f_classifier + projected_filters,
                                                   out_f_classifier + projected_filters,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   groups=out_f_classifier + projected_filters,
                                                   activation=activation),
                                        ConvBNReLU(out_f_classifier + projected_filters,
                                                   out_f_classifier,
                                                   kernel_size=1,
                                                   stride=1,
                                                   activation=activation))
            self.conv_2 = nn.Sequential(ConvBNReLU(out_f_classifier,
                                                   out_f_classifier,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   groups=out_f_classifier,
                                                   activation=activation),
                                        ConvBNReLU(out_f_classifier,
                                                   out_f_classifier,
                                                   kernel_size=1,
                                                   stride=1,
                                                   activation=activation))
        else:
            self.conv_1 = ConvBNReLU(out_f_classifier + projected_filters,
                                     out_f_classifier,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     activation=activation)
            self.conv_2 = ConvBNReLU(out_f_classifier,
                                     out_f_classifier,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     activation=activation)

        self.conv_logits = nn.Conv2d(out_f_classifier,
                                     num_outputs,
                                     kernel_size=1,
                                     bias=True)

    def forward(self, x, x_low, input_shape):
        decoder_height, decoder_width = x_low.shape[-2:]
        x = nn.functional.interpolate(x,
                                      size=(decoder_height, decoder_width),
                                      mode='bilinear',
                                      align_corners=False)
        x_low = self.low_level_reduce(x_low)
        x = torch.cat((x_low, x), dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_logits(x)
        x = nn.functional.interpolate(x,
                                      size=input_shape,
                                      mode='bilinear',
                                      align_corners=False)
        return x
