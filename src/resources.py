# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/astmt.
#
import torch


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def compute_gflops(net, device, in_shape=(1, 3, 224, 224)):

    assert in_shape[0] == 1, 'can only compute FLOPS for batchsize=1'
    net = add_flops_counting_methods(net)
    net.eval()
    net.start_flops_count()

    with torch.no_grad():
        cache_inputs = torch.rand(in_shape).to(device)
        _ = net(cache_inputs)
        gflops = net.compute_average_flops_cost() / 1e9

    net.stop_flops_count()
    net = remove_flops_counting_methods(net)

    return gflops


def add_flops_counting_methods(net_main_module):
    """Adds flops counting functions to an existing model. After that
    the flops count should be activated and the model should be run on an input
    image.

    Example:

    fcn = add_flops_counting_methods(fcn)
    fcn.start_flops_count()

    _ = fcn(batch)
    fcn.compute_average_flops_cost() / 1e9  # Result in GFLOPs per image

    fcn.stop_flops_count()
    fcn = remove_flops_counting_methods(fcn)

    Attention: we are counting multiply-adds as two flops in this work, because in
    most resnet models convolutions are bias-free (BN layers act as bias there)
    and it makes sense to count muliply and add as separate flops therefore.
    This is consistent with most modern benchmarks. For example in
    "Spatially Adaptive Computation Time for Residual Networks" by Figurnov et al
    multiply-add was counted as two flops.

    The module works with registered hook functions which
    are being called each time the respective layer is executed.

    Parameters
    ----------
    net_main_module : torch.nn.Module
        Main module containing network

    Returns
    -------
    net_main_module : torch.nn.Module
        Updated main module with new methods/attributes that are used
        to compute flops.
    """

    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(
        net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(
        net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(
        net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
        net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def remove_flops_counting_methods(net_main_module):
    del net_main_module.start_flops_count
    del net_main_module.stop_flops_count
    del net_main_module.reset_flops_count
    del net_main_module.compute_average_flops_cost
    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.
    """
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
    return flops_sum


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.
    """
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    """
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.
    """
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions


def ignore_flops_counter_hook(module, inp, out):
    module.__flops__ += 0


def linear_flops_counter_hook(module, inp, out):
    out = out[0]
    total_mul = module.in_features
    total_add = module.in_features - 1
    total_add += 1 if module.bias is not None else 0
    num_elements = out.numel()
    module.__flops__ += (total_mul + total_add) * num_elements


def pool_flops_counter_hook(module, inp, out):
    if hasattr(module, "kernel_size"):
        out = out[0]
        out_c, out_h, out_w = out.shape
        if isinstance(module.kernel_size, (tuple, list)):
            kernel_height, kernel_width = module.kernel_size
        else:
            kernel_height, kernel_width = module.kernel_size, module.kernel_size
        total_flops = kernel_height * kernel_width * out_h * out_w * out_c
    else:
        inp = inp[0]
        total_flops = inp.numel()

    module.__flops__ += total_flops


def deconv_flops_counter_hook(module, inp, out):
    inp = inp[0]
    in_h, in_w = inp.shape[1:]

    kernel_height, kernel_width = module.kernel_size
    in_c = module.in_channels
    out_c = module.out_channels
    groups = module.groups

    kernel_mul = kernel_height * kernel_width * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * in_h * in_w * (out_c // groups)
    kernel_add_group = kernel_add * in_h * in_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    overall_flops = total_mul + total_add

    module.__flops__ += overall_flops


def conv_flops_counter_hook(module, inp, out):
    out = out[0]
    out_h, out_w = out.shape[1:]

    kernel_height, kernel_width = module.kernel_size
    in_c = module.in_channels
    out_c = module.out_channels
    groups = module.groups

    kernel_mul = kernel_height * kernel_width * (in_c // groups)
    kernel_add = kernel_mul - 1 + (0 if module.bias is None else 1)

    kernel_mul_group = kernel_mul * out_h * out_w * (out_c // groups)
    kernel_add_group = kernel_add * out_h * out_w * (out_c // groups)

    total_mul = kernel_mul_group * groups
    total_add = kernel_add_group * groups

    overall_flops = total_mul + total_add

    module.__flops__ += overall_flops


MODULES_MAPPING = {
    # convolutions
    torch.nn.Conv2d: conv_flops_counter_hook,
    # activations
    torch.nn.ReLU: ignore_flops_counter_hook,
    torch.nn.ReLU6: ignore_flops_counter_hook,
    # poolings
    torch.nn.MaxPool2d: pool_flops_counter_hook,
    torch.nn.AvgPool2d: pool_flops_counter_hook,
    torch.nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    torch.nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    # BNs
    torch.nn.BatchNorm2d: ignore_flops_counter_hook,
    # FC
    torch.nn.Linear: linear_flops_counter_hook,
    # deconvolution
    torch.nn.ConvTranspose2d: deconv_flops_counter_hook,
    # misc
    torch.nn.Dropout: ignore_flops_counter_hook,
    torch.nn.Identity: ignore_flops_counter_hook,
}


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING:
        return True
    return False


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return
        handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
        if hasattr(module, '__flops__'):
            del module.__flops__
