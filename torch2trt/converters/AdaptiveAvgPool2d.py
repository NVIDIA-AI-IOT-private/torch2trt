from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.nn.AdaptiveAvgPool2d.forward')
def convert_AdaptiveAvgPool2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    # Determine the target output size
    target_output_size = module.output_size
    
    if not isinstance(target_output_size, tuple):
        target_output_size = (target_output_size, ) * 2
        
    # Handle cases where target output size has None values
    if None in target_output_size:
        target_output_shape = tuple(output.shape)
        target_output_shape = target_output_shape[2:]
        new_output_size = []
        for size, shape in zip(target_output_size, target_output_shape):
            if size is None:
                new_output_size.append(shape)
            else:
                new_output_size.append(size)

        target_output_size = tuple(new_output_size)
    
    # Calculate stride and kernel size
    stride = (input_trt.shape[-2] // target_output_size[-2], input_trt.shape[-1] // target_output_size[-1])
    kernel_size = stride
    
    # Create pooling layer
    layer = ctx.network.add_pooling(
        input=input_trt, type=trt.PoolingType.AVERAGE, window_size=kernel_size)
    layer.stride = stride

    # Set _trt attribute for output
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_1x1():
    return torch.nn.AdaptiveAvgPool2d((1, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_2x2():
    return torch.nn.AdaptiveAvgPool2d((2, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_3x3():
    return torch.nn.AdaptiveAvgPool2d((3, 3))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_None_1():
    return torch.nn.AdaptiveAvgPool2d((None, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_1_None():
    return torch.nn.AdaptiveAvgPool2d((1, None))