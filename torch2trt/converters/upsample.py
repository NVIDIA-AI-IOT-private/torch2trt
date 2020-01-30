from torch2trt.torch2trt import *                                 
from torch2trt.module_test import add_module_test
                                                   
@tensorrt_converter('torch.nn.functional.interpolate')
def convert_interpolate(ctx):                                     
    #parse args                     
    input = get_arg(ctx, 'input', pos=0, default=None) 
    size = get_arg(ctx, 'kernel_size', pos=1, default=None)
    scale_factor=get_arg(ctx, 'scale_factor', pos=2, default=None)
    mode = get_arg(ctx, 'mode', pos=3, default='nearest')
    align_corners = get_arg(ctx, 'align_corners', pos=4, default=None)

    input_dim = input.dim() - 2
    
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    layer = ctx.network.add_resize(input=input_trt)

    shape = size
    if shape != None:
        if isinstance(shape, list):
           shape  = (1,) + tuple(shape)
        else:
            shape = (1,) + (shape,) * input_dim
        layer.shape = shape

    scales = scale_factor
    if scales != None:
        if not isinstance(scales, tuple):
            scales = (1,) + (scales,) * input_dim
        layer.scales = scales

    resize_mode = mode
    if resize_mode.lower() in ["linear","bilinear","trilinear"]:
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode=trt.ResizeMode.NEAREST

    if align_corners != None:
        layer.align_corners = align_corners

    output._trt = layer.get_output(0)

@add_module_test(torch.float32, torch.device('cuda'), [(1,2,12,12)])
def test_nearest_mode():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,12,12)])
def test_bilinear_mode():
    return torch.nn.Upsample(scale_factor=3, mode="bilinear",align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,3,12,12)])
def test_align_corner():
    return torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,5,13,13)])
def test_bilinear_mode_odd_input_shape():
    return torch.nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,4,12,12)])
def test_size_parameter():
    return torch.nn.Upsample(size=3,mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,2,13,13)])
def test_size_parameter_odd_input():
    return torch.nn.Upsample(size=[6,3],mode="nearest")


@add_module_test(torch.float32, torch.device('cuda'), [(1,1,12,12,12)])
def test_nearest_mode_3d():
    return torch.nn.Upsample(scale_factor=2, mode="nearest")

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,12,12,12)])
def test_bilinear_mode_3d():
    return torch.nn.Upsample(scale_factor=3, mode="trilinear",align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,12,12,12)])
def test_align_corner_3d():
    return torch.nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,13,13,13)])
def test_bilinear_mode_odd_input_shape_3d():
    return torch.nn.Upsample(scale_factor=2, mode="trilinear",align_corners=False)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,12,12,12)])
def test_size_parameter_3d():
    return torch.nn.Upsample(size=3,mode="trilinear", align_corners=True)

@add_module_test(torch.float32, torch.device('cuda'), [(1,1,13,13,13)])
def test_size_parameter_odd_input_3d():
    return torch.nn.Upsample(size=[6,3,3],mode="trilinear", align_corners=False)