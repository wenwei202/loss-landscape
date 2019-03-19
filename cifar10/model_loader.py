import os
import torch, torchvision
import cifar10.models.vgg as vgg
import cifar10.models.resnet as resnet
import cifar10.models.densenet as densenet
import cifar10.models.resnetv2 as resnetv2

# map between model name and function
models = {
    'vgg9'                  : vgg.VGG9,
    'densenet121'           : densenet.DenseNet121,
    'resnet18'              : resnet.ResNet18,
    'resnet18_noshort'      : resnet.ResNet18_noshort,
    'resnet34'              : resnet.ResNet34,
    'resnet34_noshort'      : resnet.ResNet34_noshort,
    'resnet50'              : resnet.ResNet50,
    'resnet50_noshort'      : resnet.ResNet50_noshort,
    'resnet101'             : resnet.ResNet101,
    'resnet101_noshort'     : resnet.ResNet101_noshort,
    'resnet152'             : resnet.ResNet152,
    'resnet152_noshort'     : resnet.ResNet152_noshort,
    'resnet20'              : resnet.ResNet20,
    'resnet20_noshort'      : resnet.ResNet20_noshort,
    'resnet32_noshort'      : resnet.ResNet32_noshort,
    'resnet44_noshort'      : resnet.ResNet44_noshort,
    'resnet50_16_noshort'   : resnet.ResNet50_16_noshort,
    'resnet56'              : resnet.ResNet56,
    'resnet56_noshort'      : resnet.ResNet56_noshort,
    'resnet110'             : resnet.ResNet110,
    'resnet110_noshort'     : resnet.ResNet110_noshort,
    'wrn56_2'               : resnet.WRN56_2,
    'wrn56_2_noshort'       : resnet.WRN56_2_noshort,
    'wrn56_4'               : resnet.WRN56_4,
    'wrn56_4_noshort'       : resnet.WRN56_4_noshort,
    'wrn56_8'               : resnet.WRN56_8,
    'wrn56_8_noshort'       : resnet.WRN56_8_noshort,
    'wrn110_2_noshort'      : resnet.WRN110_2_noshort,
    'wrn110_4_noshort'      : resnet.WRN110_4_noshort,
    'CifarResNetBasic'      : resnetv2.CifarResNetBasic,
    'ResNetBasic'           : resnetv2.ResNetBasic,
    'CifarPlainNet'         : resnetv2.CifarPlainNet,
    'CifarSwitchResNetBasic': resnetv2.CifarSwitchResNetBasic,
    'CifarPlainNoBNNet'     : resnetv2.CifarPlainNoBNNet,
    'PlainNoBNNet'          : resnetv2.PlainNoBNNet,
    'ResNetBottleneck'      : resnetv2.ResNetBottleneck,
}

def zeroinit_nonexisting_params(model, state_dict):
    for n, p in model.named_parameters():
        if n not in state_dict:
            print('%s are set to zeros' % n)
            p.data.zero_()

def load(model_name, model_file=None, data_parallel=False, num_blocks=None, strict=True):
    if num_blocks is not None:
        net = models[model_name](num_blocks)
    else:
        net = models[model_name]()

    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    if model_file:
        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location=lambda storage, loc: storage)
        stored_states = None
        if 'state_dict' in stored.keys():
            stored_states = stored['state_dict']
        elif 'net' in stored.keys():
            stored_states = stored['net']
        else:
            stored_states = stored
        net.load_state_dict(stored_states, strict=strict)
        if not strict:
            zeroinit_nonexisting_params(net, stored_states)

    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net
