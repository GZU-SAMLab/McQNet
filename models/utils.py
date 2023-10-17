from . import model_dict


def create_model(name):
    """create model by name"""
    if name.endswith('v2') or name.endswith('v3'):
        model = model_dict[name](num_classes=n_cls)
    elif name.startswith('resnet50'):
        print('use imagenet-style resnet50')
        model = model_dict[name](num_classes=n_cls)
    elif name.startswith('resnet') or name.startswith('seresnet'):
        model = model_dict[name](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls)
    elif name.startswith('wrn'):
        model = model_dict[name](num_classes=n_cls)
    elif name.startswith('convnet'):
        model = model_dict[name](num_classes=n_cls)
    else:
        raise NotImplementedError('model {} not supported in dataset {}:'.format(name, dataset))
    return model

