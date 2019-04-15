import torch


def get_node_in_model(module, full_name):
    res = None
    splitted_name = full_name.split(".")
    desired_path = None
    if len(splitted_name) > 1:
        desired_path = splitted_name[0]
        name = ".".join(splitted_name[1:])
    elif len(splitted_name) == 1:
        name = splitted_name[0]
    else:
        return res

    for sub_layer, sub_module in module._modules.items():
        if sub_layer == name:
            res = sub_module
            break
        elif sub_layer == desired_path and \
                sub_module is not None and \
                len(sub_module._modules.items()) > 0:
            res = get_node_in_model(sub_module, name)
            if res is not None:
                break

    return res


def total_num_filters(modules):
    filters = 0

    if isinstance(modules, torch.nn.modules.conv.Conv2d):
        filters = filters + modules.out_channels
    else:
        if len(modules._modules.items()) > 0:
            for name, sub_module in modules._modules.items():
                if sub_module is not None:
                    filters = filters + total_num_filters(sub_module)

        else:
            if isinstance(modules, torch.nn.modules.conv.Conv2d):
                filters = filters + modules.out_channels
    return filters
