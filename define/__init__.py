import importlib
from os import path as osp

from utils.misc import scandir


arch_folder = '/data/cyclic-restormer'
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('cyclic.py') or v.endswith('cyclic2.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'{file_name}')
    for file_name in arch_filenames
]


def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class: Instantiated class.
    """

    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)


def define_network(opt):
    network_type = opt.pop('type')
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net
