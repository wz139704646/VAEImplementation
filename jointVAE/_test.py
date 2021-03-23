import os


def set_basic_config():
    """set basic test configuration"""
    conf = {}
    conf['checkpoint_path'] = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/mnist/t1616112033-epoch50_z20_temp0.67_gamma30_30.pth.tar')

    return conf