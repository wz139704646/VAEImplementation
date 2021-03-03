import os


def set_basic_config():
    """set basic test configuration"""
    conf = {}
    conf['checkpoint_path'] = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/mnist/t1610177574-epoch150_z15_beta4.pth.tar')

    return conf