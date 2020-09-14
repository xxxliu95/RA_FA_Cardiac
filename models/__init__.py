from .sdnet_ada import *
from .weight_init import *
import sys


def get_model(name, params):
    if name == 'sdnet':
        return SDNet(params['width'], params['height'], params['num_classes'], params['ndf'], params['z_length'], params['norm'], params['upsample'], params['decoder_type'], params['anatomy_out_channels'], params['num_mask_channels'])
    else:
        print("Could not find the requested model ({})".format(name), file=sys.stderr)