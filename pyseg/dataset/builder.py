
import logging

from .cityscapes import build_cityloader

logger = logging.getLogger('global')

def get_loader(cfg):
    cfg_dataset = cfg['dataset']
    if cfg_dataset['type'] == 'cityscapes':
        trainloader = build_cityloader('train', cfg)
        valloader = build_cityloader('val', cfg)
    else:
        raise NotImplementedError("dataset type {} is not supported".format(cfg_dataset))
    logger.info('Get loader Done...')
 
    return trainloader, valloader
