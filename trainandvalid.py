import argparse
import collections

import model_metric.loss as module_loss
import model_metric.metric as module_metric
import numpy as np
import torch
from parse_config import ConfigParser

import dataloader.dataset as module_data
from model import AFStroke as module_arch
from trainer import Trainer

SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    print(logger)
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_dataset(valid=True)
    test_data_loader = data_loader.split_dataset(test=True)
    adj = data_loader.get_sparse_adj()
    node_num = data_loader.get_node_num()
    type_num = data_loader.get_type_num()

    # build model architecture, then print to console
    model = module_arch(node_num=node_num,
                        type_num=type_num,

                        adj=adj,
                        number_of_heads=config['arch']['args']['number_of_heads'],
                        emb_dim=config['arch']['args']['emb_dim'],
                        gcn_layersize=config['arch']['args']['gcn_layersize'],
                        dropout=config['arch']['args']['dropout'],
                        mlpdropout=config['arch']['args']['mlpdropout'],
                        mlpbn=config['arch']['args']['mlpbn'],
                        alpha=config['arch']['args']['alpha'],
                        gcn=config['arch']['args']['gcn'],
                        num_of_direction=config['arch']['args']['num_of_direction']
                        )
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    weight_loss_0 = config['weight_loss_0']
    weight_loss_1 = config['weight_loss_1']
    weight_loss_pos = config['weight_loss_pos']
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      weight_loss_0=weight_loss_0,
                      weight_loss_1=weight_loss_1,
                      weight_loss_pos=weight_loss_pos)

    trainer.train()

    """Test."""
    logger = config.get_logger('test')
    logger.info(model)
    test_metrics = [getattr(module_metric, met) for met in config['metrics']]

    # load best checkpoint
    resume = str(config.save_dir / 'model_best.pth')
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    test_output = trainer.test()
    log = {'loss': test_output['total_loss'] / test_output['n_samples']}
    log.update({
        met.__name__: test_output['total_metrics'][i].item() / test_output['n_samples'] \
        for i, met in enumerate(test_metrics)})
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default="/config/config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
