# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/22
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.
import argparse
import os

import torch
import torch.nn.functional as F

from onekey_core.core import create_model
from onekey_core.core import create_standard_image_transformer
from onekey_algo.datasets.image_loader import default_loader
from onekey_algo.utils.MultiProcess import MultiProcess

parser = argparse.ArgumentParser(description='PyTorch Classification Inference')

parser.add_argument('-c', '--config', dest='c', required=True, help='Model and transformer configuration')
parser.add_argument('-m', '--core', dest='core', required=True, help='Model parameters!')
parser.add_argument('-d', '--directory', dest='d', default=None, help='Inference data directory.')
parser.add_argument('-l', '--list_file', dest='l', default=None, help='Inference data list file')
parser.add_argument('--labels_file', default=None, help='Labels file')
parser.add_argument('--gpus', type=int, nargs='*', default=None, help='GPU index to be used!')
parser.add_argument('--num_process', type=int, default=1, help='Number of process!')

args = parser.parse_args()


def test_model(samples, thread_id, params):
    # config = json.loads(open(params.config).read())
    config = {'transform': {'input_size': 299, 'normalize_method': '-1+1'},
              'core': {'model_name': 'inception_v3', 'num_classes': 3}}

    # Configuration of transformer.
    transform_config = {'phase': 'valid'}
    if 'transform' in config:
        transform_config.update(config['transform'])
    assert 'input_size' in transform_config, '`input_size` must in `transform`'
    transformer = create_standard_image_transformer(**transform_config)

    # Configuration of core
    model_config = {'pretrained': False}
    if 'core' in config:
        model_config.update(config['core'])
    assert 'model_name' in model_config and 'num_classes' in model_config, '`model_name` and `num_classes` must in ' \
                                                                           '`core`'
    model = create_model(**model_config)
    # Configuration of device
    device_info = 'cpu'
    if params.gpus:
        gpu_idx = params.gpus[thread_id % len(params.gpus)]
        device_info = f"cuda:{gpu_idx}" if torch.cuda.is_available() and gpu_idx else "cpu"
    device = torch.device(device_info)
    model = model.to(device)
    state_dict = torch.load(params.model, map_location=device)['model_state_dict']
    mapped_state_dict = {k.lstrip('module.'): state_dict[k] for k in state_dict}

    model.load_state_dict(mapped_state_dict)
    model.eval()

    # Inference
    with torch.set_grad_enabled(False):
        for sample in samples:
            sample_ = transformer(default_loader(sample))
            sample_ = sample_.to(device)
            # print(sample_.size())
            outputs = model(sample_.view(1, *sample_.size()))
            print(sample, F.softmax(outputs, dim=1))


if __name__ == "__main__":
    if args.d is not None:
        test_samples = [os.path.join(args.d, p) for p in os.listdir(args.d) if p.endswith('.jpg')]
    elif args.l is not None:
        with open(args.l) as f:
            test_samples = [l.strip() for l in f.readlines()]
    else:
        raise ValueError('You must provide a directory or list file for inference.')
    MultiProcess(test_samples, test_model, num_process=args.num_process, params=args).run()
