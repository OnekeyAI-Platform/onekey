# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/22
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.
import argparse
import json
import os
from functools import partial
from typing import Iterable

import numpy as np
import torch

from onekey_algo.datasets.image_loader import default_loader
from onekey_algo.utils.about_log import logger
from onekey_core.core import create_model
from onekey_core.core import create_standard_image_transformer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def extract(samples, model, transformer, device=None, fp=None):
    results = []
    # Inference
    if not isinstance(samples, (list, tuple)):
        samples = [samples]
    with torch.set_grad_enabled(False):
        for sample in samples:
            fp.write(f"{os.path.basename(sample)},")
            sample_ = transformer(default_loader(sample))
            sample_ = sample_.to(device)
            # print(sample_.size())
            outputs = model(sample_.view(1, *sample_.size()))
            results.append(outputs)
    return results


def print_feature_hook(module, inp, outp, fp):
    print(','.join(map(lambda x: f"{x:.6f}", np.reshape(outp.cpu().numpy(), -1))), file=fp)


def reg_hook_on_module(name, model, hook):
    find_ = 0
    for n, m in model.named_modules():
        if name == n:
            m.register_forward_hook(hook)
            find_ += 1
    if find_ == 0:
        logger.warning(f'{name} not found in {model}')
    elif find_ > 1:
        logger.info(f'Found {find_} features named {name} in {model}')
    return find_


def init_from_model(model_name, model_path=None, num_classes=1000, model_state='model_state_dict',
                    img_size=(224, 224), **kwargs):
    # Configuration of core
    kwargs.update({'pretrained': True if model_path is None else False,
                   'model_name': model_name, 'num_classes': num_classes})
    model = create_model(**kwargs).eval()
    # Config device automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)[model_state]
        model.load_state_dict(state_dict)
    if 'inception' in model_name.lower():
        if isinstance(img_size, int):
            if img_size != 299:
                logger.warning(f'{model_name} is inception structure, `img_size` is set to be 299 * 299.')
                img_size = 299
        elif isinstance(img_size, Iterable):
            if 299 not in img_size:
                logger.warning(f'{model_name} is inception structure, `img_size` is set to be 299 * 299.')
                img_size = (299, 299)
    transformer = create_standard_image_transformer(img_size, phase='valid')
    return model, transformer, device


def init_from_onekey(config_path):
    config = json.loads(open(os.path.join(config_path, 'task.json')).read())
    model_path = os.path.join(config_path, 'BEST-training-params.pth')
    assert 'model_name' in config and 'num_classes' in config and 'transform' in config
    # Configuration of transformer.
    transform_config = {'phase': 'valid'}
    transform_config.update(config['transform'])
    assert 'input_size' in transform_config, '`input_size` must in `transform`'
    transformer = create_standard_image_transformer(**transform_config)

    # Configuration of core
    model_config = {'pretrained': False, 'model_name': config['model_name'], 'num_classes': config['num_classes']}
    model = create_model(**model_config)
    # Configuration of device
    device_info = 'cpu'
    device = torch.device(device_info)
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)['model_state_dict']
    for key in list(state_dict.keys()):
        if key.startswith('module.'):
            new_key = key[7:]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
    model.eval()
    return model, transformer, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Inference')

    parser.add_argument('-c', '--config_path', dest='c', default='20211014/resnet18/viz',
                        help='Model and transformer configuration')
    parser.add_argument('-d', '--directory', dest='d',
                        default=r'C:\Users\onekey\Project\data\labelme', help='Inference data directory.')
    parser.add_argument('-l', '--list_file', dest='l', default=None, help='Inference data list file')

    args = parser.parse_args()
    if args.d is not None:
        test_samples = [os.path.join(args.d, p) for p in os.listdir(args.d) if p.endswith('.jpg')]
    elif args.l is not None:
        with open(args.l) as f:
            test_samples = [l.strip() for l in f.readlines()]
    else:
        raise ValueError('You must provide a directory or list file for inference.')
    model_name = 'resnet18'
    model, transformer, device = init_from_model(model_name=model_name)
    # print(model)
    # for n, m in model.named_modules():
    #     print(n, m)
    feature_name = 'avgpool'
    outfile = open('feature.txt', 'w')
    hook = partial(print_feature_hook, fp=outfile)
    find_num = reg_hook_on_module(feature_name, model, hook)
    results_ = extract(test_samples[:5], model, transformer, device, fp=outfile)
    print(json.dumps(results_, ensure_ascii=False, indent=True))
