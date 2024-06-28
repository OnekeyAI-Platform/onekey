# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/22
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import torch.utils as utils
import pandas as pd

from onekey_algo.datasets.ClassificationDataset import ListDataset4Test
from onekey_algo.datasets.image_loader import default_loader
from onekey_algo.utils.about_log import logger
from onekey_core.core import create_model
from onekey_core.core import create_standard_image_transformer


def inference(samples, model, transformer, labels=None, device=None):
    # Inference
    if labels is None:
        labels = {}
    results = []
    if not isinstance(samples, (list, tuple)):
        samples = [samples]
    with torch.set_grad_enabled(False):
        for idx, sample in enumerate(samples):
            if len(samples) > 1e4 and idx % 1000 == 0:
                logger.info(f'正在预测中，已完成：{idx}, 完成率：{idx * 100 / len(samples):.4f}%')
            sample_ = transformer(default_loader(sample))
            sample_ = sample_.to(device)
            # print(sample_.size())
            outputs = model(sample_.view(1, *sample_.size()))
            prob = F.softmax(outputs, dim=1)[0].cpu()
            prediction = torch.argmax(prob)
            results.append((os.path.basename(sample), dict(zip(labels, prob.numpy().tolist())),
                            labels[prediction.item()]))
    return results


def inference_dataloader(samples, model, transformer, labels=None, device=None, batch_size=1, num_workers=1,
                         cached_dir: str = None):
    # Inference
    if labels is None:
        labels = {}
    results = []
    if not isinstance(samples, (list, tuple)):
        samples = [samples]
    dataloader = utils.data.DataLoader(ListDataset4Test(samples, transformer), batch_size=batch_size, drop_last=False,
                                       shuffle=False, num_workers=num_workers)
    calc_num = 0
    with torch.set_grad_enabled(False):
        start_time = time.time()
        for idx, (sample_, fnames) in enumerate(dataloader):
            sample_ = sample_.to(device)
            if len(samples) > 1e4 and idx % 100 == 0 and idx != 0:
                speed = (time.time() - start_time) * 1000 / (idx * batch_size)
                logger.info(
                    f'正在预测中，已完成：{idx * batch_size}, 完成率：{idx * batch_size * 100 / len(samples):.4f}%，'
                    f'移动平均速度是：{speed:.4f} msec/img')
            outputs = model(sample_)
            probs = F.softmax(outputs, dim=1).cpu()
            predictions = torch.argmax(probs, dim=1).cpu()
            # print(probs.shape, predictions.shape)
            for fname, prob, prediction in zip(fnames, probs, predictions):
                results.append((fname, json.dumps(dict(zip(labels, prob.numpy().tolist())), ensure_ascii=False),
                                labels[prediction.item()]))
                calc_num += 1
                if cached_dir is not None and len(samples) > 100 and calc_num % math.ceil(len(samples) / 100) == 0:
                    logger.info(f'Saving cached {calc_num * 100 // len(samples):03d} results...')
                    os.makedirs(cached_dir, exist_ok=True)
                    r = pd.DataFrame(results, columns=['fname', 'prob', 'label'])
                    r.to_csv(os.path.join(cached_dir, f'cached_{calc_num // math.ceil(len(samples) / 100):03d}.csv'),
                             index=False)
                    results = []
        if cached_dir is not None and len(results):
            os.makedirs(cached_dir, exist_ok=True)
            r = pd.DataFrame(results, columns=['fname', 'prob', 'label'])
            r.to_csv(os.path.join(cached_dir, f'cached_100.csv'), index=False)
            results = []
    return results


def init(config_path):
    config = json.loads(open(os.path.join(config_path, 'task.json')).read())
    labels = [l.strip() for l in open(os.path.join(config_path, 'labels.txt'), encoding='utf8').readlines()]
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
    device_info = config.get('device', None) or 'cpu'
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
    return model, transformer, labels, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Inference')

    parser.add_argument('-c', '--config_path', dest='c', default='20211014/resnet18/viz',
                        help='Model and transformer configuration')
    parser.add_argument('-d', '--directory', dest='d',
                        default=r'G:\skin_classification\images', help='Inference data directory.')
    parser.add_argument('-l', '--list_file', dest='l', default=None, help='Inference data list file')

    args = parser.parse_args()

    if args.d is not None:
        test_samples = [os.path.join(args.d, p) for p in os.listdir(args.d) if p.endswith('.jpg')]
    elif args.l is not None:
        with open(args.l) as f:
            test_samples = [l.strip() for l in f.readlines()]
    else:
        raise ValueError('You must provide a directory or list file for inference.')
    model, transformer, labels_, device = init(config_path=args.c)
    results_ = inference(test_samples, model, transformer, labels_, device)
    print(json.dumps(results_, ensure_ascii=False, indent=True))
