# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/24
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.
import torch.optim as optim

__all__ = ['create_lr_scheduler', 'create_optimizer']


def create_optimizer(opt_name, parameters, **kwargs):
    """
    Create optimizer with specific optimizer name. Supported optimizers are as followings.
        'ASGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'LBFGS', 'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam'

    :param opt_name: optimizer name.
    :param parameters: Parameters to be optimized.
    :param kwargs: other optimizer settings.
    :return: optimizer
    :raises:
        ValueError, Optimizer not found.
        AssertError, `params` not found in pre-parameter settings.
    """
    supported_optimizer = {name.lower(): name for name in optim.__dict__
                           if not name.startswith("__")
                           and callable(optim.__dict__[name])}
    if opt_name.lower() not in supported_optimizer:
        raise ValueError(f'Optimizer name {opt_name} not supported!')

    # If Pre-parameter settings.
    if isinstance(parameters, list):
        for param_ in parameters:
            if isinstance(param_, dict):
                assert 'params' in param_, '`params` must contains in pre-parameter settings.'

    return optim.__dict__[supported_optimizer[opt_name.lower()]](params=parameters, **kwargs)


def create_lr_scheduler(scheduler_name: str, optimizer, **kwargs):
    """Learning rate scheduler to change lr dynamically.

    :param scheduler_name: learning rate scheduler name
    :param optimizer: A instance of optimizer.
    :param kwargs: other key args for lr scheduler.
    :return: learning rate scheduler.
    :raise: ValueError, learning rate scheduler not found.
    """
    supported_optimizer = {'lambda': optim.lr_scheduler.LambdaLR,
                           'step': optim.lr_scheduler.StepLR,
                           'mstep': optim.lr_scheduler.MultiStepLR,
                           'exponential': optim.lr_scheduler.ExponentialLR,
                           'cosine': optim.lr_scheduler.CosineAnnealingLR,
                           'reduce': optim.lr_scheduler.ReduceLROnPlateau,
                           'circle': optim.lr_scheduler.CyclicLR}

    if scheduler_name.lower() not in supported_optimizer:
        raise ValueError(f'Scheduler name {scheduler_name} not supported!')
    return supported_optimizer[scheduler_name](optimizer, **kwargs)
