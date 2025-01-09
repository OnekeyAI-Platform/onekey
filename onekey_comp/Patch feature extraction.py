#!/usr/bin/env python
# coding: utf-8

# Patch feature extraction
# You need to enter the training files corresponding to the two test sets of 'train' and 'valid', with one sample per row. 'labels.txt' is an optional argument with one row for each category. 'data_pattern' is a generic directory that is spliced with the first column in train and val.


# train.txt: a list of training data
# val.txt: a list of validation data
# labels.txt: a collection of labels that indicates how many labels are used in the training data
# data_pattern: the common prefix of the directory where all data exists, if it is train.txt, the val.txt file is stored in the absolute path, data_pattern set to None

import os
from onekey_algo.classification.run_classification import main as clf_main
from collections import namedtuple

# parameters setting
# save_dir = r'your feature file path'
val_f = os.path.join(save_dir, 'val.txt')
labels_f = os.path.join(save_dir, 'labels.txt')
data_pattern = os.path.join(save_dir, 'images')

params = dict(train=train_f,
              valid=val_f,
              labels_file=labels_f,
              data_pattern=data_pattern,
              j=0,
              max2use=None,
              val_max2use=None,
              batch_balance=False,
              normalize_method='imagenet',
              model_name='resnet50',
              vit_settings = {'patch_size': 64, 'dim': 1024, 'depth': 6, 'heads': 16, 'mlp_dim': 2048},
              gpus=[0],
              batch_size=64,
              epochs=40,
              init_lr=0.01,
              optimizer='sgd',
              retrain=None,
              model_root='.',
              add_date=False,
              iters_start=0,
              iters_verbose=1,
              save_per_epoch=False,
              pretrained=True)
# model training
Args = namedtuple("Args", params)
clf_main(Args(**params))

