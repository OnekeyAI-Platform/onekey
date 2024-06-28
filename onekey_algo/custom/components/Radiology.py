# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/12/11
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.
import concurrent.futures
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import radiomics
import yaml
from radiomics import featureextractor

from onekey_algo.utils.about_log import logger
from onekey_core.core.image_loader import image_loader_3d

radiomics.logger.setLevel(logging.ERROR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_image_mask_from_dir(path, images=None, masks=None):
    items = os.listdir(path)
    images = images or 'images'
    masks = masks or 'masks'
    assert images in items and masks in items
    images_path = Path(os.path.join(path, images))
    masks_path = Path(os.path.join(path, masks))
    images = []
    masks = []
    for l_ in os.listdir(images_path):
        if not l_.startswith('.'):
            f_name, _ = os.path.splitext(l_)
            mask_file = list(masks_path.glob(f_name + '*'))
            if len(mask_file) == 1:
                images.append(os.path.abspath(os.path.join(images_path, l_)))
                masks.append(os.path.abspath(mask_file[0]))
            elif len(mask_file) > 1:
                logger.warning(f"我们找到多个与{l_}有共同pattern的文件, 他们是{[os.path.basename(l) for l in mask_file]}， "
                               f"我们将使用完全匹配获取数据。")
                if os.path.exists(os.path.join(masks_path, l_)):
                    images.append(os.path.abspath(os.path.join(images_path, l_)))
                    masks.append(os.path.abspath(os.path.join(masks_path, l_)))
            else:
                logger.warning(f"我们没有找到与{l_}有共同pattern的文件，Onekey将放弃这个数据数据。")

    return images, masks


def get_pair_from_2dir(xpath, ypath, strict: bool = True):
    assert os.path.isdir(xpath) and os.path.isdir(ypath)
    images = []
    masks = []
    xpath = Path(xpath)
    ypath = Path(ypath)
    if strict:
        for l_ in os.listdir(xpath):
            if not l_.startswith('.'):
                f_name, _ = os.path.splitext(l_)
                mask_file = [str(p) for p in ypath.glob(f_name + '*')]
                if len(mask_file) == 1:
                    images.append(os.path.abspath(os.path.join(xpath, l_)))
                    masks.append(os.path.abspath(mask_file[0]))
                else:
                    if os.path.join(ypath, l_) in mask_file:
                        images.append(os.path.abspath(os.path.join(xpath, l_)))
                        masks.append(os.path.abspath(os.path.join(ypath, l_)))
    else:
        images = sorted([os.path.join(xpath, i) for i in os.listdir(xpath) if not i.startswith('.')])
        masks = sorted([os.path.join(ypath, i) for i in os.listdir(ypath) if not i.startswith('.')])
    assert len(images) == len(masks), "获取的图像和mask数量不匹配"
    return images, masks


def diagnose_3d_image_mask_settings(ipath, mpath, assume_masks: List[int] = None, verbose: bool = False):
    """
    检查 Pyradiomics 特征提取的数据是否符合要求。
    Args:
        ipath: images的集合，list
        mpath: masks的集合，list
        assume_masks: 预定mask包括的label集合。
        verbose: 是否打印中间结果日志。

    Returns: 没有错误的images，masks

    """
    diagnose = []
    label_set = set()
    join_label_set = None
    correct_images = []
    correct_masks = []
    if len(ipath) != len(mpath):
        diagnose.append(f"图像和Mask的数量不相等，检查图像数据量和Mask数据量。")
    for i, m in zip(ipath, mpath):
        if not (os.path.exists(i) and os.path.isfile(i)):
            diagnose.append(f"图像文件：{i}不存在！")
        if not (os.path.exists(m) and os.path.isfile(m)):
            diagnose.append(f"Mask文件：{m}不存在！")
        bi = os.path.basename(i)
        bm = os.path.basename(m)
        try:
            image = image_loader_3d(i)
            mask = image_loader_3d(m)
            mask_labels = np.unique(mask)
            if verbose:
                label_set |= set(mask_labels)
                if join_label_set is None:
                    join_label_set = set(mask_labels)
                join_label_set &= set(mask_labels)
                logger.info(f'正在检查：{bi}{image.shape}和{bm}{mask.shape}，标签集合：{mask_labels}')

            # import numpy as np
            # print(np.unique(mask.get_data()))
            test_pass = True
            if not image.shape == mask.shape:
                test_pass = False
                diagnose.append(f"图像 {bi}({image.shape}) 和Mask {bm}({mask.shape})的尺寸不匹配")
            if assume_masks and sorted(mask_labels) != sorted(assume_masks):
                test_pass = False
                diagnose.append(f"Mask: {bm}的labels（{mask_labels[:3]}...）与预期（{assume_masks}）不同")
            if len(image.shape) not in (2, 3):
                test_pass = False
                diagnose.append(f"图像 {bi} 和Mask {bm}不是2D或者3D数据")
            if test_pass:
                correct_images.append(i)
                correct_masks.append(m)
        except Exception as e:
            traceback.print_exc()
            diagnose.append(f"图像 {bi} 和Mask {bm} 存在{e}")
    if not diagnose:
        print('检查通过！')
    else:
        print('请检查如下设置：')
        for idx, d in enumerate(diagnose):
            print(f"问题{idx + 1}： {d}")
    if verbose:
        print(f'标签集合为：{label_set}, 共有标签为：{join_label_set}')
    return correct_images, correct_masks


class ConventionalRadiomics(object):
    def __init__(self, params_file: str = None, **params):
        settings = {}
        if params_file is not None:
            print(f"Onekey Lite不支持参数文件指定，如需自定义配置文件，请升级Onekey Professional。")
            time.sleep(3)

        self.params_file = None
        self.params = params
        self.settings = settings
        self._features = {}
        self.feature_names = set()
        self.statics_names = set()
        self.extractor = None
        self.df = None
        self.errors = []

        # Initialize feature extractor
        self.extractor = self.init_extractor(self.settings)

    def init_extractor(self, settings=None):
        settings = settings or self.settings
        return featureextractor.RadiomicsFeatureExtractor(settings, **self.params)

    def extract(self, images: Union[str, List[str]], masks: Union[str, List[str]],
                labels: Union[int, List[int]] = 1, settings=None, workers: int = 1):
        """

        Args:
            images:
            masks:
            labels:
            settings:
            workers:

        Returns:

        """
        logger.info('Extracting features...')
        if settings is not None:
            extractor = self.init_extractor(settings)
        else:
            extractor = self.extractor
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(masks, (list, tuple)):
            masks = [masks]
        assert len(images) == len(masks), '图像和标注数据必须一一对应。'
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        if workers == 1:
            for image, mask in zip(images, masks):
                image_name = os.path.basename(image)
                self._features[image_name] = {}
                for label in labels:
                    try:
                        statics = {}
                        features = {}
                        logger.info(f'\tExtracting feature from {image} using label {label}')
                        featureVector = extractor.execute(image, mask, label=label)
                        for featureName in featureVector.keys():
                            f_type, c_name, f_name = featureName.split('_')
                            if f_type == 'diagnostics':
                                self.statics_names.add(f"{f_type}_{c_name}_{f_name}")
                                if f"{f_type}_{c_name}" not in statics:
                                    statics[f"{f_type}_{c_name}"] = {}
                                statics[f"{f_type}_{c_name}"].update({f_name: featureVector[featureName]})
                            else:
                                self.feature_names.add(f"{f_type}_{c_name}_{f_name}")
                                if f"{f_type}_{c_name}" not in features:
                                    features[f"{f_type}_{c_name}"] = {}
                                features[f"{f_type}_{c_name}"].update({f_name: float(featureVector[featureName])})
                        self._features[image_name][label] = {"statics": statics, 'features': features}
                    except Exception as e:
                        logger.error(f"{image_name} extract {label} error, {e}")
                        self.errors.append((image_name, label, e))
            # print(json.dumps(self._features, indent=True))
        elif workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                parallel_labels = [labels] * len(images)
                parallel_settings = [settings] * len(images)
                results = executor.map(self.extract_unit, images, masks, parallel_labels, parallel_settings)
                for _f, fn, sn in results:
                    self.statics_names = sn
                    self.feature_names = fn
                    self._features.update(_f)
        logger.info(f'特征提取完成！')
        return self._features

    def extract_unit(self, images: Union[str, List[str]], masks: Union[str, List[str]],
                     labels: Union[int, List[int]] = 1, settings=None):
        _features = {}
        feature_names = set()
        statics_names = set()
        if settings is not None:
            extractor = self.init_extractor(settings)
        else:
            extractor = self.extractor
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(masks, (list, tuple)):
            masks = [masks]
        assert len(images) == len(masks), '图像和标注数据必须一一对应。'
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        for image, mask in zip(images, masks):
            image_name = os.path.basename(image)
            _features[image_name] = {}
            for label in labels:
                try:
                    statics = {}
                    features = {}
                    logger.info(f'\tExtracting feature from {image} using label {label}')
                    featureVector = extractor.execute(image, mask, label=label)
                    for featureName in featureVector.keys():
                        f_type, c_name, f_name = featureName.split('_')
                        if f_type == 'diagnostics':
                            statics_names.add(f"{f_type}_{c_name}_{f_name}")
                            if f"{f_type}_{c_name}" not in statics:
                                statics[f"{f_type}_{c_name}"] = {}
                            statics[f"{f_type}_{c_name}"].update({f_name: featureVector[featureName]})
                        else:
                            feature_names.add(f"{f_type}_{c_name}_{f_name}")
                            if f"{f_type}_{c_name}" not in features:
                                features[f"{f_type}_{c_name}"] = {}
                            features[f"{f_type}_{c_name}"].update({f_name: float(featureVector[featureName])})
                    _features[image_name][label] = {"statics": statics, 'features': features}
                except Exception as e:
                    logger.error(f"{image_name} extract {label} error, {e}")
        # print(json.dumps(self._features, indent=True))
        return _features, feature_names, statics_names

    @property
    def features(self, labels: Union[list, tuple, set] = None):
        if self._features:
            feature = {}
            for k_, v_ in self._features.items():
                feature[k_] = {l_: f_['features'] for l_, f_ in v_.items() if labels is None or l_ in labels}
            return feature
        else:
            logger.warning(f'No features found! Perhaps you should input images and masks!')

    @property
    def statics(self, labels: Union[list, tuple, set] = None):
        if self._features:
            statics = {}
            for k_, v_ in self._features.items():
                statics[k_] = {l_: f_['statics'] for l_, f_ in v_.items() if labels is None or l_ in labels}
            return statics
        else:
            logger.warning(f'No features found! Perhaps you should input images and masks!')

    def get_label_data_frame(self, label: int = 1, column_names=None, ftype='features'):
        if ftype == 'features':
            column_names = column_names or sorted(list(self.feature_names))
            features_dict = self.features.items()
        else:
            column_names = column_names or sorted(list(self.statics_names))
            features_dict = self.statics.items()
        not_has = set()
        for k_, v_ in features_dict:
            if v_ and label in v_:
                for name in column_names:
                    f_type, c_name, f_name = name.split('_')
                    if f"{f_type}_{c_name}" not in v_[label]:
                        not_has.add(name)
        column_names = sorted(list(set(column_names) - not_has))
        if not_has:
            logger.warning(f"存在某些特征{not_has}在提取的时候并不是出现在所有样本中，一般可以忽略这个问题。")
        indexes = []
        df = []
        for k_, v_ in features_dict:
            if v_:
                data = [k_]
                if label in v_:
                    indexes.append(k_)
                    for name in column_names:
                        f_type, c_name, f_name = name.split('_')
                        data.append(v_[label][f"{f_type}_{c_name}"][f_name])
                    df.append(data)
                else:
                    logger.warning(f"{k_}的label={label}没有计算出任何特征！"
                                   f"你可能需要修改：radiomics.extract(images, masks, labels=[{label}])")
        self.df = pd.DataFrame(df, columns=['ID'] + column_names, index=indexes)
        return self.df
