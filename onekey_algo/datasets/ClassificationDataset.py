# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2019/7/22
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2019 All Rights Reserved.
import math
import os
import random
from functools import partial
from typing import List

import torch.utils.data as data
import torch
from onekey_algo.datasets.image_loader import default_loader
from onekey_algo.utils.about_log import ColorPrinter
from onekey_algo.utils.about_log import logger
from onekey_algo.utils.common import check_file_exists, common_annotation_parser, \
    check_directory_exists, create_dir_if_not_exists, \
    get_files_in_dir

__all__ = ['create_classification_dataset', 'save_classification_dataset_labels', 'ListDataset', 'FolderDataset']

POSSIBLE_EXTENSION = ['.jpg', '.jpeg', '.png', '.nii', '.nii.gz']


class ListDataset(data.Dataset):
    """
    The training or validation dataset is stored in a or a list of  isolate file.
    One must pass `record_parser` to parse record, _common_record_parser is used by default.
    """

    def __init__(self, records: List[str], record_parser=None, labels_file: str = None, classes: list = None,
                 ori_img_root: str = None, max2use: int = None, batch_balance: bool = False,
                 loader=None, transform=None, target_transform=None, check_sample_exists: bool = True,
                 retry: int = None, **kwargs):
        """
        Initial function for ListDataset. Do the following thing.

            1. Parse record and get samples check file exist if necessary.
            2. Cut number of samples to `max2use` whose samples larger than `max2use`.
            3. Enlarge others' samples to the largest label's number if `batch_balance`.
            4. Match the original and new labels set check or create.
            5. Transform input and target on runtime.

        :param records: A list of records, a single file is sometimes ok.
        :param record_parser: Record parser. Default is utils.common_annotation_parser with tab separator.
        :param labels_file: Map each column to readable label.
            If exists check original labels equal to new otherwise create a new file.
        :param classes: Classes list, if provided will not check labels_file settings which may be used as valid set.
        :param ori_img_root: Original image file root.
            File name in record may be absolute or relative path, either exist is ok.
        :param loader: Method load data from path to PIL.
        :param max2use: Maximum samples to use of each label.
        :param batch_balance: Ensure each batch sample of each label is equal probably appear.
        :param transform: Transformer of input image.
        :param target_transform: Transformer of target.
        :param check_sample_exists: Whether to check image file exists
            default True which may slow down the creation of dataset.
        :param retry: How many time to re get samples if `index` failed. RANDOM index will be used!
        :param kwargs: Unused args.
        """
        self.file_names = []
        self.labels = []
        self.boxes = []
        self.cp = ColorPrinter()
        del kwargs
        if record_parser is None:
            record_parser = partial(common_annotation_parser, sep='\t')
        if loader is None:
            loader = default_loader
        if isinstance(records, str):
            records = [records]
        assert all(r and os.path.exists(r) and os.path.isfile(r) for r in records), f"Not all records file exist! " \
                                                                                    f"Check {records}."

        # Parse records file.
        for record in records:
            logger.info(self.cp.color_text(f'Parsing record file {record}'))
            file_name_, label_, bbox_ = record_parser(record)
            if len(file_name_) >= 1e5:
                logger.info(f"\tSkip checking file exists in {record}")
            for file_name, label, bbox in zip(file_name_, label_, bbox_):
                if check_sample_exists and len(file_name_) < 1e5:
                    abs_file = check_file_exists(file_name, ori_img_root)
                    if abs_file:
                        self.file_names.append(abs_file)
                    else:
                        continue
                else:
                    if ori_img_root is not None:
                        self.file_names.append(os.path.join(ori_img_root, file_name))
                    else:
                        self.file_names.append(file_name)
                self.labels.append(label)
                self.boxes.append(bbox)

        # Cut samples and batch balance it if necessary.
        if max2use is not None or batch_balance:
            _labels_samples = {}
            for file_name, label, bbox in zip(self.file_names, self.labels, self.boxes):
                if label not in _labels_samples:
                    _labels_samples[label] = []
                _labels_samples[label].append((file_name, bbox))

            # Max to use firstly.
            _max_samples = 0
            for l in _labels_samples:
                random.shuffle(_labels_samples[l])
                _labels_samples[l] = _labels_samples[l][:max2use]
                if len(_labels_samples[l]) > _max_samples:
                    _max_samples = len(_labels_samples[l])
            # Batch balance secondly.
            if batch_balance:
                logger.info(f'正在使用Batch Balance参数进行训练！')
                for l in _labels_samples:
                    _enlarge_ratio = math.ceil(_max_samples / len(_labels_samples[l]))
                    _labels_samples[l] = _labels_samples[l] * _enlarge_ratio
                    _labels_samples[l] = _labels_samples[l][:_max_samples]
                    random.shuffle(_labels_samples[l])
            # unzip data and reinitialise
            self.file_names = []
            self.labels = []
            self.boxes = []
            for l in _labels_samples:
                for file_name, bbox in _labels_samples[l]:
                    self.file_names.append(file_name)
                    self.labels.append(l)
                    self.boxes.append(bbox)

        if classes and isinstance(classes, list):
            self.classes = classes
        else:
            if labels_file and os.path.exists(labels_file) and os.path.isfile(labels_file):
                with open(labels_file, encoding='utf8') as f:
                    self.classes = [l.strip() for l in f.readlines()]
                    if len(self.labels) == 0:
                        raise ValueError('没有找到任何匹配的数据，请检查数据配置！')
                    if not sorted(self.classes) == sorted(list(set(self.labels))):
                        if set(self.classes) - set(self.labels):
                            print(f"标签和预设存在差异：{set(self.classes) - set(self.labels)} "
                                  f"或者 {set(self.labels) - set(self.classes)}")
                        raise ValueError(f"Labels's set must equal to {labels_file}")
            else:
                self.classes = sorted(list(set(self.labels)))
                if labels_file:
                    create_dir_if_not_exists(os.path.dirname(labels_file))
                    with open(labels_file, 'w', encoding='utf8') as f:
                        f.write('\n'.join(self.classes))

        self.classes_to_idx = dict([(c, i) for i, c in enumerate(self.classes)])
        self.labels = [self.classes_to_idx[l] for l in self.labels]

        # Initial other hyper parameters.
        self.labels_file = labels_file
        self.root = ori_img_root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.retry = retry
        if not len(self.labels):
            logger.warning(self.cp.color_text('0 sample in this dataset!', 'yellow'))

    def __get_item(self, index):
        path_ = self.file_names[index]
        label_ = self.labels[index]

        sample_ = self.loader(path_)
        if self.transform is not None:
            sample_ = self.transform(sample_)
            if path_.endswith('.nii.gz') or path_.endswith('.nii'):
                sample_ = torch.squeeze(sample_)
        if self.target_transform is not None:
            label_ = self.target_transform(label_)
        return sample_, label_, path_

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Get an example.

        :param index: Index
        :return: (sample, bbox, label) where label is class_index of the target class.
        """
        try:
            return self.__get_item(index)
        except Exception as e:
            attempt = 0
            times = 'infinite' if self.retry is None else self.retry
            logger.warning(self.cp.color_text(f'{self.file_names[index]} is dropped because of {e}', 'yellow'))
            logger.info(self.cp.color_text(f'We now attempt {times} times to get datasets sample randomly!'))
            while self.retry is None or attempt < self.retry:
                _rand_idx = random.randint(0, len(self.labels) - 1)
                logger.info(self.cp.color_text(f'Attempting at {attempt + 1} using index {_rand_idx}!', 'cyan'))
                try:
                    return self.__get_item(_rand_idx)
                except:
                    pass
                attempt += 1
            raise e

    def __repr__(self):
        head = "Dataset is " + self.__class__.__name__
        body = ["Number of samples: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        if self.labels_file is not None:
            body.append(f'Labels file is stored in {os.path.abspath(self.labels_file)}')
        lines = [head] + [f"\t{line}" for line in body]
        return '\n'.join(lines)


class FolderDataset(data.Dataset):
    """
    The training or validation dataset is stored in an isolate directory. Each sample must in jpeg format.
    """

    def __init__(self, ori_img_root: List[str], recursive: bool = False, labels_file: str = None, classes: list = None,
                 max2use: int = None, batch_balance: bool = False,
                 loader=None, transform=None, target_transform=None, retry: int = None, **kwargs):
        """
        Initial function for FolderDataset. Do the following thing.

            1. Get samples in each sub directory `ori_img_root`, recursively if necessary.
            2. Match the original and new labels set check or create.
            3. Cut number of samples to `max2use` whose samples larger than `max2use`.
            4. Enlarge others' samples to the largest label's number if `batch_balance`.
            5. Transform input and target on runtime.

        :param ori_img_root: Original image file root.
            Each folder is a class, All folder's classes must be equal to each other if created.
        :param recursive: Get samples in sub directory recursively, default False.
        :param labels_file: Map each column to readable label.
            If exists check original labels equal to new otherwise create a new file.
        :param classes: Classes list, if provided will not check labels_file settings which may be used as valid set.
        :param max2use: Maximum samples to use of each label.
        :param batch_balance: Ensure each batch sample of each label is equal probably appear.
        :param loader: Method load data from path to PIL.
        :param transform: Transformer of input image.
        :param target_transform: Transformer of target.
        :param retry: How many time to re get samples if `index` failed. RANDOM index will be used!
        :param kwargs: Unused args.
        """
        self.bbox_mapping_ = {}
        self.file_names = []
        self.labels = []
        self.boxes = []
        self.cp = ColorPrinter()
        del kwargs
        if loader is None:
            loader = default_loader

        if not isinstance(ori_img_root, (list, tuple)):
            ori_img_root = [ori_img_root]
        check_directory_exists(*ori_img_root, prefixes='ori_img_root')

        if classes and isinstance(classes, list):
            self.classes = classes
        else:
            if labels_file and os.path.exists(labels_file) and os.path.isfile(labels_file):
                with open(labels_file, encoding='utf8') as f:
                    self.classes = [l.strip() for l in f.readlines()]
            else:
                self.classes = sorted(l for l in os.listdir(ori_img_root[0])
                                      if os.path.isdir(os.path.join(ori_img_root[0], l)))
                for directory_ in ori_img_root[1:]:
                    assert sorted(l for l in os.listdir(directory_)
                                  if os.path.isdir(os.path.join(directory_, l))) == self.classes

                # Create labels file if `labels_file` is not None.
                if labels_file:
                    create_dir_if_not_exists(os.path.dirname(labels_file))
                    with open(labels_file, 'w', encoding='utf8') as f:
                        f.write('\n'.join(self.classes))

        # Map class to  index.
        self.classes_to_idx = dict([(c, i) for i, c in enumerate(self.classes)])
        # Get samples in each directory.
        _labels_samples = {}
        for directory_ in ori_img_root:
            logger.info(self.cp.color_text(f'Getting samples in {directory_}'))
            for label_ in self.classes:
                sub_directory = os.path.join(directory_, label_)
                samples = get_files_in_dir(sub_directory, recursive=recursive, extension=POSSIBLE_EXTENSION)
                # r = '-L' if recursive else ""
                # samples = [bytes.decode(i.strip())
                #            for i in subprocess.Popen("find %s %s -name '*.jpg'" % (r, sub_directory),
                #                                      shell=True, stdout=subprocess.PIPE,
                #                                      stderr=subprocess.PIPE).stdout.readlines()]
                if label_ not in _labels_samples:
                    _labels_samples[label_] = []
                _labels_samples[label_].extend([(s, self.bbox_mapping(s)) for s in samples])
                logger.info(self.cp.color_text(f'\tTotal got {len(samples)} samples in {sub_directory}'))

        # Cut samples and batch balance it if necessary.
        if max2use is not None or batch_balance:
            # Max to use firstly.
            _max_samples = 0
            for l in _labels_samples:
                random.shuffle(_labels_samples[l])
                _labels_samples[l] = _labels_samples[l][:max2use]
                if len(_labels_samples[l]) > _max_samples:
                    _max_samples = len(_labels_samples[l])
            # Batch balance secondly.
            for l in _labels_samples:
                logger.info(f'正在使用Batch Balance参数进行训练！')
                _enlarge_ratio = math.ceil(_max_samples / len(_labels_samples[l]))
                _labels_samples[l] = _labels_samples[l] * _enlarge_ratio
                _labels_samples[l] = _labels_samples[l][:_max_samples]
                random.shuffle(_labels_samples[l])
        # unzip data and reinitialise
        self.file_names = []
        self.labels = []
        self.boxes = []
        for l in self.classes:
            if classes is None and not (l in _labels_samples and len(_labels_samples[l])):
                raise RuntimeError(f'All labels must has samples, but {l}.')
            for file_name, bbox in _labels_samples[l]:
                self.file_names.append(file_name)
                self.labels.append(l)
                self.boxes.append(bbox)

        self.labels = [self.classes_to_idx[l] for l in self.labels]

        # Initial other hyper parameters.
        self.labels_file = labels_file
        self.root = ori_img_root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.retry = retry
        if not len(self.labels):
            logger.warning(self.cp.color_text('0 sample in this dataset!', 'yellow'))

    def bbox_mapping(self, name):
        if name in self.bbox_mapping_:
            return self.bbox_mapping_[name]
        if os.path.basename(name) in self.bbox_mapping_:
            return self.bbox_mapping_[os.path.basename(name)]
        return None

    def __get_item(self, index):
        path_ = self.file_names[index]
        label_ = self.labels[index]

        sample_ = self.loader(path_)
        if self.transform is not None:
            sample_ = self.transform(sample_)
        if self.target_transform is not None:
            label_ = self.target_transform(label_)
        return sample_, label_, os.path.basename(path_)

    @property
    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        """
        Get an example.

        :param index: Index
        :return: (sample, bbox, label) where label is class_index of the target class.
        """
        try:
            return self.__get_item(index)
        except Exception as e:
            attempt = 0
            times = 'infinite' if self.retry is None else self.retry
            logger.warning(self.cp.color_text(f'{self.file_names[index]} is dropped because of {e}', 'yellow'))
            logger.info(self.cp.color_text(f'We now attempt {times} times to get datasets sample randomly!'))
            while self.retry is None or attempt < self.retry:
                _rand_idx = random.randint(0, len(self.labels) - 1)
                logger.info(self.cp.color_text(f'Attempting at {attempt + 1} using index {_rand_idx}!', 'cyan'))
                try:
                    return self.__get_item(_rand_idx)
                except:
                    pass
                attempt += 1
            raise e

    def __repr__(self):
        head = "Dataset is " + self.__class__.__name__
        body = ["Samples: {},Classes: {}".format(self.__len__(), self.num_classes)]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        if self.labels_file is not None:
            body.append(f'Labels file is stored in {os.path.abspath(self.labels_file)}')
        lines = [head] + [f"\t{line}" for line in body]
        return '\n'.join(lines)


def create_classification_dataset(dataset_name=None, **kwargs):
    """
    Create an specified dataset with given args.
    If `records` params in `kwargs` and is NOT NONE we treat this as ListDataset else treated as FolderDataset.

    WE RECOMMEND YOU USE SPECIFY DATASET LIKE `ListDataset` or `FolderDataset`.

    :param dataset_name: Dataset name, list or folder.
    :param kwargs: Key word to create dataset.
    :return: a dataset.
    """
    cp = ColorPrinter()
    assert dataset_name is None or dataset_name in ['list', 'folder'], f'Dataset {dataset_name} not supported!'
    if dataset_name is None:
        logger.info(cp.color_text('WE RECOMMEND YOU USE SPECIFY dataset_name LIKE list for ListDataset OR '
                                  'folder for FolderDataset.',
                                  'yellow', attrs='blink'))
        if 'records' in kwargs and kwargs['records'] is not None:
            dataset = ListDataset(**kwargs)
        else:
            dataset = FolderDataset(**kwargs)
        logger.info(cp.color_text(f'We infer your kwargs to be {type(dataset)}.', 'cyan'))
    elif dataset_name == 'list':
        return ListDataset(**kwargs)
    else:
        return FolderDataset(**kwargs)
    return dataset


def save_classification_dataset_labels(dataset, path):
    assert isinstance(dataset, ListDataset) or isinstance(dataset, FolderDataset), f'{type(dataset)} not supported!'
    create_dir_if_not_exists(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write('\n'.join(dataset.classes))


class ListDataset4Test(data.Dataset):
    def __init__(self, data_list, transform):
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, index):
        path_ = self.data_list[index]

        sample_ = default_loader(path_)
        sample_ = self.transform(sample_)
        return sample_, os.path.basename(path_)

    def __len__(self):
        return len(self.data_list)
