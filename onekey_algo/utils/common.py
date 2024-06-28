# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2021/3/21
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2021 All Rights Reserved.

import os
import random
import re
import shutil
import sys
import time
from datetime import datetime
from functools import partial
from typing import List

import numpy as np
import pandas

from onekey_algo.utils.about_log import ColorPrinter
from onekey_algo.utils.about_log import logger

__all__ = ['del_k_in_dict_if_exist', 'delete_dir_if_exists',
           'create_directories_if_not_exists', 'create_dir_if_not_exists',
           'truncate_dir', 'check_file_exists', 'check_directory_exists', 'common_annotation_parser',
           'calc_md5', 'parse_txt', 'parse_records',
           'get_abs_save_to', 'get_function_name', 'get_files_in_dir', 'get_value_in_dict', 'cmp_ab_with_color',
           'split_parameter', 'load_category', 'split_dataset']


def delete_dir_if_exists(directory, directly=False, warning=False):
    """
    Delete directory if exists, If `directly`, delete without message prompt otherwise user should confirm.

    :param directory: Where to delete recursively.
    :param directly: Whether to delete directly.
    :param warning: WARNING INFO if ture
    :return:
    """
    if os.path.exists(directory):
        if not directly:
            logger.warning(f"{directory} already exists! Delete it? yes[y]/No[n]")
            i = input()
            if i.lower() == 'y' or i.lower() == 'yes':
                directly = True
        if directly:
            shutil.rmtree(directory, ignore_errors=True)
            logger.info(f"Successfully delete directory {directory}")
    elif warning:
        logger.warning(f"{directory} not exists!")


def create_directories_if_not_exists(*directories, truncate=False):
    """Create directories if not exists.

    :param directories: Directory to create.
    :param truncate: Truncate directory or not.
    """
    for directory in directories:
        if truncate:
            shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)


def create_dir_if_not_exists(directory, add_date=False, add_time=False) -> str:
    """Create directory if not exists.

    :param directory: Directory to create.
    :param add_date: Add date directory. If True, `directory/DATE` will be created.
    :param add_time: Add datetime directory. If True, `directory/DATETIME` will be created.
    :return path: The created path.
    """
    path = None
    if directory:
        path = directory
        if add_date:
            path = os.path.join(path, datetime.now().strftime("%Y%m%d"))
        if add_time:
            path = os.path.join(path, datetime.now().strftime("%H%M%S"))
        os.makedirs(path, exist_ok=True)
    return path


def truncate_dir(directory, del_directly=False, **kwargs):
    """
    Truncate directory.

    :param directory: Which directory to be truncated!
    :param del_directly:  Delete directory directly or not.
    :return:
    """
    delete_dir_if_exists(directory, del_directly)
    return create_dir_if_not_exists(directory, **kwargs)


def check_directory_exists(*directories, prefixes=''):
    """Check directory if not exists raise IOError."""
    # Check prefix's length and type.
    if type(prefixes) == list or type(prefixes) == tuple:
        assert len(prefixes) == len(directories), 'length of `prefix` and `directories` must be equal!'
    elif type(prefixes) == str:
        prefixes = [prefixes] * len(directories)
    for directory, prefix in zip(directories, prefixes):
        if directory and not os.path.exists(directory):
            raise IOError('%s: %s not exists!' % (prefix, directory))


def check_file_exists(file_name, ori_img_root=None, force_file: bool = True):
    """
    Check whether file exists! We will use the first found file in ori_img_root param.

    :param file_name: File name, absolute file path or relative file path.
    :param ori_img_root: Origin image root for `file_name` is relative file path.
    :param force_file: Force file to be file not a directory. Default True.
    :return:
    """
    abs_file = None
    if not isinstance(ori_img_root, (list, tuple)):
        ori_img_root = [ori_img_root]
    if file_name and os.path.exists(file_name):
        abs_file = file_name
    else:
        for ori_img_root_ in ori_img_root:
            if ori_img_root_ and file_name and os.path.exists(os.path.join(ori_img_root_, file_name)):
                abs_file = os.path.join(ori_img_root_, file_name)
                break
    if abs_file and (not force_file or os.path.isfile(abs_file)):
        return abs_file
    else:
        return None


def parse_txt(record_file, sep: str = '\t| ', is_record_valid_func=None, force_valid: bool = False,
              convert2float_if_necessary=None) -> list:
    """
    Parse data from text file.

    :param record_file: Record file, If None use stdin as input file.
    :param sep: Separation for each column of each record.
    :param is_record_valid_func: Judge whether to drop the record.
    :param force_valid: Force record to be valid otherwise raise ValueError.
    :param convert2float_if_necessary: convert to float if necessary, Default False.
    :return: The parsed data in format list.
    :raise ValueError, if is not valid record.
    """
    results = []
    if record_file is None:
        f = sys.stdin
    else:
        check_directory_exists(record_file)
        f = open(record_file)
    for l in f.readlines():
        items = re.split(sep, l.strip())
        if convert2float_if_necessary is not None:
            for idx_, i_ in enumerate(items):
                try:
                    if isinstance(convert2float_if_necessary, bool) and convert2float_if_necessary:
                        items[idx_] = float(i_)
                    else:
                        if not isinstance(convert2float_if_necessary, (list, tuple)):
                            convert2float_if_necessary = [convert2float_if_necessary]
                        if idx_ in convert2float_if_necessary:
                            items[idx_] = float(i_)
                except:
                    pass
        if is_record_valid_func is None or (is_record_valid_func is not None and is_record_valid_func(items)):
            results.append(items)
        elif is_record_valid_func is not None and force_valid:
            raise ValueError(f'{l} is not valid record!')
    f.close()
    return results


def parse_records(record_file, is_csv=None, header: int = None, sep: str = '\t') -> pandas.DataFrame:
    """Return the record in pandas' DataFrame format.

    :param record_file: Record file to be parsed.
    :param is_csv: Whether the record file is in csv format. True or `record_file` ends with .csv.
    :param header: Which line to be used as name.
    :param sep: Separator for each column.
    :return: Parsed record in pandas data frame format.
    """
    # If the original data is type of DataFrame then return.
    if isinstance(record_file, pandas.DataFrame):
        return record_file
    if is_csv is None and record_file is not None:
        is_csv = record_file.endswith('.csv')
    if is_csv:
        df = pandas.read_csv(record_file, header=header)
    else:
        results = parse_txt(record_file, sep=sep)
        if header is not None:
            names = results.pop(header)
            df = pandas.DataFrame(results)
            df.columns = names
        else:
            df = pandas.DataFrame(results)
    # df.columns = [str(i) for i in range(len(df.columns))]
    return df


def common_annotation_parser(record, sep='\t'):
    """
    Common record parser for data samples.

    :param record: Record file has 2 or 6 columns.
    :param sep: Separation of each column.
    :return: tuple of file_name, label, bbox list.
    """
    file_name_ = []
    label_ = []
    bbox_ = []
    with open(record, encoding='utf8') as f:
        for l in f.readlines():
            items = l.strip().split(sep)
            if not (len(items) == 2 or len(items) == 6):
                logger.error(f"文件：{record}，此行存在问题：{l}")
                raise ValueError("Annotation must be length of 2 for `file_name` and `label` or "
                                 "length of 6 for `file_name`, `label`, `top`, `left`, `height`, `width`.")
            file_name_.append(items[0])
            label_.append(items[1])
            bbox_.append([float(b) for b in items[2:]])
    return file_name_, label_, bbox_


def load_category(category_path: str, start_from: int = 0):
    """ Parse category.
    :param category_path: Category path, each line is a specific label whose numeric label is it's index.
    :param start_from: Use start_from to be used as start number. Default 0.
    :return category: from index to label
    :return rev_category: from label to index
    """
    category = {}
    rev_category = {}
    with open(category_path) as f:
        for idx, label in enumerate(f.readlines()):
            category[str(idx + start_from)] = label.strip()
            rev_category[label.strip()] = str(idx + start_from)
    return category, rev_category


def get_abs_save_to(save_to: str, prefix='', create_dir: bool = True):
    """Get absolute file path to save records.
    If `save_to` is directory then create file with prefix and time else create file as `save_to`.

    :param save_to: Where to save records.
    :param prefix: File name prefix.
    :param create_dir: Create directory if save_to is directory. default True.
    :return:
    """
    if save_to:
        save_to = os.path.abspath(save_to)
        if os.path.exists(os.path.dirname(save_to)) and not os.path.exists(save_to):
            return save_to
        else:
            if create_dir:
                create_dir_if_not_exists(save_to)
            return os.path.join(save_to, '%s%d.csv' % (prefix, time.time()))


def calc_md5(content):
    """Calculate md5sum for a string content"""
    import hashlib
    md5obj = hashlib.md5()
    md5obj.update(content.encode('utf8'))
    md5 = md5obj.hexdigest()
    return md5


def split_parameter(params: dict, param_name: str, default_value=None, raise_error_if_not_exists=False):
    """
    Split `param_name` from `params`, delete it from `params`.

    :param params: type of dict, parameter indexed by param_name.
    :param param_name: parameter's name.
    :param default_value: Default value.
    :param raise_error_if_not_exists: Raise an error is `param_name` not in `params`.
    :return: tuple value of parameter and remained params.
    :raise: KeyError, if raise_error_if_not_exists and `param_name` not in params.
    :raise AssertError, params must be type of dict.
    """
    assert isinstance(params, dict), f'`params` must be type of dict, now {type(params)}'
    if param_name in params:
        value = params[param_name]
        del params[param_name]
        return value, params
    elif raise_error_if_not_exists:
        raise KeyError(f'{param_name} not in `params!`')
    return default_value, params


def get_function_name(func):
    """
    Get function name with function instance.

    :param func: function instance.
    :return: str, function name
    """
    if isinstance(func, partial):
        func_name = func.func.__name__
    else:
        func_name = func.__name__
    return func_name


def del_k_in_dict_if_exist(d_, *keys):
    """
    Delete key in keys from dict type of d_
    :param d_: dict, dictionary.
    :param keys: keys to delete
    :return: a new copy of d_
    """
    new_d_ = {}
    for k, v in d_.items():
        if k not in keys:
            new_d_[k] = v
    return new_d_


def get_value_in_dict(d: dict, name: str, default=None, if_type_tolist=None, force_has_value: bool = False):
    """
    Get value in dict data with giving name.
    :param d: Dict value.
    :param name: value key name.
    :param default: If not has value, return default.
    :param if_type_tolist: tolist or not.
    :param force_has_value: Raise ValueError if name not in d.
    :return:
    """
    assert isinstance(d, dict) or (
            isinstance(d, (list, tuple)) and all([len(d_) == 2 and isinstance(d_[0], str) for d_ in d])), 'type error'
    d = dict(d)
    if name in d:
        if if_type_tolist and isinstance(d[name], if_type_tolist):
            return [d[name]]
        else:
            return d[name]
    elif not force_has_value:
        return default
    else:
        raise ValueError(f'{name} not in {d}')
    #
    # if __name__ == '__main__':
    #     print(get_reversed_cnt([(1, 2), (2, 1), (3, 3)]))


def cmp_ab_with_color(a, b, pos_better: bool = True, diff_only: bool = False):
    """
    Compare a, b with percent change. Finally color the returned text.
    Computation: (a - b) / (b + 1e-6)
    :param a: value
    :param b: value
    :param pos_better: if pos_better and  a > b, color is red else negative. Default True.
    :param diff_only: return diff only or not. Default False.
    :return: format is a(diff)
    """
    boosting = (a - b) / (b + 1e-6)
    cp = ColorPrinter()
    if pos_better:
        if boosting >= 0:
            diff = cp.color_text('+{:.4f}'.format(boosting), 'red')
        else:
            diff = cp.color_text('{:.4f}'.format(boosting))
    else:
        if boosting <= 0:
            diff = cp.color_text('{:.4f}'.format(boosting), 'red')
        else:
            diff = cp.color_text('+{:.4f}'.format(boosting))
    if diff_only:
        return diff
    else:
        if isinstance(a, float):
            return "{:.4f}({})".format(a, diff)
        else:
            return "{}({})".format(a, diff)


def get_files_in_dir(*directories, recursive=False, extension: list = None) -> List[str]:
    """
    Get files in the following directory.

    Args:
        *directories: list of directories.
        recursive: Recursive or not.
        extension: File extension.

    Returns: Samples list
    """
    check_directory_exists(*directories, prefixes='get_file_in_dir')
    samples = []
    if not isinstance(extension, (list, tuple)):
        extension = [extension]
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            if root == directory or recursive:
                samples.extend([os.path.join(root, f) for f in files
                                if extension is None or any(f.lower().endswith(ext) for ext in extension)])
    return samples


def split_dataset(data, partition=None, shuffle=True):
    """
    Split dataset into `train`, `valid` and `test`.
        1 - train_ratio - valid_ratio will be used as test ratio.

    :param data: Data to be split.
    :param partition: list. Training, Validation, Test part ratio.
    :param shuffle: Shuffle data or not, default True.
    :return: tuple (train, valid, test)
    """
    if partition is None:
        partition = [0.7, 0.1, 0.2]
    assert len(partition) >= 2 and sum(partition[:2]) <= 1, 'Ratio can not larger than 1'
    train_ratio = partition[0]
    valid_ratio = partition[1]
    if len(partition) > 2:
        test_ratio = partition[2]
    else:
        test_ratio = 1 - train_ratio - valid_ratio
    total_samples = len(data)

    if isinstance(data, np.ndarray):
        assert len(data.shape) == 2, 'Only dim=2 support for splitting dataset.'
        if shuffle:
            np.random.seed(901005)
            np.random.shuffle(data)
        return (data[:int(total_samples * train_ratio), :],
                data[int(total_samples * train_ratio):
                     int(total_samples * (train_ratio + valid_ratio)), :],
                data[int(total_samples * (train_ratio + valid_ratio)):
                     int(total_samples * (train_ratio + valid_ratio + test_ratio)), :])
    else:
        if shuffle:
            random.seed(901005)
            random.shuffle(data)
        return (data[:int(total_samples * train_ratio)],
                data[int(total_samples * train_ratio):
                     int(total_samples * (train_ratio + valid_ratio))],
                data[int(total_samples * (train_ratio + valid_ratio)):
                     int(total_samples * (train_ratio + valid_ratio + test_ratio))])


def get_attr(args: object, attr, default=None):
    try:
        return args.__getattribute__(attr)
    except:
        return default
# if __name__ == '__main__':
#     results = get_files_in_dir('/Users/zhangzhiwei/data', recursive=True, extension='.txt')
#     print('\n'.join(results))
