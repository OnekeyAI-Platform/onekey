# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/3/22
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import copy
from typing import List, Union

import numpy as np
import pandas as pd

from onekey_algo.utils.about_log import logger


def map2numerical(data: pd.DataFrame, mapping_columns: Union[str, List[str]], inplace=True, map_nan: bool = False):
    """
    把数据集的非数值数据映射成分类数值
    Args:
        data: 数据
        mapping_columns: 需要映射的列
        inplace: bool
        map_nan: 是否映射空值，默认不进行映射，可以在后续任务进行填充

    Returns:

    """
    mapping = {}
    if not inplace:
        new_data = copy.deepcopy(data)
    else:
        new_data = data
    if not isinstance(mapping_columns, list):
        mapping_columns = [mapping_columns]
    assert all(c in data.columns for c in mapping_columns)
    if map_nan:
        for c in mapping_columns:
            unique_labels = {v: idx for idx, v in enumerate(sorted(np.unique(np.array(data[c]).astype(str))))}
            mapping[c] = unique_labels
            new_data[[c]] = new_data[[c]].applymap(lambda x: unique_labels[str(x)])
    else:
        for c in mapping_columns:
            ul = sorted([ul_ for ul_ in np.unique(np.array(data[c]).astype(str)) if ul_ != 'nan'])
            unique_labels = {v: idx for idx, v in enumerate(ul)}
            mapping[c] = unique_labels
            new_data[[c]] = new_data[[c]].applymap(lambda x: unique_labels[str(x)] if str(x) != 'nan' else None)
    return new_data, mapping


def print_join_info(left: pd.DataFrame, right: pd.DataFrame, on='ID'):
    left_set = set(left[on])
    right_set = set(right[on])
    if left_set == right_set:
        logger.info(f'{on}特征完全匹配！')
    else:
        logger.warning(f"存在{on}特征不完全匹配的问题！在左边不在右边的{on}：{left_set - right_set}；"
                       f"在右边不在左边的{on}：{right_set - left_set}")
