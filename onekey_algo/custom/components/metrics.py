# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/1/19
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import warnings
from typing import Union, List

import numpy as np
import pandas as pd
import sklearn.metrics as sm
from scipy import stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import column_or_1d, check_consistent_length, assert_all_finite
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.multiclass import type_of_target

from onekey_algo.custom.components.delong import calc_95_CI


def calc_array_95ci(data, confidence=0.95):
    data = column_or_1d(np.array(data))
    std = stats.tstd(data)
    sem = stats.sem(data)
    return stats.t.interval(confidence, df=len(data) - 1, loc=np.mean(data), scale=sem)


def calc_value_95ci(a, b) -> tuple:
    """
    实现： Wilson, E. B. "Probable Inference, the Law of Succession, and Statistical Inference,"
          Journal of the American Statistical Association, 22, 209-212 (1927).

    Args:
        a: 分子
        b: 分母

    Returns: 95% CI [lower, upper]

    """
    sum_value = a + b + 1e-6
    ratio = a / sum_value
    std = (ratio * (1 - ratio) / sum_value) ** 0.5
    return max(0, ratio - 1.96 * std), min(ratio + 1.96 * std, 1)


def map_ci(ci):
    ci_float = [float(f"{i_:.6f}") for i_ in ci]
    ci_float[0] = ci_float[0] if not np.isnan(ci_float[0]) else 1
    ci_float[1] = ci_float[1] if not np.isnan(ci_float[1]) else 1
    # print(ci_float)
    return ci_float


def check_pos_label_consistency(pos_label, y_true):
    """Check if `pos_label` need to be specified or not.

    In binary classification, we fix `pos_label=1` if the labels are in the set
    {-1, 1} or {0, 1}. Otherwise, we raise an error asking to specify the
    `pos_label` parameters.

    Parameters
    ----------
    pos_label : int, str or None
        The positive label.
    y_true : ndarray of shape (n_samples,)
        The target vector.

    Returns
    -------
    pos_label : int
        If `pos_label` can be inferred, it will be returned.

    Raises
    ------
    ValueError
        In the case that `y_true` does not have label in {-1, 1} or {0, 1},
        it will raise a `ValueError`.
    """
    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in 'OUS' or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and pos_label is not "
            f"specified: either make y_true take value in {{0, 1}} or "
            f"{{-1, 1}} or pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1.0

    return pos_label


def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True targets of binary classification.

    y_score : ndarray of shape (n_samples,)
        Estimated probabilities or output of a decision function.

    pos_label : int or str, default=None
        The label of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    fps : ndarray of shape (n_thresholds,)
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : ndarray of shape (n_thresholds,)
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : ndarray of shape (n_thresholds,)
        Decreasing score values.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    pos_label = check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    tns = fps[-1] - fps
    fns = tps[-1] - tps
    return fps, tps, tns, fns, y_score[threshold_idxs]


def any_curve(y_true, y_score, *, pos_label=None, sample_weight=None):
    fps, tps, tns, fns, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    if tns[0] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "true negative value should be meaningless",
                      UndefinedMetricWarning)
        tnr = np.repeat(np.nan, tns.shape)
    else:
        tnr = tns / tns[0]

    if fns[0] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "false negative value should be meaningless",
                      UndefinedMetricWarning)
        fnr = np.repeat(np.nan, fns.shape)
    else:
        fnr = fns / fns[0]

    return fpr, tpr, tnr, fnr, thresholds


def calc_sens_spec(y_true, y_score, **kwargs):
    fpr, tpr, tnr, fnr, thresholds = any_curve(y_true, y_score)
    idx = 0
    maxv = -1e6
    for i, v in enumerate(tpr - fpr):
        if v > maxv:
            maxv = v
            idx = i
    #    idx = np.argmax(tpr - fpr)
    return tpr[idx], tnr[idx], thresholds[idx]


def analysis_pred_binary(y_true: Union[List, np.ndarray, pd.DataFrame], y_score: Union[List, np.ndarray, pd.DataFrame],
                         y_pred: Union[List, np.ndarray, pd.DataFrame] = None, alpha=0.95,
                         use_youden: bool = True, with_aux_ci: bool = False):
    """

    Args:
        y_true:
        y_score:
        y_pred:
        alpha: 0.95
        use_youden: 是否使用youden指数
        with_aux_ci: 是否输出额外的CI

    Returns:

    """
    aux_ci = {}
    if isinstance(y_score, (list, tuple)):
        y_score = np.array(y_score)
    y_true = column_or_1d(np.array(y_true))
    assert sorted(np.unique(y_true)) == [0, 1], f"结果必须是2分类！"
    assert len(y_true) == len(y_score), '样本数必须相等！'
    if len(y_score.shape) == 2:
        y_score = column_or_1d(y_score[:, 1])
    elif len(y_score.shape) > 2:
        raise ValueError(f"y_score不支持>2列的数据！现在是{y_score.shape}")
    else:
        y_score = column_or_1d(y_score)
    tpr, tnr, thres = calc_sens_spec(y_true, y_score)
    if y_pred is None:
        y_pred = np.array(y_score >= (thres if use_youden else 0.5)).astype(int)
    acc = np.sum(y_true == y_pred) / len(y_true)
    tp = np.sum(y_true[y_true == 1] == y_pred[y_true == 1])
    tn = np.sum(y_true[y_true == 0] == y_pred[y_true == 0])
    fp = np.sum(y_pred[y_true == 0] == 1)
    fn = np.sum(y_pred[y_true == 1] == 0)
    ppv = tp / (tp + fp + 1e-6)
    aux_ci['ppv'] = calc_value_95ci(tp, fp)
    npv = tn / (tn + fn + 1e-6)
    aux_ci['npv'] = calc_value_95ci(tn, fn)
    auc, ci = calc_95_CI(y_true, y_score, alpha=alpha, with_auc=True)
    if not use_youden:
        tpr = tp / (tp + fn + 1e-6)
        tnr = tn / (fp + tn + 1e-6)
    aux_ci['sens'] = calc_value_95ci(tp, fn)
    aux_ci['spec'] = calc_value_95ci(tn, fp)
    f1 = 2 * tpr * ppv / (ppv + tpr)
    # print(tp, tn, fp, fn)
    if with_aux_ci:
        return acc, auc, map_ci(ci), tpr, map_ci(aux_ci['sens']), tnr, map_ci(aux_ci['spec']), \
               ppv, map_ci(aux_ci['ppv']), npv, map_ci(aux_ci['npv']), ppv, tpr, f1, thres
    else:
        return acc, auc, map_ci(ci), tpr, tnr, ppv, npv, ppv, tpr, f1, thres


if __name__ == '__main__':
    y_true_ = [0, 0, 1, 1, 1, 1, 0]
    y_pred_ = [1, 1, 0, 0, 0, 0, 1]
    event_ = [1, 1, 0, 0, 0, 0, 1]
    y_pred_1 = [0.51, 0.61, 0.0, 0.01, 0.53, 0.99, 0.88]
    y_pred_2 = [1, 0.61, 1, 0.01, 0.53, 0.99, 0.88]
    print(analysis_pred_binary(y_true_, y_pred_1, with_aux_ci=True))
    print(calc_value_95ci(0, 2))
    # print(IDI(y_true_, pred_x=y_pred_2, pred_y=y_pred_1))
    # print(NRI(y_true_, y_pred_, event_))
