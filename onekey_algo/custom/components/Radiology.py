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



def compute_derivatives(volume):
    # 使用高斯滤波对数据进行平滑处理，减少噪声影响
    smoothed = gaussian_filter(volume, sigma=1)

    # 计算一阶和二阶导数
    grad_x = np.gradient(smoothed, axis=0)
    grad_y = np.gradient(smoothed, axis=1)
    grad_z = np.gradient(smoothed, axis=2)

    grad_xx = np.gradient(grad_x, axis=0)
    grad_yy = np.gradient(grad_y, axis=1)
    grad_zz = np.gradient(grad_z, axis=2)
    grad_xy = np.gradient(grad_x, axis=1)
    grad_xz = np.gradient(grad_x, axis=2)
    grad_yz = np.gradient(grad_y, axis=2)

    return grad_xx, grad_yy, grad_zz, grad_xy, grad_xz, grad_yz


def compute_hessian_features(volume, mask):
    if isinstance(volume, SimpleITK.Image) or isinstance(mask, SimpleITK.Image):
        volume = SimpleITK.GetArrayFromImage(volume)
        mask = SimpleITK.GetArrayFromImage(mask)
    grad_xx, grad_yy, grad_zz, grad_xy, grad_xz, grad_yz = compute_derivatives(volume)
    hessian_features = []

    # 遍历每个体素
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                # 构造每个体素的海森矩阵
                H = np.array([
                    [grad_xx[i, j, k], grad_xy[i, j, k], grad_xz[i, j, k]],
                    [grad_xy[i, j, k], grad_yy[i, j, k], grad_yz[i, j, k]],
                    [grad_xz[i, j, k], grad_yz[i, j, k], grad_zz[i, j, k]]
                ])

                # 计算特征值
                eigvals = np.linalg.eigvals(H)
                if mask[i, j, k] == 1:
                    hessian_features.append(eigvals)

    # 转换为numpy数组以方便进一步处理
    hessian_features = np.array(hessian_features)
    featurs = {}
    featurs.update(get_stats_features(hessian_features[:, 0], prefix='AxisX'))
    featurs.update(get_stats_features(hessian_features[:, 1], prefix='AxisY'))
    featurs.update(get_stats_features(hessian_features[:, 2], prefix='AxisZ'))
    return featurs


def box_counting(data, box_size):
    """
    计算给定box大小下非空box的数量
    """
    sampled_data = zoom(data, zoom=1 / box_size, order=0)
    return np.sum(sampled_data > 0)


def compute_curvature(vertices, faces):
    """
    计算曲率：高斯曲率和平均曲率
    """
    gaussian_curvature = np.zeros(len(vertices))
    mean_curvature = np.zeros(len(vertices))

    # 邻接表
    vertex_faces = {v: [] for v in range(len(vertices))}
    for face_id, face in enumerate(faces):
        for v in face:
            vertex_faces[v].append(face_id)

    # 计算每个顶点的曲率
    for i, vertex in enumerate(vertices):
        adjacent_faces = [faces[fid] for fid in vertex_faces[i]]
        edge_vectors = []
        area_sum = 1e-6

        for face in adjacent_faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge_vectors.append(v1 - v0)
            edge_vectors.append(v2 - v0)

            area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
            area_sum += area

        gaussian_curvature[i] = (2 * np.pi - np.sum(edge_vectors)) / area_sum
        mean_curvature[i] = np.sum([np.linalg.norm(e) for e in edge_vectors]) / area_sum

    return gaussian_curvature, mean_curvature


def get_percentile(arr, percentiles=None, prefix=''):
    if percentiles is None:
        percentiles = [i * 10 for i in range(1, 10)]
    features = {}
    for percentile in percentiles:
        pstr = f'{percentile}Percentile'
        value = np.percentile(arr, percentile)
        features[f"{prefix}{pstr}"] = value
    return features


def get_stats_features(arr, percentiles=None, prefix=''):
    features = {}
    features[f"{prefix}Min"] = np.min(arr)
    features[f"{prefix}Max"] = np.max(arr)
    features[f"{prefix}Mean"] = np.mean(arr)
    features[f"{prefix}Std"] = np.std(arr)
    features.update(get_percentile(arr, percentiles, prefix))
    return features


# 提取特征
def compute_topology_features(mask, mask_spec_id=1):
    fnames = ['original_topology_Volume', 'original_topology_SurfaceArea', 'original_topology_ConvexVolume',
              'original_topology_EulerCharacteristic',
              'original_topology_GaussianCurvatureMean', 'original_topology_GaussianCurvatureSTD',
              'original_topology_GaussianCurvatureMin', 'original_topology_GaussianCurvatureMax',
              'original_topology_MeanCurvatureMean', 'original_topology_MeanCurvatureSTD',
              'original_topology_MeanCurvatureMin', 'original_topology_MeanCurvatureMax',
              'original_topology_Skeleton Length',
              'original_topology_GaussianCurvature10Percentile',
              'original_topology_GaussianCurvature20Percentile', 'original_topology_GaussianCurvature30Percentile',
              'original_topology_GaussianCurvature40Percentile', 'original_topology_GaussianCurvature50Percentile',
              'original_topology_GaussianCurvature60Percentile', 'original_topology_GaussianCurvature70Percentile',
              'original_topology_GaussianCurvature80Percentile', 'original_topology_GaussianCurvature90Percentile',
              'original_topology_MeanCurvature10Percentile', 'original_topology_MeanCurvature20Percentile',
              'original_topology_MeanCurvature30Percentile', 'original_topology_MeanCurvature40Percentile',
              'original_topology_MeanCurvature50Percentile', 'original_topology_MeanCurvature60Percentile',
              'original_topology_MeanCurvature70Percentile', 'original_topology_MeanCurvature80Percentile',
              'original_topology_MeanCurvature90Percentile']
    try:
        if mask_spec_id is None:
            mask = (mask != 0).astype(np.uint8)
        else:
            mask = (mask == mask_spec_id).astype(np.uint8)
        labeled_matrix, num_features = label(mask)
        region = mask
        # 形态学特征提取
        props = regionprops(region.astype(int))[0]
        volume = props.area  # 体积
        bbox = props.bbox  # 包围盒
        centroid = props.centroid  # 质心
        convex_hull = ConvexHull(props.coords)  # 凸包
        convex_volume = convex_hull.volume  # 凸包体积

        # 提取表面网格
        verts, faces, normals, values = marching_cubes(region, level=0)
        surface_area = np.sum(np.linalg.norm(np.cross(
            verts[faces[:, 1]] - verts[faces[:, 0]],
            verts[faces[:, 2]] - verts[faces[:, 0]],
        ), axis=1)) / 2

        # 欧拉特征数（拓扑学）
        euler_characteristic = len(verts) - len(faces)

        # 计算高斯曲率和平均曲率
        gaussian_curvature, mean_curvature = compute_curvature(verts, faces)

        # 计算骨架（线性特征长度）
        skeleton = skeletonize_3d(region)
        skeleton_length = np.sum(skeleton)

        # 记录特征
        features = {
            "Volume": volume,
            "SurfaceArea": surface_area,
            "ConvexVolume": convex_volume,
            "EulerCharacteristic": euler_characteristic,
            "GaussianCurvatureMean": np.mean(gaussian_curvature),
            "GaussianCurvatureSTD": np.mean(gaussian_curvature),
            "GaussianCurvatureMin": np.min(gaussian_curvature),
            "GaussianCurvatureMax": np.max(gaussian_curvature),
            "MeanCurvatureMean": np.mean(mean_curvature),
            "MeanCurvatureSTD": np.mean(mean_curvature),
            "MeanCurvatureMin": np.min(mean_curvature),
            "MeanCurvatureMax": np.max(mean_curvature),
            "Skeleton Length": skeleton_length,
        }
        features.update(get_percentile(gaussian_curvature, prefix='GaussianCurvature'))
        features.update(get_percentile(gaussian_curvature, prefix='MeanCurvature'))

        return {f"original_topology_{k}": v for k, v in features.items()}
    except Exception as e:
        error_info = str(e).split('\n')[0]
        logger.error(f"提取拓扑学特征失败：{error_info}")
        return {fn: np.nan for fn in fnames}


def fractal_dimension_statistics_adjusted(data, max_box_size=None):
    """
    计算三维矩阵的分形维数，并返回其min、max、mean、std等统计指标
    """
    min_box_size = 1
    if max_box_size is None:
        max_box_size = min(data.shape)

    # 存储分形维数
    fractal_dimensions = []
    fractal_lacunarity = []

    # 遍历不同的最大box大小
    for box_size in np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=10, endpoint=True, dtype=int):
        if box_size < 1:
            continue
        #
        x, y, z = data.shape
        sums = []

        # Slide the box through the 3D volume
        for i in range(0, x - box_size + 1, box_size):
            for j in range(0, y - box_size + 1, box_size):
                for k in range(0, z - box_size + 1, box_size):
                    # Extract the 3D window
                    window = data[i:i + box_size, j:j + box_size, k:k + box_size]
                    sums.append(np.sum(window))

        sums = np.array(sums)
        mean = np.mean(sums)
        variance = np.var(sums)

        # Calculate lacunarity
        lacunarity = variance / (mean ** 2) if mean != 0 else 0
        fractal_lacunarity.append(lacunarity)

        # 在当前最大box大小下，计算多个分形维数
        counts = []
        inner_box_sizes = np.logspace(np.log10(min_box_size), np.log10(box_size), num=5, endpoint=True, dtype=int)
        inner_box_sizes = np.unique(inner_box_sizes)  # 去重

        for inner_box_size in inner_box_sizes:
            if inner_box_size >= 1:
                counts.append(box_counting(data, inner_box_size))

        # 检查是否有足够的数据点进行线性拟合
        if len(counts) > 1:
            log_box_sizes = np.log(inner_box_sizes)
            log_counts = np.log(counts)

            # 执行线性拟合
            try:
                coeffs = np.polyfit(log_box_sizes, log_counts, 1)
                fractal_dim = -coeffs[0]
                if not np.isnan(fractal_dim):
                    fractal_dimensions.append(fractal_dim)
            except np.linalg.LinAlgError:
                # 忽略无法拟合的数据点
                continue
    # 计算分形维数的统计指标
    features = {'FractalMin': np.min(fractal_dimensions),
                'FractalMax': np.max(fractal_dimensions),
                'FractalMean': np.mean(fractal_dimensions),
                'FractalSTD': np.std(fractal_dimensions),
                'LacunarityMin': np.min(fractal_lacunarity),
                'LacunarityMax': np.max(fractal_lacunarity),
                'LacunarityMean': np.mean(fractal_lacunarity),
                'LacunaritySTD': np.std(fractal_lacunarity), }
    features.update(get_percentile(fractal_dimensions, prefix='Fractal'))
    features.update(get_percentile(fractal_lacunarity, prefix='Lacunarity'))
    return {f"original_fractal_{k}": v for k, v in features.items()}



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
