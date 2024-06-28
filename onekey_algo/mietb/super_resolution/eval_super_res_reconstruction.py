# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2023/04/15
# Forum: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2023 All Rights Reserved.
import argparse
import os

import nibabel as nib
import numpy as np
import torch
import tqdm

from onekey_algo.mietb.super_resolution.rcan import RCAN
from onekey_algo.mietb.utils import normalize, denormalize, clip
from onekey_algo.utils.about_log import logger


def init(scale, model_path=None, img_range=255):
    model_path = model_path or os.path.join(os.environ.get('ONEKEY_HOME'), 'pretrain', f'RCAN_BIX{scale}.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    model = RCAN(scale=scale, n_colors=3, img_range=img_range)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    model.to(device)
    return model, device


def inference(input_data, model, device, scale, img_range=255, save_dir=None):
    assert scale in [2, 4], f'{scale}不支持的重建倍数，目前只支持2x和4x的超清采样。'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(input_data, str):
        if os.path.exists(input_data):
            input_data = [input_data]
        else:
            raise ValueError(f'input_data数据错误，{input_data}不存在。')
    for idx, input_data_path in enumerate(input_data):
        filename = os.path.basename(input_data_path)
        logger.info("count: {}, data path: {}".format(idx, input_data_path))
        input_nii = nib.load(input_data_path)
        input_nii_data = input_nii.get_fdata()
        input_nii_affine = input_nii.get_affine()

        voxel_size = input_nii.header.get_zooms()
        input_nii.header.set_zooms(tuple([voxel_size[0] / scale, voxel_size[1] / scale, voxel_size[2]]))

        rlt = []
        for jdx in tqdm.tqdm(range(input_nii_data.shape[2])):
            img = input_nii_data[:, :, jdx:jdx + 1].repeat(3, axis=2)
            _min, _max = img.min(), img.max()
            img = normalize(_min, _max, img).astype(np.float32) * 255.
            img_torch = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
            output = model(img_torch)
            output = output[0].cpu().numpy().transpose(1, 2, 0)[:, :, 0]
            # if args.save_rlt and jdx % 10 == 0:
            #     filepath = os.path.join(args.dst_dir, filename[:-7])
            #     os.makedirs(filepath, exist_ok=True)
            #     img = cv2.resize(img, (img.shape[0] * args.scale, img.shape[1] * args.scale),
            #                      interpolation=cv2.INTER_NEAREST)
            #     cv2.imwrite(os.path.join(filepath, "{:03d}_input.png".format(jdx)), img)
            #     cv2.imwrite(os.path.join(filepath, "{:03d}_output_{}X.png".format(jdx, args.scale)), output)
            output = output / img_range
            output = clip(0, 1, output)
            output = denormalize(_min, _max, output)
            rlt.append(output)
        if save_dir is not None:
            output_nii_path = os.path.join(save_dir, filename)
        else:
            output_nii_path = input_data_path.replace('.nii.gz', f'_X{scale}.nii.gz')
        flow_nii_data = np.stack(rlt, axis=2)
        nib.Nifti1Image(flow_nii_data, input_nii_affine, input_nii.header).to_filename(output_nii_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # changed configs
    parser.add_argument('--input_data', type=str, nargs='*', default="./demo/input",
                        help="path to input data, only support nii.gz data")
    parser.add_argument('--dst_dir', type=str, default=None,
                        help="path to result data")
    parser.add_argument('--scale', type=int, default=4,
                        help="choose your upsample scale factor from [2,4]")
    parser.add_argument('--img_range', type=int, default=255,
                        help="choose img range for test")
    args = parser.parse_args()
    model_path = os.path.join(os.environ.get('ONEKEY_HOME'), 'pretrain', f'RCAN_BIX{args.scale}.pt')
    model, device = init(model_path, scale=args.scale, )
    args.input_data = r'C:\Users\onekey\Project\OnekeyDS\CT\images/0.nii.gz'
    inference(args.input_data, model, device, scale=args.scale, save_dir=args.dst_dir)
