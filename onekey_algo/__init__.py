# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/1/18
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
__VERSION__ = '2.4.0'

import json
import os

import yaml


def hello_onekey():
    from onekey_algo.custom.components import comp1
    print(f"""
#######################################################
##          欢迎使用Onekey，当前版本：{__VERSION__}          ##
##       OnekeyAI助力科研，我们将竭诚为您服务！      ##
#######################################################
""")


if os.environ.get('ONEKEY_HOME'):
    ONEKEYDS_ROOT = os.path.join(os.environ.get('ONEKEY_HOME'), 'OnekeyDS')
else:
    ONEKEYDS_ROOT = os.environ.get('ONEKEY_HOME') or os.path.expanduser(r'~/Project/OnekeyDS')


class OnekeyDS:
    ct = os.path.join(ONEKEYDS_ROOT, 'CT')
    ct_features = os.path.join(ONEKEYDS_ROOT, 'CT', 'rad_features.csv')
    tumour_stroma = os.path.join(ONEKEYDS_ROOT, 'tumour_stroma')
    complaint = os.path.join(ONEKEYDS_ROOT, "complaint.csv")
    grade = os.path.join(ONEKEYDS_ROOT, 'grade.csv')
    Metabonomics = os.path.join(ONEKEYDS_ROOT, 'Metabonomics.csv')
    phy_bio = os.path.join(ONEKEYDS_ROOT, 'phy_bio.csv')
    survival = os.path.join(ONEKEYDS_ROOT, 'survival.csv')


def get_config(directory=os.getcwd(), config_file='config.txt') -> dict:
    if os.path.exists(os.path.join(directory, config_file)):
        with open(os.path.join(directory, config_file), encoding='utf8') as c:
            content = c.read()
            if '\\\\' not in content:
                content = content.replace('\\', '\\\\')
            if config_file.endswith('.txt'):
                config = json.loads(content)
            elif config_file.endswith('.yaml'):
                config = yaml.load(content, Loader=yaml.FullLoader)
            return config
    else:
        return {}


def get_param_in_cwd(param: str, default=None, **kwargs):
    directory = kwargs.get('directory', os.getcwd())
    config_file = 'config.yaml' if os.path.exists(os.path.join(directory, 'config.yaml')) else 'config.txt'
    config = get_config(directory, config_file)
    ret = config.get(param, None) or default
    if isinstance(ret, str) and 'ONEKEY_HOME' in ret:
        ret = ret.replace('ONEKEY_HOME', os.environ.get('ONEKEY_HOME'))
    if isinstance(ret, str) and 'ONEKEYDS_HOME' in ret:
        ret = ret.replace('ONEKEYDS_HOME', ONEKEYDS_ROOT)
    return ret


if __name__ == '__main__':
    okds = OnekeyDS()
    print(okds.ct)
