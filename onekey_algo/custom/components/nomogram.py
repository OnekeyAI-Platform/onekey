# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/4/20
# Blog: www.wangqingbaidu.cn
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.
import os
from typing import List, Union, Iterable

import pandas as pd
import rpy2.robjects as robjects
from PIL import Image
from pandas import DataFrame
from rpy2.robjects import globalenv
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

rms = importr("rms")
survival = importr("survival")
RTEMP = """
dd=datadist(rdf)
options(datadist="dd")
f2 <- cph(Surv({duration},{result}) ~ {columns}, data=rdf, x=TRUE, y=TRUE, surv=TRUE)

# med <- Quantile(f2)
surv <- Survival(f2)

png( 
    filename = "{save_name}",
    width = {width},
    height = {height},
    units = "px",
    bg = "white",
    res = 600)
    
nom <- nomogram(f2, fun=list(
{func}
                             # function(x) med(lp=x)
                             ), lp=F, 
                             funlabel=c({funlabel}), fun.at=c({x_range}))
plot(nom, xfrac=.2)
dev.off()
"""
FUNCTEMP = "                             function(x) surv({time}, x)"

RRISKTEMP = """
dd=datadist(rdf)
options(datadist="dd") 
f2 <- lrm({result} ~ {columns}, data = rdf)

png( 
    filename = "{save_name}",
    width = {width},
    height = {height},
    units = "px",
    bg = "white",
    res = 600)

nom <- nomogram(f2, fun= function(x)1/(1+exp(-x)), 
                lp=F, funlabel="Risk", fun.at=c({x_range}))
plot(nom, xfrac=.2)
dev.off()
"""


def nomogram(df: Union[str, DataFrame], duration: str, result: str, columns: Union[str, List[str]],
             survs: Union[int, List[int]], surv_names: Union[str, List[str]] = None, x_range='0.01,0.5,0.99',
             width: int = 8000, height: int = 3200, save_name='nomogram.png', with_r: bool = False) -> Image:
    """
    绘制nomogram图，Nomogram的图存储在当前文件夹下的nomogram.png
    Args:
        df: 数据路径，或者是读取之后的Dataframe格式。
        duration: OS
        result: OST
        columns: 使用那些列计算nomogram
        survs: 生存时间转化成x 年生存率
        surv_names: survs对应的列名。
        x_range:
        width: nomogram分辨率--宽度，默认960
        height: nomogram分辨率--宽度，默认480
        save_name: 保存的文件名。
        with_r: 是否输出R语言代码

    Returns: PIL.Image
    """
    if not isinstance(survs, Iterable):
        survs = [survs]
    if isinstance(surv_names, Iterable):
        assert len(survs) == len(surv_names), f"预测的标尺名称和标尺个数必须相等。"
    else:
        surv_names = [surv_names] * len(survs)
    if not isinstance(columns, Iterable):
        columns = [columns]
    assert all(c_ in df.columns for c_ in [duration, result] + columns), '所有列名必须在df参数中'
    if isinstance(x_range, (list, tuple)):
        x_range = ','.join(map(lambda x: str(x), x_range))
    if isinstance(df, str) and os.path.exists(df):
        df = pd.read_csv(df, header=0)
    pandas2ri.activate()
    rdf = pandas2ri.py2rpy(df)
    globalenv['rdf'] = rdf
    columns = '+'.join(map(lambda x: str(x), columns))
    func = ','.join(FUNCTEMP.format(time=surv) for surv in survs)
    funlabel = ','.join([f'"{surv_name}"' if surv_names is not None else f'"{surv_name} Survival"'
                         for surv_name in surv_names])
    rscript = RTEMP.format(duration=duration, result=result, columns=columns, func=func, funlabel=funlabel,
                           width=width, height=height, save_name=save_name, x_range=x_range)
    if with_r:
        print(rscript)
    robjects.r(rscript)
    return Image.open(save_name)


def risk_nomogram(df: Union[str, DataFrame], result: str, columns: Union[str, List[str]], x_range='0.01,0.5,0.99',
                  width: int = 8000, height: int = 3200, save_name='nomogram.png', with_r: bool = False) -> Image:
    """
    绘制nomogram图，Nomogram的图存储在当前文件夹下的nomogram.png
    Args:
        df: 数据路径，或者是读取之后的Dataframe格式。
        result: OST
        columns: 使用那些列计算nomogram
        x_range: 横坐标的取值区间
        width: nomogram分辨率--宽度，默认960
        height: nomogram分辨率--宽度，默认480
        save_name: 保存的文件名
        with_r: 是否输出r语言代码

    Returns: PIL.Image
    """
    if not isinstance(columns, Iterable):
        columns = [columns]
    assert all(c_ in df.columns for c_ in [result] + columns), '所有列名必须在df参数中'
    if isinstance(x_range, (list, tuple)):
        x_range = ','.join(map(lambda x: str(x), x_range))
    if isinstance(df, str) and os.path.exists(df):
        df = pd.read_csv(df, header=0)
    pandas2ri.activate()
    rdf = pandas2ri.py2rpy(df)
    globalenv['rdf'] = rdf
    columns = '+'.join(map(lambda x: str(x), columns))
    rscript = RRISKTEMP.format(result=result, columns=columns, width=width, height=height, save_name=save_name,
                               x_range=x_range)
    if with_r:
        print(rscript)
    robjects.r(rscript)
    return Image.open(save_name)
