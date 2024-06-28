# README

这是一个用Python编写的医学图像处理程序 **M**edical **I**mage **E**nhancement **T**ool **B**ox(MIETB) ，可以用于各种医学图像处理相关的任务。

## 效果说明

该程序可以实现以下功能：

- 图像超分辨率重建

## 依赖环境

该程序依赖于以下Python库：

- numpy

- matplotlib

- opencv-python

- scikit-image

- scikit-learn

   可以使用以下命令安装：

```pip install numpy matplotlib opencv-python scikit-image scikit-learn```



## 运行方式

根据任务不同，运行方式有所区别。



### 超分辨率重建

在命令行中运行以下命令：

```python sr_demo.py --src_dir path/to/your/input/data --dst_dir path/to/your/output/data --scale 4 ```

参数释义：

- src_dir path/to/your/input/data
- dst_dir path/to/your/output/data
- scale 输入一个放大倍数，支持[2,3,4,8]倍

## TODO

- 支持更多图像处理方法

- 支持Image2Image Translation 方法，如T1转T2

