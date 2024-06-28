import numpy as np


def normalize(_min, _max, _data):
    return (_data - _min) / (_max - _min)


def denormalize(_min, _max, _data):
    return (_max - _min) * _data + _min


def clip(_min, _max, _data):
    return np.clip(_data, _min, _max)