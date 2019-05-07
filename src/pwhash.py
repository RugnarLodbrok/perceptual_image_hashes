from os.path import join as j
from PIL import Image

import pywt
import numpy as np

from common import show, bit_array_to_int


def wt1(a):
    """
    :param a:
    :return:
    """


def dwt_output_to_matrix(low_pass, dd):
    (d1, d2, d3) = dd
    r1 = np.concatenate([low_pass, d1], axis=1)
    r2 = np.concatenate([d2, d3], axis=1)
    return np.concatenate([r1, r2], axis=0)


def single_level_example(img):
    low_pass, (d1, d2, d3) = pywt.dwt2(img, 'haar')
    print("------LOW_PASS-----")
    print(low_pass)
    print("--------D1--------")
    print(d1)
    print("--------D2--------")
    print(d2)
    print("--------D3--------")
    print(d3)

    m = dwt_output_to_matrix(low_pass, (d1, d2, d3))
    print(m.shape)
    show(m)


def max_level_recur(img):
    if min(img.shape) <= 1:
        return img / 2
    low_pass, (d1, d2, d3) = pywt.dwt2(img, 'haar')
    low_pass_dwt = max_level_recur(low_pass) / 2
    m = dwt_output_to_matrix(low_pass_dwt, (d1, d2, d3))
    return m


def max_level_example(img):
    show(max_level_recur(img))


def max_level_hashing(img):
    m = max_level_recur(img)
    avg = np.average(m)  # todo: compute avg per dwt layer instead of whole dwt
    bits = (m > avg).astype(np.int8)
    show(bits)


def pwhash(path):
    im = Image.open(path)
    img = im.resize((64, 64), Image.ANTIALIAS).convert('L')
    img = np.array(img)
    m = max_level_recur(img)
    m = m[:8, :8]

    avg = np.average(m)  # todo: compute avg per dwt layer instead of whole dwt
    bits = (m > avg).astype(np.int8)

    return bit_array_to_int(bits.reshape([64]))


if __name__ == '__main__':
    path = j('..', 'data', 'bee-1024.jpg')
    im = Image.open(path)
    img = im.resize((64, 64), Image.ANTIALIAS).convert('L')

    # single_level_example(img)
    # max_level_example(np.array(img))
    print(bin(pwhash(path)))
