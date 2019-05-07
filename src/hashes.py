from PIL import Image
import numpy as np
import pywt
from scipy.fftpack import dct, idct

from common import bit_array_to_int


def avghash(path):
    im = Image.open(path)
    a = np.asanyarray(im.resize((8, 8), Image.ANTIALIAS).convert('L'))
    return bit_array_to_int((a > np.average(a)).reshape([64]))


def dct2d(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')


def idct2d(f):
    return idct(idct(f, norm='ortho').T, norm='ortho').T


def phash(path, mode=1):  # mode==1 is the best so far
    # source: https://hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    im = Image.open(path)
    im_gray_32 = im.resize((32, 32), Image.ANTIALIAS).convert('L')
    a = np.asanyarray(im_gray_32)
    a = a - np.average(a)
    ff = dct2d(a)

    ff = ff[:8, :8]
    if mode & int('01', base=2):
        ff **= 2
    bits = ff.reshape([64])
    avg = np.average(bits)
    if mode & int('10', base=2):
        avg = np.median(bits)
    return bit_array_to_int(bits > avg)


def _dwt_output_to_matrix(low_pass, dd):
    (d1, d2, d3) = dd
    r1 = np.concatenate([low_pass, d1], axis=1)
    r2 = np.concatenate([d2, d3], axis=1)
    return np.concatenate([r1, r2], axis=0)


def _max_level_recur(img):
    if min(img.shape) <= 1:
        return img / 2
    low_pass, (d1, d2, d3) = pywt.dwt2(img, 'haar')
    low_pass_dwt = _max_level_recur(low_pass) / 2
    m = _dwt_output_to_matrix(low_pass_dwt, (d1, d2, d3))
    return m


def pwhash(path):
    im = Image.open(path)
    img = im.resize((64, 64), Image.ANTIALIAS).convert('L')
    img = np.array(img)
    m = _max_level_recur(img)
    m = m[:8, :8]

    avg = np.average(m)  # todo: compute avg per dwt layer instead of whole dwt
    bits = (m > avg).astype(np.int8)

    return bit_array_to_int(bits.reshape([64]))
