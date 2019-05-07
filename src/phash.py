from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct

from common import bit_array_to_int


# source: https://hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

def dct2d(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')


def idct2d(f):
    return idct(idct(f, norm='ortho').T, norm='ortho').T


def phash(path, mode=1):  # mode==1 is the best so far
    im = Image.open(path)
    im_gray_32 = im.resize((32, 32), Image.ANTIALIAS).convert('L')
    a = np.asanyarray(im_gray_32)
    a = a - np.average(a)
    ff = dct2d(a)

    # ff[8:, :] = 0
    # ff[:, 8:] = 0
    ff = ff[:8, :8]
    if mode & int('01', base=2):
        ff **= 2
    bits = ff.reshape([64])
    avg = np.average(bits)
    if mode & int('10', base=2):
        avg = np.median(bits)
    return bit_array_to_int(bits > avg)
