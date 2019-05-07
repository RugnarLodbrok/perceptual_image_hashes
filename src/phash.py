from os.path import join as j
from PIL import Image
import numpy as np

# 1 Reduce size. Like Average Hash, pHash starts with a small image. However, the image is larger than 8x8; 32x32 is a good size. This is really done to simplify the DCT computation and not because it is needed to reduce the high frequencies.
# Reduce color. The image is reduced to a grayscale just to further simplify the number of computations.
# Compute the DCT. The DCT separates the image into a collection of frequencies and scalars. While JPEG uses an 8x8 DCT, this algorithm uses a 32x32 DCT.
# Reduce the DCT. While the DCT is 32x32, just keep the top-left 8x8. Those represent the lowest frequencies in the picture.
# Compute the average value. Like the Average Hash, compute the mean DCT value (using only the 8x8 DCT low-frequency values and excluding the first term since the DC coefficient can be significantly different from the other values and will throw off the average). Thanks to David Starkweather for the added information about pHash. He wrote: "the dct hash is based on the low 2D DCT coefficients starting at the second from lowest, leaving out the first DC term. This excludes completely flat image information (i.e. solid colors) from being included in the hash description."
# Further reduce the DCT. This is the magic step. Set the 64 hash bits to 0 or 1 depending on whether each of the 64 DCT values is above or below the average value. The result doesn't tell us the actual low frequencies; it just tells us the very-rough relative scale of the frequencies to the mean. The result will not vary as long as the overall structure of the image remains the same; this can survive gamma and color histogram adjustments without a problem.
# Construct the hash. Set the 64 bits into a 64-bit integer. The order does not matter, just as long as you are consistent. To see what this fingerprint looks like, simply set the values (this uses +255 and -255 based on whether the bits are 1 or 0) and convert from the 32x32 DCT (with zeros for the high frequencies) back into the 32x32 image:
from scipy.fftpack import dct, idct

from common import bit_array_to_int


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

