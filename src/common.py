import numpy as np
from matplotlib import pyplot as plt


def show(data):
    plt.imshow(data, interpolation='nearest')

    plt.show()


def walk_matrix_to_line(m):
    """
    example: for 4x4:
     1--2  9--10
        |  |  |
     4--3  8  11
     |     |  |
     5--6--7  12
              |
     16-15-14-13

    :return: [1, 2, 3, 4, ..., 16]
    """
    i = 0
    j = 0
    r = 0
    m_w, m_h = m.shape
    assert m_w == m_h
    s = m_w
    T = True

    while r < s:
        if T:
            yield m[j, i]
        else:
            yield m[i, j]

        for _ in range(r):
            i += 1
            if T:
                yield m[j, i]
            else:
                yield m[i, j]
        for _ in range(r):
            j -= 1
            if T:
                yield m[j, i]
            else:
                yield m[i, j]
        r += 1
        i += 1
        j, i = i, j
        T = not T


def bit_array_to_int(arr):
    return int(''.join(str(x) for x in (arr).astype(np.int)), base=2)
