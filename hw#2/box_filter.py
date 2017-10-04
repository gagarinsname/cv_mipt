from __future__ import print_function
from sys import argv
import os.path
import cv2
import numpy as np


def integral_image(src):
    dst = np.array(list(map(lambda x: np.cumsum(x), list(src))), dtype=np.int32)
    dst = dst.T
    dst = np.array(list(map(lambda x: np.cumsum(x), list(dst))), dtype=np.int32)
    dst = dst.T
    return dst


def box_flter(src_path, dst_path, w, h):
    # Read an image
    src = cv2.imread(src_path)
    assert src is not None

    # Convert RGB to grayscale
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_h, src_w = src.shape
    # src = cv2.resize(src, (src_w // 10, src_h // 10), interpolation=cv2.INTER_LINEAR)
    # src_h, src_w = src.shape

    # Filtering an image
    src_int = integral_image(src)
    dst = np.zeros(src.shape)
    for y in range(src_h):
        for x in range(src_w):
            x_a, y_a = max(0, x-w//2), max(0, y-h//2)
            x_b, y_b = min(src_w-1, x+w//2), max(0, y-h//2)
            x_c, y_c = max(0, x-w//2), min(src_h-1, y+h//2)
            x_d, y_d = min(src_w-1, x+w//2), min(src_h-1, y+h//2)
            dst[y, x] = src_int[y_d, x_d] - src_int[y_b, x_b] - src_int[y_c, x_c] + src_int[y_a, x_a]
            dst[y, x] //= (w*h)

    # Normalization
    # dst = (dst / (w*h) + 0.51).astype(int)

    # Saving the result
    cv2.imwrite(dst_path, dst)
    return


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_flter(*argv[1:])
