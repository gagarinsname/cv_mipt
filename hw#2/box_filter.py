from __future__ import print_function
from sys import argv
import os.path
import cv2
import numpy as np

def box_flter(src_path, dst_path, w, h):
    # Read an image
    src = cv2.imread(src_path)
    assert src is not None
    # Convert RGB to grayscale
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Filtering an image
    kernel = 1.0 / 9 * np.ones((h, w))
    dst = cv2.filter2D(src.astype(float), -1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    # Saving the result
    cv2.imwrite(dst_path, dst)
    return


if __name__ == '__main__':
    assert len(argv) == 5
    print (argv[1])
    assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_flter(*argv[1:])
