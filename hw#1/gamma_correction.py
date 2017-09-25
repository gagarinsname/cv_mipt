from __future__ import print_function
from sys import argv
import os.path
import numpy as np
import cv2
from matplotlib import pyplot

def gamma_correction(src_path, dst_path, a, b):
    src = cv2.imread(src_path)
    assert src is not None

    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    print('original shape:', src.shape)
    print('type: %s' % type(src))
    src = src.astype(float) / 255

    dst = np.array(map(lambda x: a*x**b, src.ravel()))
    dst_max = max(dst)
    if dst_max > 0:
        dst = dst / dst_max * 255
    dst.shape = src.shape
    dst = dst.astype(np.uint8)

    print('result shape:', dst.shape)
    print('type: %s' % type(dst))

    pyplot.subplot(121)
    pyplot.imshow(src)
    pyplot.subplot(122)
    pyplot.imshow(dst)
    pyplot.show()

    cv2.imwrite(dst_path, dst)
    return

if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    gamma_correction(*argv[1:])
