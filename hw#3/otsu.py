from __future__ import print_function
from sys import argv
import os.path
import numpy as np
import cv2


def otsu(src_path, dst_path):
    # Read an image
    src = cv2.imread(src_path)
    assert src is not None

    # Convert RGB to grayscale
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    n_bins = 256
    src_hist = np.histogram(src.ravel(), bins=n_bins, range=[0, n_bins])[0]
    src_hist = src_hist.astype(float) / src_hist.max()
    cumulative_hist = src_hist.cumsum()
    bins = np.arange(n_bins)

    within_class_var_min = np.inf
    thresh = -1
    for i in range(1, n_bins):
        w0, w1 = cumulative_hist[i], cumulative_hist[n_bins-1] - cumulative_hist[i]  # cumulative sum for classes
        if w0 == 0 or w1 == 0:
            continue
        s0, s1 = np.hsplit(src_hist, [i])  # weights
        b0, b1 = np.hsplit(bins, [i])  # hist indices
        # mean and variance variables
        m0, m1 = np.sum(s0 * b0) / w0, np.sum(s1 * b1) / w1
        v0, v1 = np.sum(((b0 - m0) ** 2) * s0) / w0, np.sum(((b1 - m1) ** 2) * s1) / w1
        # minimization of the within class variance
        within_class_var = v0 * w0 + v1 * w1
        if within_class_var < within_class_var_min:
            within_class_var_min = within_class_var
            thresh = i-1

    assert thresh != -1

    dst = np.copy(src)
    dst[dst <= thresh] = 0
    dst[dst > thresh] = 255

    # Saving the result
    cv2.imwrite(dst_path, dst)
    pass


if __name__ == '__main__':
    assert len(argv) == 3
    assert os.path.exists(argv[1])
    otsu(*argv[1:])
