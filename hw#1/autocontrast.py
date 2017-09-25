from __future__ import print_function
from sys import argv
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

def autocontrast(src_path, dst_path, white_perc, black_perc):

    def autocontrast_layer(img):
        imgValues = sorted(src[:,:,i].ravel())
        img_min = imgValues[int(black_perc * im_size)]
        img_max = imgValues[int((1 - white_perc) * im_size)]

        img[img > img_max] = img_max
        img[img < img_min] = img_min

        img = np.array(map(lambda x: 255 * (x - img_min) / (img_max - img_min), img.ravel()))
        img.shape = [h,w]
        return img

    src = cv2.imread(src_path)
    assert src is not None

    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    h,w,_ = src.shape
    im_size = h * w

    dst = list()
    for i in range(src.shape[2]):
        img = src[:,:,i]
        dst.append(autocontrast_layer(img))

    dst = np.dstack(dst).astype(np.uint8)

    cv2.imwrite(dst_path, dst)
    pass


if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = float(argv[3])
    argv[4] = float(argv[4])

    assert 0 <= argv[3] < 1
    assert 0 <= argv[4] < 1

    autocontrast(*argv[1:])
