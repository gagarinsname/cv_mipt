from __future__ import print_function
from sys import argv
import cv2
import numpy as np
from miptcv_utils import *
# Optimal parameters:
# <input> res_ht.png res_lines.txt 0.1 1 <n_lines> 5 0.2


def visualize_lines(img, lines):
    img = img.astype(np.uint8)

    h, w = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for k, b in lines:
        if abs(b) > h + w or abs(k) > h+w:
            # print('line will be drawn in x = k_ y + b_')
            k_inv = 1.0 / k
            b_inv = -b / k
            for y in range(h):
                x = int(k_inv * y + b_inv)
                if 0 <= x < w:
                    cv2.line(img, (x, y), (x, y), (0, 255, 0), thickness=2)
        else:
            for x in range(w):
                y = int(k * x + b)
                if 0 <= y < h:
                    cv2.line(img, (x, y), (x, y), (0, 255, 0), thickness=2)

    cv2.imwrite(r"res_lines.png", img)

def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)

    return magnitude


def hough_transform(img, theta, rho):
    h, w = img.shape

    rho_max = max(h, w)
    rho_array = np.arange(-rho_max, rho_max, step=rho)
    theta_array = np.arange(-np.pi/2+theta, np.pi/2, step=theta)

    sinuses = np.sin(theta_array)
    cosinuses = np.cos(theta_array)

    ht_map = np.zeros([len(rho_array), len(theta_array)], dtype=int)
    y_nonzero, x_nonzero = np.nonzero(img)
    for i in range(len(x_nonzero)):
        x = x_nonzero[i]
        y = y_nonzero[i]
        for i in range(len(theta_array)):
            c_rho = np.int((x*cosinuses[i] + y*sinuses[i]) + rho_max)
            if 0 < c_rho < 2 * rho_max:
                ht_map[int(c_rho // rho), i] += 1

    return ht_map, theta_array, rho_array


def get_lines(ht_map, n_lines, thetas, rhos, min_dr, min_dt):
    h, w = ht_map.shape
    c_lines = 0
    lines = []
    theta_rho_res = []
    dy, dx = int(min_dr / (rhos[1]-rhos[0]))+1, int(min_dt / (thetas[1]-thetas[0]))+1

    while c_lines < n_lines:
        ind = np.argmax(ht_map)
        y_max = ind // w
        x_max = ind % w
        ht_map[y_max, x_max] = 0

        y_slice = slice(max(0, y_max-dy), min(y_max+dy, h))
        x_slice = slice(max(0, x_max-dx), min(x_max+dx, h))
        ht_map[y_slice, x_slice] = 0

        # plt.imshow(ht_map, cmap=cm.gray, vmax=125)
        # plt.show()

        c_theta = thetas[x_max]
        c_rho = rhos[y_max]
        for min_t, max_t, min_r, max_r in theta_rho_res:
            if min_t < c_theta < max_t or min_r < c_rho < max_r:
                ind = -1
                break

        if ind == -1:
            continue

        theta_rho_res.append((c_theta-min_dt, c_theta+min_dt, c_rho-min_dr, c_rho+min_dr))

        k = -np.cos(c_theta) / np.sin(c_theta) if abs(np.cos(c_theta)) > 1e-10 else 0
        b = c_rho / np.sin(c_theta)

        lines.append((k, b))
        c_lines += 1
    return lines


if __name__ == '__main__':
    assert len(argv) == 9, 'Wrong input number'
    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0, 'Wrong theta parameter'
    assert rho > 0.0, 'Wrong rho parameter'
    assert n_lines > 0, 'Wrong line number'
    assert min_delta_rho > 0.0, 'Wrong parameter of minimum rho difference'
    assert min_delta_theta > 0.0, 'Wrong parameter of minimum theta difference'

    image = cv2.imread(src_path, 0)
    assert image is not None, 'Could not read an image {file}'.format(file=src_path)

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta)

    visualize_lines(image, lines)
    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write('%0.3f, %0.3f\n' % line)
