from __future__ import print_function
from sys import argv
import os.path, json
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt


def plot_line_on_data(data, line):
    x, y = data[:, 0], data[:, 1]
    max_y = max(y)
    max_x = max(x)
    xs = ys = data.shape[0]
    a, b, c = line

    plt.scatter(x, y)

    if b == 0:
        if a != 0:
            xn = (-float(c)/a) * np.ones(xs)
            yn = max_y * np.random.sample(xs)
    else:
        xn = max_x * np.random.sample(xs)
        yn = np.array(map(lambda t: -float(a) / b * t - float(c) / b, xn)).reshape(xs)

    plt.scatter(xn, yn)
    plt.show()


def generate_data(img_size, line_params, n_points, sigma, inlier_ratio):
    ys, xs = img_size
    a, b, c = line_params

    n_in = int(n_points * inlier_ratio)
    n_out = n_points - n_in

    if b == 0:
        if a == 0:
            return -1
        else:
            x_in = (-float(c)/a) * np.ones(n_in)
            y_in = ys * np.random.sample(n_in)
    else:
        x_in = xs * np.random.sample(n_in)
        y_in = np.array(map(lambda x: -float(a)/b * x - float(c) / b + np.random.normal(0, sigma), x_in)).reshape(n_in)

    y_out, x_out = np.random.sample((2, n_out))
    y_out *= ys
    x_out *= xs

    x = np.hstack((x_in, x_out)).reshape(n_points, 1)
    y = np.hstack((y_in, y_out)).reshape(n_points, 1)

    # plt.scatter(x, y)
    # plt.show()

    return np.hstack((x, y))


def compute_ransac_thresh(alpha, sigma):
    # thresh^2 = F_inv(alpha) * sigma^2, where F is the cdf for chi-squared distribution
    return np.sqrt(chi2.ppf(alpha, 1) * (sigma ** 2))


def compute_ransac_iter_count(conv_prob, inlier_ratio):
    # should think more about it
    return int(np.log(1 - conv_prob)/np.log(1 - inlier_ratio**2))+1


def get_model(p_1, p_2):
    a = b = c = None
    x1, y1 = p_1
    x2, y2 = p_2
    if y2 == y1:
        a, b, c = 0, 1, -y1
    elif x2 == x1:
        a, b, c = 1, 0, -x1
    else:
        k = float(y2 - y1) / (x2 - x1)
        b = y2 - k * x2
        a, b, c = -k, 1, -b
    model = a, b, c
    return model


def est_model(model, data, t):
    a, b, c = model
    res = list(map(lambda x: abs(a * x[0] + b * x[1] + c)/np.sqrt(a*a+b*b) <= t, data))
    return res

def fit_model(data):
    # data * z = b, where b = np.ones((data_size,1))

    A_inv = np.matmul(data.T, data)
    z = np.sum(np.matmul(np.linalg.inv(A_inv), data.T), 1)
    a, b = z
    model = a, b, 1
    return model


def compute_line_ransac(data, t, n):
    sz = data.shape[0]
    prev_out = 0
    best_model = None

    for it in range(n):
        indices = sz * np.random.random(2)
        indices = indices.astype(int)
        pt1, pt2 = data[indices]
        model = get_model(pt1, pt2)
        is_inlier = est_model(model, data, t)
        n_inliers = sum(is_inlier)

        if n_inliers > prev_out:
            data_fit = data[is_inlier, :]
            fit_size = data_fit.shape[0]
            indices = fit_size * np.random.random(2)
            indices = indices.astype(int)
            cpt1, cpt2 = data_fit[indices]
            best_model = get_model(cpt1, cpt2)
            new_model = fit_model(data_fit)
            best_model = new_model
            prev_out = n_inliers

    # print(prev_out)
    return best_model


def main():
    print(argv)
    assert len(argv) == 2
    assert os.path.exists(argv[1])

    with open(argv[1]) as fin:
        params = json.load(fin)

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used

    sigma - Gaussian noise
    alpha - probability of point is an inlier

    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    data = generate_data((params['w'], params['h']),
                         (params['a'], params['b'], params['c']),
                         params['n_points'], params['sigma'],
                         params['inlier_ratio'])

    t = compute_ransac_thresh(params['alpha'], params['sigma'])
    n = compute_ransac_iter_count(params['conv_prob'], params['inlier_ratio'])

    detected_line = compute_line_ransac(data, t, n)
    print(detected_line)
    plot_line_on_data(data, detected_line)


if __name__ == '__main__':
    main()
