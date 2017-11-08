from __future__ import print_function
from sys import argv
import os.path, json
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_line_on_data(data, line):
    x, y = data[:, 0], data[:, 1]

    max_x, min_x = max(x), min(x)
    max_y, min_y = max(y), min(y)
    y_lim = generate_y([min_x, max_x], (min_y, max_y), line)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x, y, s=16, c='red')
    plt_line = Line2D([min_x, max_x], y_lim, color='blue')
    ax.add_line(plt_line)

    plt.show()


def generate_y(x, y_range, line_params):
    a, b, c = line_params

    n = len(x)
    x = np.array(x)

    y_min, y_max = y_range
    if a == 0:
        return (float(-c) / b) * np.ones(n)
    if b == 0:
        return np.linspace(y_min, y_max, n)
    return (x * a + c) / (-b)


def generate_data(img_size, line_params, n_points, sigma, inlier_ratio):
    ys, xs = img_size
    a, b, c = line_params
    x_min = y_min = 0
    y_max, x_max = ys, xs

    if b == 0:
        x_avg = float(-c) / a
        x_min_line, x_max_line = x_avg, x_avg
        x_min = x_avg - xs / 2
        x_max = x_avg + xs / 2
    else:
        y_low, y_high = generate_y([x_min, x_max], [y_min, y_max], line_params)
        if abs(y_high - y_low) > y_max - y_min:
            x_avg = (x_min + x_max) / 2
            x_delta = (x_max - x_min) * (y_max - y_min) / abs(y_high - y_low)
            x_min_line = int(x_avg - x_delta / 2)
            x_max_line = int(x_avg + x_delta / 2)
        else:
            x_min_line = x_min
            x_max_line = x_max

    y_low, y_high = generate_y([x_min_line, x_max_line], (y_min, y_max), line_params)
    y_mean = (y_low + y_high) / 2
    y_min = y_mean - ys / 2
    y_max = y_mean + ys / 2

    n_in = int(n_points * inlier_ratio)
    n_out = n_points - n_in
    x_in = np.linspace(x_min_line, x_max_line, n_in)
    y_in = generate_y(x_in, (y_min, y_max), line_params)
    y_noise = np.random.normal(loc=0, scale=sigma, size=n_in)
    y_in = y_in + y_noise

    y_out, x_out = np.random.sample((2, n_out))
    y_out = y_out * (y_max - y_min) + y_min
    x_out = x_out * (x_max - x_min) + x_min

    x = np.hstack((x_in, x_out)).reshape(n_points, 1)
    y = np.hstack((y_in, y_out)).reshape(n_points, 1)

    return np.hstack((x, y))


def compute_ransac_thresh(alpha, sigma):
    return np.sqrt(chi2.ppf(alpha, 2) * (sigma ** 2)) + 1e-6


def compute_ransac_iter_count(conv_prob, inlier_ratio):
    return int(np.log(1 - conv_prob)/np.log(1 - inlier_ratio**2))+1


def get_model(p_1, p_2):
    x1, y1 = p_1
    x2, y2 = p_2

    a = (y1 - y2)
    b = (x2 - x1)
    c = (x1 * y2 - x2 * y1)
    norm = a ** 2 + b ** 2
    a /= norm
    b /= norm
    c /= norm
    model = a, b, c
    return model


def est_model(model, data, t):
    a, b, c = model
    res = list(map(lambda x: abs(a * x[0] + b * x[1] + c)/np.sqrt(a*a+b*b) <= t, data))
    return res


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
            best_model = model
            prev_out = n_inliers

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
