import os
import numpy as np
from skimage import io


def eval_size(n):
    r"""
    Check if the size is an integer, a list, or a tuple. If n is an integer, then the image size is n x n.
    If n is a list or a tuple, then the image size size is n[0] x n[1].
    :param n: int, list or tuple
        The image size.
    :return: tuple
        The image size in format height x width.
    """
    if isinstance(n, int):
        rows = cols = n
    elif isinstance(n, (list, tuple)) and len(n) == 2:
        rows, cols = n[:2]
    else:
        raise TypeError("Invalid data type for n, it must be int, list or tuple.")

    return rows, cols


def gaussian_kernel(x, y, sigma_1, sigma_2):
    r"""
    2D Gaussian kernel for phase generation (Eq. 13 in the paper).
    :param x: numpy ndarray
        Coordinate matrix of x-axis.
    :param y: numpy ndarray
        Coordinate matrix of y-axis.
    :param sigma_1: float
        \sigma_x component of the covariance matrix.
    :param sigma_2: float
        \sigma_y component of the covariance matrix.
    :return: numpy ndarray
        Evaluation of the gaussian kernel.
    """
    k_1 = 1. / (2. * sigma_1 ** 2)
    k_2 = 1. / (2. * sigma_2 ** 2)
    z = np.exp(- k_1 * x ** 2 - k_2 * y ** 2)

    return z


def generate_speckle(phase):
    r"""
    Generates the clean and noisy fringe pattern images given the phase of the image.
    :param phase: numpy ndarray
        Phase of the fringe pattern image (Eq. 12 in the paper).
    :return: tuple (numpy ndarray, numpy ndarray)
        Clean and noisy image pair.
    """
    rows, cols = phase.shape[:2]

    # Generate noisy image
    phi_0 = np.random.uniform(-np.pi, np.pi, (rows, cols))

    a_r_squared = np.ones_like(phase)
    a_o_squared = np.random.uniform(0.1, 0.25) + np.random.exponential(0.2, size=phase.shape)
    a_o_squared = np.clip(a_o_squared, 0., 1.)

    aa = 4. * a_r_squared * a_o_squared

    noisy_image = aa + aa * np.cos(phase + np.pi)
    noise = - aa * (1. - np.cos(phase)) * np.cos(2. * phi_0 + phase)
    noisy_image = noisy_image + noise

    # Clean image.
    clean_image = 1. + 1. * np.cos(phase + np.pi)

    # Normalize images to range [0., 1.]
    clean_image = (clean_image - clean_image.min()) / (clean_image.max() - clean_image.min())
    noisy_image = (noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min())

    return clean_image, noisy_image


def random_peaks(n, peaks, probabilities, weights_max, k=5.):
    r"""
    Generates the phase of the fringe pattern image (Eq. 12 the paper).
    :param n: int, list, tuple
        The size if the image. If n is an integer, then the image size is n x n. If n is
        a tuple or a list, then the image size is n[0] x n[1].
    :param peaks: list
        A list with number of basis functions in the phase.
    :param probabilities: numpy ndarray
        The probabilities associated with each entry in peaks.
    :param weights_max: float
        The clip value for the weights associated to the basis functions.
    :param k: float
        The scale factor to control fringe density.
    :return: numpy ndarray
        The phase of the fringe pattern.
    """
    rows, cols = eval_size(n)
    x = np.linspace(-3, 3, cols, endpoint=True)
    y = np.linspace(-3, 3, rows, endpoint=True)
    xx, yy = np.meshgrid(x, y)

    n_peaks = int(np.random.choice(peaks, 1, True, probabilities))
    centers = np.random.randn(2, n_peaks)
    s = 2.5 * np.random.rand(2, n_peaks) + 0.5
    w = np.random.uniform(-weights_max, weights_max, n_peaks)

    positive = w > 0
    negative = w < 0
    w[positive] = np.clip(w[positive], 1., max(w.max(), 1.))
    w[negative] = np.clip(w[negative], min(-1., w.min()), -1)

    z = np.zeros((rows, cols))

    for i in np.arange(n_peaks):
        z = z + w[i] * gaussian_kernel(xx + centers[0, i], yy + centers[1, i], s[0, i], s[1, i])

    z = np.random.uniform(1., k) * z

    return z


if __name__ == '__main__':
    np.random.seed(1)
    # Change dataset_path to your desired destination folder
    dataset_path = './Dataset'
    samples = 1700
    n_peaks = [i for i in range(1, 9)]
    weights = 15.
    kappa = 5.
    probabilities = np.array([0.1, 2, 3, 3, 2, 2, 1, 0.25])
    probabilities = probabilities / probabilities.sum()

    size = (256, 256)
    data_path = os.path.join(dataset_path, 'data')
    label_path = os.path.join(dataset_path, 'label')

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    for i in range(1, samples + 1):
        img_name = '{:04d}.png'.format(i)
        phase = random_peaks(size, n_peaks, probabilities, weights, kappa)
        clean_image, noisy_image = generate_speckle(phase)

        clean_image = np.around(255. * clean_image).astype('uint8')
        noisy_image = np.around(255. * noisy_image).astype('uint8')

        io.imsave(os.path.join(data_path, img_name), noisy_image)
        io.imsave(os.path.join(label_path, img_name), clean_image)
