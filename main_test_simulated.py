import yaml
import torch
import numpy as np
import os

from os.path import join
from model import LightRDUNet
from skimage import io
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from sewar.full_ref import uqi

from utils import correct_model_dict, mod_pad, build_ensemble, separate_ensemble, predict_ensemble


def mean_absolute_error(image_true, image_test):
    r"""
    Mean absolute error between two images.
    :param image_true: numpy ndarray
        The reference image.
    :param image_test: numpy ndarray
        The predicted image.
    :return: float
        Mean absolute error value.
    """
    return np.mean(np.abs(image_true - image_test))


def get_image_metrics(image_true, image_test):
    r"""
    Evaluate the predicted image.
    :param image_true: numpy ndarray
        The reference image.
    :param image_test: numpy ndarray
        The estimated image.
    :return: list
        The results of PSNR, MAE, SSIM, and Q evaluations.
    """
    psnr = peak_signal_noise_ratio(image_true, image_test, data_range=1.)
    ssim = structural_similarity(image_true, image_test, data_range=1., multichannel=False,
                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    mae = mean_absolute_error(image_true, image_test)
    q_index = uqi(image_true, image_test)

    return psnr, ssim, mae, q_index


def save_image(image, path):
    r"""
    Save estimated image.
    :param image: numpy ndarray
        The estimated image to save.
    :param path: str
        The path of the estimated image, it includes the evaluation metrics.
    :return: None
    """
    image = (255 * image).astype('uint8').squeeze()
    io.imsave(path, image)


def predict(model, noisy_dataset, label_dataset, device, results_path, save_images=False):
    r"""
    Estimate the images in the dataset.
    :param model: torch Module
        The denoising model.
    :param noisy_dataset: list
        A list with the noisy images.
    :param label_dataset: list
        A list with the clean images (ground truth).
    :param device: torch device
        The device of the model and dataset.
    :param results_path: str
        The path to the directory where the images will be saved.
    :param save_images: bool
        If save_images=True, the images are saved in results_path directory.
    :return: tuple
        The average of the evaluation metrics.
    """
    X, Y = np.array(noisy_dataset), np.array(label_dataset)
    n_images, height, width, channels = X.shape

    Y_pred, Y_pred_ens = np.empty_like(Y), np.empty_like(Y)
    psnr_list, ssim_list, q_index_list, mae_list = [], [], [], []
    ens_psnr_list, ens_ssim_list, ens_q_index_list, ens_mae_list = [], [], [], []

    for i in range(n_images):
        x, y = X[i, ...], Y[i, ...]
        x, size = mod_pad(x, 8)
        x = build_ensemble(x, normalize=False)

        with torch.no_grad():
            y_hat_ens = predict_ensemble(model, x, device)
            y_hat_ens, y_hat = separate_ensemble(y_hat_ens, return_single=True)

            y_hat = y_hat[:size[0], :size[1], ...]
            y_hat_ens = y_hat_ens[:size[0], :size[1], ...]

            Y_pred[i, ...] = np.expand_dims(y_hat, -1)
            Y_pred_ens[i, ...] = np.expand_dims(y_hat_ens, -1)

            y, y_hat, y_hat_ens = np.squeeze(y), np.squeeze(y_hat), np.squeeze(y_hat_ens)

            psnr, ssim, mae, q_index = get_image_metrics(y, y_hat)
            psnr_ens, ssim_ens, mae_ens, q_index_ens = get_image_metrics(y, y_hat_ens)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            q_index_list.append(q_index)
            mae_list.append(mae)

            ens_psnr_list.append(psnr_ens)
            ens_ssim_list.append(ssim_ens)
            ens_q_index_list.append(q_index_ens)
            ens_mae_list.append(mae_ens)

            message = 'Image:{} - PSNR:{:.4f} - MAE:{:.4f} - SSIM:{:.4f} - Q:{:.4f} - ' \
                      'ens PSNR:{:.4f} - ens MAE:{:.4f} - ens SSIM:{:.4f} - ens Q:{:.4f}'
            print(message.format(i + 1, psnr, mae, ssim, q_index, psnr_ens, mae_ens, ssim_ens, q_index_ens))

    if save_images:
        os.makedirs(results_path, exist_ok=True)
        base_name = '{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'
        for i in range(n_images):
            name = base_name + '.png'
            name = name.format(i + 1, psnr_list[i], mae_list[i], ssim_list[i], q_index_list[i])
            save_image(Y_pred[i, ...], os.path.join(results_path, name))

            name = base_name + '_ens.png'
            name = name.format(i + 1, ens_psnr_list[i], ens_mae_list[i], ens_ssim_list[i], ens_q_index_list[i])
            save_image(Y_pred_ens[i, ...], os.path.join(results_path, name))

    return np.mean(psnr_list), np.mean(mae_list), np.mean(ssim_list), np.mean(q_index_list), \
           np.mean(ens_psnr_list), np.mean(ens_mae_list), np.mean(ens_ssim_list), np.mean(ens_q_index_list)


if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    test_params = config['test']
    model_params = config['model']
    model_path = join(config['test']['pretrained models path'], 'model_speckle.pth')
    model = LightRDUNet(**model_params)

    device = torch.device(config['test']['device'])
    print("Using device: {}".format(device))

    state_dict = torch.load(model_path, map_location=device)
    state_dict = correct_model_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    noisy_images, label_images = [], []

    with open('test_files.txt', 'r') as f:
        test_files = f.read().splitlines()

    for file in test_files:
        noisy = io.imread(join(test_params['dataset path'], 'data', file)) / 255.
        label = io.imread(join(test_params['dataset path'], 'label', file)) / 255.
        noisy_images.append(np.expand_dims(noisy, -1).astype('float32'))
        label_images.append(np.expand_dims(label, -1).astype('float32'))

    psnr, mae, ssim, q_index, psnr_ens, mae_ens, ssim_ens, q_index_ens = predict(model, noisy_images, label_images,
                                                                                 device, test_params['results path'],
                                                                                 test_params['save images'])
    message = 'PSNR:{:.4f} - MAE:{:.4f} - SSIM:{:.4f} - Q:{:.4f} - ' \
              'ens PSNR:{:.4f} - ens MAE:{:.4f} - ens SSIM:{:.4f} - ens Q:{:.4f}'

    print(message.format(np.around(psnr, decimals=4), np.around(mae, decimals=4), np.around(ssim, decimals=4),
                         np.around(q_index, decimals=4), np.around(psnr_ens, decimals=4), np.around(mae_ens, decimals=4),
                         np.around(ssim_ens, decimals=4), np.around(q_index_ens, decimals=4)))
