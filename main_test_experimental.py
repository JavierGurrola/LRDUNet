import os
import yaml
import torch
import numpy as np
from skimage import io

from model import LightRDUNet
from utils import correct_model_dict, mod_pad, build_ensemble, separate_ensemble, predict_ensemble


def predict(model, noisy_dataset, device, results_path, save_images=False):
    r"""
    Estimate the images in the dataset.
    :param model: torch Module
        The denoising model.
    :param noisy_dataset: list
        A list with the noisy images.
    :param device: torch device
        The device of the model and dataset.
    :param results_path: str
        The path to the directory where the images will be saved.
    :param save_images: bool
        If save_images=True, the images are saved in results_path directory.
    :return: None
    """
    X = noisy_dataset
    n_images = len(X)
    Y_pred, Y_pred_ens = [], []

    for i in range(n_images):
        x = X[i]
        x, size = mod_pad(x, 8)
        x = build_ensemble(x, normalize=False)

        with torch.no_grad():
            y_hat_ens = predict_ensemble(model, x, device)
            y_hat_ens, y_hat = separate_ensemble(y_hat_ens, return_single=True)

            y_hat = y_hat[:size[0], :size[1], ...]
            y_hat_ens = y_hat_ens[:size[0], :size[1], ...]

            y_hat = (y_hat - y_hat.min()) / (y_hat.max() - y_hat.min())
            y_hat_ens = (y_hat_ens - y_hat_ens.min()) / (y_hat_ens.max() - y_hat_ens.min())

            Y_pred.append(y_hat)
            Y_pred_ens.append(y_hat_ens)
            print('Image: {} done!'.format(i + 1))

    if save_images:
        for i in range(n_images):
            y_hat = (255 * Y_pred[i]).astype('uint8')
            y_hat_ens = (255 * Y_pred_ens[i]).astype('uint8')

            y_hat = np.squeeze(y_hat)
            y_hat_ens = np.squeeze(y_hat_ens)

            os.makedirs(results_path, exist_ok=True)
            io.imsave(os.path.join(results_path, '{}.png'.format(i + 1)), y_hat)
            io.imsave(os.path.join(results_path, '{}_ens.png'.format(i + 1)), y_hat_ens)


if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    model_params = config['model']
    test_params = config['test']
    model_path = os.path.join(test_params['pretrained models path'], 'model_speckle.pth')
    model = LightRDUNet(**model_params)

    device = torch.device(config['test']['device'])
    print("Using device: {}".format(device))

    state_dict = torch.load(model_path, map_location=device)
    state_dict = correct_model_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    test_files = sorted(os.listdir(test_params['experimental dataset path']))
    noisy_dataset = []

    for file in test_files:
        noisy = io.imread(os.path.join(test_params['experimental dataset path'], file))
        noisy = np.expand_dims(noisy / 255., -1).astype('float32')
        noisy_dataset.append(noisy)

    predict(model, noisy_dataset, device, test_params['experimental results path'], test_params['save images'])
