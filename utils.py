import random
import torch
import numpy as np
from collections import OrderedDict


def correct_model_dict(state_dict):
    r"""
    Corrects the names of the torch nn.Modules of the model
    in the case that it was trained using multiple GPUs.
    :param state_dict: dict
        A dict containing parameters and persistent buffers.
    :return: dict
        Corrected state dict.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def mod_pad(img, mod, mode='constant'):
    r"""
    Pads the input image to a size compatible with the number
    of down-samplings and up-samplings of the model.
    :param img: numpy ndarray
        The image to pad.
    :param mod: int
        Module to calculate padding, it is based in the number of down-samplings and up-samplings.
    :param mode:
        Padding mode.
    :return: tuple
        Padded image and original image size.
    """
    size = img.shape[:2]
    h, w = np.mod(size, mod)
    h, w = mod - h, mod - w
    if h != mod or w != mod:
        img = np.pad(img, ((0, h), (0, w), (0, 0)), mode=mode, constant_values=0)

    return img, size


def set_seed(seed=1):
    r"""
    Set the random state for reproducibility.
    :param seed: int
        The seed for the random numbers generators.
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_ensemble(image, normalize=True):
    r"""
    Create image ensemble to estimate denoised image.
    :param image: numpy ndarray
        The noisy image.
    :param normalize: bool
        If normalize=Ture, the image is normalized to the range [0., 1.].
    :return: list
        The ensemble of noisy image (noisy image with geometric transforms).
    """
    ensemble_np = [image, np.fliplr(image), np.flipud(image), np.flipud(np.fliplr(image))]
    img_rot = np.rot90(image)
    img_rot = [img_rot, np.fliplr(img_rot), np.flipud(img_rot), np.flipud(np.fliplr(img_rot))]
    ensemble_np.extend(img_rot)

    ensemble_t = []
    for img in ensemble_np:
        if img.ndim == 2:                                           # Expand dim for channel dimension in gray scale.
            img = np.expand_dims(img.copy(), 0)                     # Use copy to avoid problems with reverse indexing.
        else:
            img = np.transpose(img.copy(), (2, 0, 1))               # Channels-first transposition.
        if normalize:
            img = img / 255.

        img_t = torch.from_numpy(np.expand_dims(img, 0)).float()    # Expand dims again to create batch dimension.
        ensemble_t.append(img_t)

    return ensemble_t


def separate_ensemble(ensemble, return_single=False):
    r"""
    Apply inverse transforms to predicted image ensemble and average them.
    :param ensemble: list
        The predicted images, ensemble[0] is the original image,
        and ensemble[i] is a transformed version of ensemble[0].
    :param return_single: bool
        If return_single=True, the functions also returns ensemble[0]
        to evaluate single prediction.
    :return: numpy array or tuple (numpy ndarray, numpy ndarray)
        The average of the predicted images, and the original image
        denoised if return_single=True.
    """
    ensemble_np = []

    for img in ensemble:
        img = img.squeeze()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        ensemble_np.append(img)

    # Vertical and Horizontal Flips
    img = ensemble_np[0] + np.fliplr(ensemble_np[1]) + np.flipud(ensemble_np[2]) + np.fliplr(np.flipud(ensemble_np[3]))

    # 90º Rotation, Vertical and Horizontal Flips
    img = img + np.rot90(ensemble_np[4], k=3) + np.rot90(np.fliplr(ensemble_np[5]), k=3)
    img = img + np.rot90(np.flipud(ensemble_np[6]), k=3) + np.rot90(np.fliplr(np.flipud(ensemble_np[7])), k=3)
    img = np.clip(img / 8, 0., 1.)

    if return_single:
        return img, np.clip(ensemble_np[0], 0., 1.)
    else:
        return img


def predict_ensemble(model, ensemble, device):
    r"""
    Auxiliary function to predict the ensemble of images.
    :param model: torch Module
        The denoising model.
    :param ensemble: list
        The ensemble image list.
    :param device: torch device
        Device of the model and the image.
    :return: list
        List of estimated images.
    """
    y_hat_ensemble = []
    for x in ensemble:
        x = x.to(device)
        with torch.no_grad():
            y_hat = model(x)
            y_hat_ensemble.append(y_hat.cpu().detach().numpy().astype('float32'))
    return y_hat_ensemble
