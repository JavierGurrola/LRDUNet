import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from skimage import io
from skimage.util import view_as_windows
import torchvision.transforms.functional as F


def blend(img1, img2, ratio):
    r"""
    Blends the pair of images given the ratio.
    :param img1: torch tensor
        The reference image.
    :param img2: torch tensor
        The second image in the bend.
    :param ratio: float
        The ratio of the blend.
    :return: torch tensor
        Blended image.
    """
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def data_augmentation(image):
    r"""
    Applies geometric transforms to generate new data samples.
    :param image: numpy ndarray
        The image to transform.
    :return: list
        The list of transformed images.
    """
    to_transform, augmented_images = [image, np.rot90(image)], []

    for t in to_transform:
        t_ud = np.flipud(t)
        t_lr = np.fliplr(t)
        t_udlr = np.fliplr(t_ud)
        augmented_images.extend([t_ud, t_lr, t_udlr])

    augmented_images.extend(to_transform)

    return augmented_images


def create_patches(image, patch_size, step):
    r"""
    Splits the image sample image into patches.
    :param image: numpy ndarray
        The image to split into patches.
    :param patch_size: tuple
        The patch size.
    :param step: tuple
        The step size at which extraction shall be performed.
    :return: numpy ndarray
        The image split image with shape (n_patches, height, width, channels).
    """
    image = view_as_windows(image, patch_size, step)
    h, w = image.shape[:2]
    image = np.reshape(image, (h * w, patch_size[0], patch_size[1], patch_size[2]))

    return image


class AWGN(object):
    r"""
    Additive white gaussian noise generator.

    Args:
        sigma (float): the noise level.
        p (float): the probability to apply Gaussian noise to an image.
    """
    def __init__(self, sigma, p=0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        """
        Applies AWGN to the speckle noisy fringe pattern.
        :param sample: dict
            The clean and noisy image pair training sample.
        :return: dict
            The clean and noisy image pair training sample with the
            transformation applied to the noisy image if it is the case.
        """
        if np.random.uniform() < self.p:
            noisy, image = sample.get('noisy'), sample.get('image')
            sigma = self.sigma * torch.rand(1)
            noisy = noisy + sigma * torch.randn(noisy.size())
            noisy = torch.clamp(noisy, 0., 1.)
            sample = {'image': image, 'noisy': noisy}

        return sample


class Brightness(object):
    r"""
    Random brightness shifting.

    Args:
        min_val (float): the min value for brightness shifting.
        max_val (float): the max value for brightness shifting.
        p (float): the probability to apply brightness shifting.
    """
    def __init__(self, min_val, max_val, p=0.5):
        self.min_val = min_val
        self.max_val = max_val
        self.p = p

    def __call__(self, sample):
        """
        Applies Brightness shifting to the speckle noisy fringe pattern.
        :param sample: dict
            The clean and noisy image pair training sample.
        :return: dict
            The clean and noisy image pair training sample with the
            transformation applied to the noisy image if it is the case.
        """
        if np.random.uniform() < self.p:
            noisy, image = sample.get('noisy'), sample.get('image')
            brightness_factor = np.random.uniform(self.min_val, self.max_val)
            noisy = blend(noisy, torch.zeros_like(noisy), brightness_factor)
            sample = {'image': image, 'noisy': noisy}

        return sample


class Blur(object):
    r"""
    Random Gaussian blur.
    """
    def __init__(self, min_val, max_val, p=0.5):
        r"""
        :param min_val: float
            The min value for blurring sigma.
        :param max_val: float
            The max value for blurring sigma.
        :param p: float
            The probability to apply brightness shifting.
        """
        self.min_val = min_val
        self.max_val = max_val
        self.p = p

    def __call__(self, sample):
        """
        Applies Gaussian blurring to the speckle noisy fringe pattern.
        :param sample: dict
            The clean and noisy image pair training sample.
        :return: dict
            The clean and noisy image pair training sample with the
            transformation applied to the noisy image if it is the case.
        """
        if np.random.uniform() < self.p:
            sigma = np.random.uniform(self.min_val, self.max_val)
            noisy, image = sample.get('noisy'), sample.get('image')
            noisy = F.gaussian_blur(noisy, 3, sigma)
            sample = {'image': image, 'noisy': noisy}

        return sample


class DataSampler(Sampler):
    r"""
    Dataset sampler to train the model in sub-epochs.

    Args:
        data_source (torch Dataset): training dataset.
        num_samples (int): number of samples per epoch (sub-epoch).
    """
    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples
        self.rand = np.random.RandomState(0)
        self.perm = []

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        """
        Generates new iterator with sample index.
        :return: iterator
            Index of the training samples of the current epoch.
        """
        n = len(self.data_source)
        if self._num_samples is not None:
            while len(self.perm) < self._num_samples:
                perm = self.rand.permutation(n).astype('int32').tolist()
                self.perm.extend(perm)
            idx = self.perm[:self._num_samples]
            self.perm = self.perm[self._num_samples:]
        else:
            idx = self.rand.permutation(n).astype('int32').tolist()

        return iter(idx)

    def __len__(self):
        return self.num_samples


class NoisyImagesDataset(Dataset):
    r"""
    Dataset sampler to train the model in sub-epochs.

    Args:
        paths (dict): root paths of the clean and noisy image pairs.
        files (list): file names of the dataset images.
        patch_size (int): size of patches, the patches will be of size patch_size x patch_size.
        transform (torchvision transform): Additional transforms to apply to the dataset during the training.
        augment_data (bool): Apply geometric data augmentation to the dataset.
    """
    def __init__(self, paths, files, patch_size, transform=None, augment_data=False):
        self.patch_size = patch_size
        self.transform = transform
        self.augment_data = augment_data
        self.dataset = {'data': [], 'label': []}
        self.load_dataset(paths, files)

    def __len__(self):
        return len(self.dataset['label'])

    def __getitem__(self, idx):
        image, noisy = self.dataset.get('label')[idx], self.dataset.get('data')[idx]
        sample = {'image': image, 'noisy': noisy}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample.get('noisy'), sample.get('image')

    def load_dataset(self, paths, files):
        r"""
        Loads the image dataset, applies data augmentation if needed, and stores the images as torch tensors.
        :param paths: dict
            Root path of the image dataset.
        :param files: list
            List of image file names.
        :return: None
        """
        patch_size = (self.patch_size, self.patch_size, 1)
        for file in files:
            # Load images in (H, W, C) format.
            img_s = np.expand_dims(io.imread(os.path.join(paths['data'], file)), -1)
            img_c = np.expand_dims(io.imread(os.path.join(paths['label'], file)), -1)

            # Convert to range [0, 1] in float32
            img_s = (img_s / 255.).astype('float32')
            img_c = (img_c / 255.).astype('float32')

            if self.augment_data:
                img_c, img_s = data_augmentation(img_c), data_augmentation(img_s)
            else:
                img_c, img_s = [img_c], [img_s]

            for c, s in zip(img_c, img_s):
                c_patches = create_patches(c, patch_size, step=self.patch_size)
                s_patches = create_patches(s, patch_size, step=self.patch_size)

                for data, label in zip(s_patches, c_patches):
                    data = torch.from_numpy(data.transpose((2, 0, 1)))
                    label = torch.from_numpy(label.transpose((2, 0, 1)))
                    self.dataset['data'].append(data)
                    self.dataset['label'].append(label)
