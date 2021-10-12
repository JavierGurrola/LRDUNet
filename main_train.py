import os
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
from torchvision import transforms

from model import LightRDUNet
from train import fit_model
from data_management import NoisyImagesDataset, DataSampler, Brightness, AWGN, Blur
from utils import set_seed

if __name__ == '__main__':
    set_seed(1)
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    # Training, validation and model parameters.
    train_params = config['train']
    val_params = config['val']
    model_params = config['model']

    # Defining LRDUNet model and print model summary.
    device = torch.device(train_params['device'])
    model = LightRDUNet(**model_params).to(device)
    param_group = []
    for name, param in model.named_parameters():
        p = {'params': param, 'weight_decay': train_params['weight decay'] if 'conv' in name else 0.}
        param_group.append(p)

    # Print model summary
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, (1, train_params['patch size'], train_params['patch size']))
        print('Model summary:')
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print("Using device: {}".format(device))
    if torch.cuda.device_count() > 1 and 'cuda' in device.type and train_params['multi gpu']:
        model = nn.DataParallel(model)
        print('Using multiple GPUs')

    # Load training and validation datasets file names.
    with open('train_files.txt', 'r') as f_train, open('val_files.txt', 'r') as f_val:
        train_files = f_train.read().splitlines()
        val_files = f_val.read().splitlines()
    paths = {
        'data': os.path.join(train_params['dataset path'], 'data'),
        'label': os.path.join(train_params['dataset path'], 'label')
    }

    # Data augmentation
    brightness_level = list(train_params['brightness level'])
    blur_level = list(train_params['blur level'])
    training_transform = transforms.Compose([
        Brightness(brightness_level[0], brightness_level[1], p=0.5),
        Blur(blur_level[0], blur_level[1], p=0.5),
        AWGN(train_params['noise level'], p=0.5)
    ])

    # Load training and validations datasets.
    training_dataset = NoisyImagesDataset(paths, train_files, train_params['patch size'], training_transform, True)
    validation_dataset = NoisyImagesDataset(paths, val_files, val_params['patch size'], None, False)
    samples_per_epoch = len(training_dataset) // train_params['dataset splits']

    # Training in sub-epochs:
    epochs = train_params['epochs'] * train_params['dataset splits']
    print('Training samples:', len(training_dataset), '\nValidation samples:', len(validation_dataset))
    sampler = DataSampler(training_dataset, num_samples=samples_per_epoch)

    data_loaders = {
        'train': DataLoader(training_dataset, batch_size=train_params['batch size'],
                            num_workers=train_params['workers'], sampler=sampler),
        'val': DataLoader(validation_dataset, batch_size=val_params['batch size'], num_workers=train_params['workers']),
    }

    # Optimization: loss function, optimizer, and learning rate scheduler.
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(param_group, lr=train_params['learning rate'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=train_params['scheduler step size'] * train_params['dataset splits'],
                                             gamma=train_params['scheduler gamma'])

    # Fit the model
    fit_model(model, data_loaders, criterion, optimizer, lr_scheduler, device, epochs, val_params['frequency'],
              train_params['checkpoint path'], 'model_speckle', train_params['verbose'])
