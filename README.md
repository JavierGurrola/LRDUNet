# LRDUNet
U-Net based neural network for fringe pattern denoising

This repository is for the LRDUNet model proposed in the following paper:

[Javier Gurrola-Ramos](https://scholar.google.com.mx/citations?user=NuhdwkgAAAAJ&hl=es), [Oscar Dalmau](https://scholar.google.com.mx/citations?user=5oUOG4cAAAAJ&hl=es&oi=sra) and [Teresa E. Alarcón](https://scholar.google.com.mx/citations?user=gSUClZYAAAAJ&hl=es&authuser=1), ["U-Net based neural network for fringe pattern denoising"](https://www.sciencedirect.com/science/article/abs/pii/S0143816621002992), Optics and Lasers in Engineering, vol 149, pp. 106829, 2022, doi: [10.1016/j.optlaseng.2021.106829](https://doi.org/10.1016/j.optlaseng.2021.106829).

## Citation

If you use this paper work in your research or work, please cite our paper:
```
@article{gurrola2022u,
  title = {U-Net based neural network for fringe pattern denoising},
  journal = {Optics and Lasers in Engineering},
  volume = {149},
  pages = {106829},
  year = {2022},
  issn = {0143-8166},
  doi = {https://doi.org/10.1016/j.optlaseng.2021.106829},
  url = {https://www.sciencedirect.com/science/article/pii/S0143816621002992},
  author = {Javier Gurrola-Ramos and Oscar Dalmau and Teresa Alarcón},
}
```

![LRDUNet](https://github.com/JavierGurrola/LRDUNet/blob/main/Figs/Model.png)

## Pre-trained model and datasets

[Link](https://drive.google.com/drive/folders/1YsE8RLdrcSKGeGUGiEOae9wyY1ryfr1F?usp=sharing) to download the pretrained model, and the training and test datasets.

## Dependencies
- Python 3.6
- Numpy 1.19.2
- PyTorch 1.8.0
- torchvision 0.9.0
- pytorch-msssim 0.2.1
- ptflops 0.6.4
- tqdm 4.49.0
- scikit-image 0.17.2
- sewar 0.4.4
- yaml 0.2.5

## Training
Default parameters used in the paper are set in the ```config.yaml``` file:

```
groups: 8
dense covolutions: 2
downsampling: 'Conv2'
residual: true
patch size: 64
batch size: 16
learning rate: 1.e-3
weight decay: 1.e-2
scheduler gamma: 0.5
scheduler step size: 5
epochs: 20
```

Additionally, you can choose the device, the number of workers of the data loader, and enable multiple GPU use.

To train the model use the following command:

```python main_train.py```

## Test

Place the pretrained model in the './Pretrained' folder. To test the model in the simulated fringe patterns, use the following command:

```python main_test_simulated.py```

To test the model in the experimetal fringe patterns, use the following command:

```python main_test_experimental.py```

## Results

![Results](https://github.com/JavierGurrola/LRDUNet/blob/main/Figs/SomeResults.png)

## Contact

If you have any question about the code or paper, please contact francisco.gurrola@cimat.mx .
