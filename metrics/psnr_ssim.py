import cv2
import numpy as np

import skimage.metrics
import torch
import torchmetrics
from torch.nn import functional as F
from torchmetrics.regression import MeanSquaredError

def calculate_psnr(img1,
                   img2):
    psnr = torchmetrics.PeakSignalNoiseRatio(data_range=2)
    return psnr(img1,img2)

def calculate_ssim(img1,
                   img2):
    ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=2)
    return ssim(img1,img2)

def calculate_rmse(img1,
                   img2):
    mse = MeanSquaredError()
    return torch.sqrt(mse(img1,img2))

