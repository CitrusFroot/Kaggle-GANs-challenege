import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def generalConv():
    pass

def generalDense():
    pass

def generalRes():
    pass

def vanillaGAN(x, xtarget, poolSize): #for experience
    ''' =========== Legend =========
    filter: an extracted feature/pattern of an image; called a kernel; = to a neuron when trained
    kernel_size: dimensions of the matrix of weights
    strides: how far the filter moves per convolution over an image. The smaller the stride, the more minute details are picked up
    padding: applying a convolution reduces the dimensions of the input. padding avoids this information loss
    residual blocK: improves accuracy by allowing layers to be skipped in a network (reduces rapid weight fluctuation)
    '''
    pass

def cycleGAN(x, xtarget, poolSize): #designed for image-to-image translations. Recommended by Kaggle
    pass

def deepConvGAN(x, xtarget, poolSize): #for experience
    pass


