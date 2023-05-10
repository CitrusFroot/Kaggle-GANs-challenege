import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D

def generalConv(poolSize, hyperparameterData):
    pass
    

def generalDense():
    pass

def generalRes():
    pass

def vanillaGAN(x, xtarget, poolSize, hyperparameterData): #for experience
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

def makeGenerator(imgDim, poolSize, hyperparameterData):
    filters    = hyperparameterData[0]
    kernelDim1 = hyperparameterData[1]
    kernelDim2 = hyperparameterData[2]
    kernelDim3 = hyperparameterData[3]
    padding    = hyperparameterData[4]
    stride1    = hyperparameterData[5]
    stride2    = hyperparameterData[6]
    stride3    = hyperparameterData[7]
    alpha      = hyperparameterData[8]
    epsilon    = hyperparameterData[9]

    weightInit = keras.initializers.RandomNormal(stddev = 0.02) #TODO: experiment with this

    input = keras.Input(shape = imgDim) #sets the input type
    pass

def makeDiscriminator(imgDim, hyperparameterData):
    filters    = hyperparameterData[0]
    kernelDim1 = hyperparameterData[1]
    kernelDim2 = hyperparameterData[2]
    kernelDim3 = hyperparameterData[3]
    padding    = hyperparameterData[4]
    stride1    = hyperparameterData[5]
    stride2    = hyperparameterData[6]
    stride3    = hyperparameterData[7]
    alpha      = hyperparameterData[8]
    epsilon    = hyperparameterData[9]

    weightInit = keras.initializers.RandomNormal(stddev = 0.02) #TODO: experiment with this

    input = keras.Input(shape = imgDim) #sets the input type

    layer = Conv2D(filters = filters, kernel_size = kernelDim1, strides = stride3, padding = padding, kernel_initializer = weightInit)(input)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer) #activation layer TODO experiment with other activation functions

    #filters = filters*2, as this seems to be the recommended flow of CycleGAN. Experiment/Research required
    layer = Conv2D(filters = filters*2, kernel_size = kernelDim2, strides = stride3, padding = padding, kernel_initializer = weightInit)(layer)
    #Try a normalization layer here. Recommended: BatchNormalization, InstanceNormalization
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)

    #filters = filters*4, as this seems to be the recommended flow of CycleGAN. Experiment/Research required
    layer = Conv2D(filters = filters*4, kernel_size = kernelDim2, strides = stride3, padding = padding, kernel_initializer = weightInit)(layer)
    #Try a normalization layer here. Recommended: BatchNormalization, InstanceNormalization
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)

    #filters = filters*8, as this seems to be the recommended flow of CycleGAN. Experiment/Research required
    layer = Conv2D(filters = filters*8, kernel_size = kernelDim2, strides = stride1, padding = padding, kernel_initializer = weightInit)(layer)
    #Try a normalization layer here. Recommended: BatchNormalization, InstanceNormalization
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)

    #Output layer
    layer = Conv2D(filters = 1, kernel_size = kernelDim2, strides = stride1, padding = padding, kernel_initializer = weightInit)(layer)

    model = keras.Model(input, layer) #our model
    #TODO: experiment with different optimizers
    model.compile(optimizer = 'Adam', loss = 'mse')
    print(model.summary())
    return model

def doCycle(domainA, generatorA, domainB, generatorB):
    pass

def deepConvGAN(x, xtarget, poolSize): #for experience
    pass


