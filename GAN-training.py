import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def trainGenerator(x, xtarget):
    ''' =========== Legend =========
    filter: an extracted feature/pattern of an image; called a kernel; = to a neuron when trained
    kernel_size: dimensions of the matrix of weights
    strides: how far the filter moves per convolution over an image. The smaller the stride, the more minute details are picked up
    padding: applying a convolution reduces the dimensions of the input. padding avoids this information loss
    residual blocK: improves accuracy by allowing layers to be skipped in a network (reduces rapid weight fluctuation)
    '''
    #constructs neural network
    generator = tf.keras.Sequential()
    generator.add(keras.layers.Reshape(target_shape = (256,256,3), input_shape = (256,256,3))) #input layer
    #first three convolutional layers
    generator.add(keras.layers.Conv3D(filters = 64, kernel_size = (7,7),  strides = 1, padding = 3, activation = 'relu', use_bias = True, name = 'conv1'))
    generator.add(keras.layers.Conv3D(filters = 128, kernel_size = (3,3), strides = 3, padding = 1, activation = 'relu', use_bias = True, name = 'conv2'))
    generator.add(keras.layers.Conv3D(filters = 256, kernel_size = (3,3), strides = 2, padding = 1, activation = 'relu', use_bias = True, name = 'conv3'))
    #adds 9 residual blocks
    
    pass
