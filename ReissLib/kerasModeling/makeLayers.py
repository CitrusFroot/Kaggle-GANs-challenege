import tensorflow as tf
from keras.layers import Conv2D, Dense, BatchNormalization, LeakyReLU, ReLU

#Note to self: start using python naming conventions (gross)

def general_conv2D_layer(filters:int, kernel_size:tuple|list, stride:tuple|list, padding:str, kernel_initializer, input):
     #first hidden conv layer
    layer = Conv2D(filters = filters, 
                   kernel_size = kernel_size, 
                   strides = stride, 
                   padding = padding, 
                   kernel_initializer = kernel_initializer)(input)
    
    layer = BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = LeakyReLU(alpha = alpha)(layer) #activation layer TODO experiment with other activation functions

def general_dense_layer():
    pass

def general_res_block():
    pass