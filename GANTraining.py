from tensorflow import keras
from keras.layers import Conv2D
import logging
import random
from ReissLib.debugMessages import createLogger as cl

logger = cl(level = 'info', dir = 'Log files') #creates the debug messager

''' =========== Legend =========
filter: an extracted feature/pattern of an image; called a kernel; = to a neuron when trained
kernel_size: dimensions of the matrix of weights
strides: how far the filter moves per convolution over an image. The smaller the stride, the more minute details are picked up
padding: applying a convolution reduces the dimensions of the input. padding avoids this information loss
residual blocK: improves accuracy by allowing layers to be skipped in a network (reduces rapid weight fluctuation)
'''

'''
creates a generator CNN
imgDim: a tuple/list of size 3
hyperparameterData: a tuple/list of hyperparameters to be experimented with
numResBlocks: an int; determines how many residual blocks are used
returns: a tf.keras.model instance. UNTRAINED
'''
def makeGenerator(imgDim:tuple|list, hyperparameterData:tuple|list, numResBlocks:int):
    logger.info(f'makeGenerator called.\timgDim: {imgDim}')
    filters    = hyperparameterData[0]
    kernelDim1 = hyperparameterData[1] #first kernel
    kernelDim2 = hyperparameterData[2] #kernel used for all other layers
    padding    = hyperparameterData[4]
    stride1    = hyperparameterData[5] #stride for input and output layers
    stride2    = hyperparameterData[6] #stride for all other layers
    alpha      = hyperparameterData[8] #hyperparameter that decreases gradient in LeakyReLU if x < 0
    epsilon    = hyperparameterData[9] #learning rate

    weightInit = keras.initializers.RandomNormal(stddev = 0.02, seed = getRandomSeed()) #TODO: experiment with this
    input = keras.Input(shape = imgDim) #sets the input type
    
    #debug warning. Input MUST be 256x256 in RGB
    if(imgDim != (256,256,3)):
        logger.warn(f'input layer is not of the correct target size. current size is {imgDim}, should be (256, 256, 3)')

    #first hidden convolutional layer. Activation size should be (256,256,64)
    layer = Conv2D(filters = filters, kernel_size = kernelDim1, strides = stride1, padding = padding, kernel_initializer = weightInit)(input)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)
    logger.info(f'hConvLayer0 created. Value: {layer}\tTarget activation size: (256, 256, 64)')

    #second hidden conv layer. Activation size should be (128,128,128)
    layer = Conv2D(filters = filters*2, kernel_size = kernelDim2, strides = stride2, padding = padding, kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)
    logger.info(f'hConvLayer1 created. Value: {layer}\tTarget activation size: (128,128,128)')

    #third hidden conv layer. Activation size should be (64,64,256)
    layer = Conv2D(filters = filters*4, kernel_size = kernelDim2, strides = stride2, padding = padding, kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)
    logger.info(f'hConvLayer2 created. Value: {layer}\tTarget activation size: (64, 64, 256)')

    #adds x amount of residual layers. 9 for 256x256 TODO: verify math and find evidence to back up. !!!STRIDE MUST BE (1,1) given everything else is standard
    for _ in range(numResBlocks):
        layer = getResNetBlock(imgDim[0], kernelDim2, stride1, padding, epsilon, layer) #when finished, layer's activation size should be (64,64,256)
    
    #we transpose to upscale the image. Every conv2D layer decreases the dimension of the image. These two layers restore the original dimensions
    layer = keras.layers.Conv2DTranspose(filters = filters*2, kernel_size = kernelDim2, strides = stride2, padding = padding, output_padding = (1,1), kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)

    layer = keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernelDim2, strides = stride2, padding = padding, output_padding = (1,1), kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)

    #final output layer
    layer = Conv2D(filters = imgDim[2], kernel_size = kernelDim2, strides = stride1, padding = padding, kernel_initializer = weightInit)(layer)

    #compiles the untrained model and returns it to driver.py
    model = keras.Model(input, layer)
    model.compile(optimizer = 'adam', loss = 'mse')
    logger.info(f'Generator Summary: {str(model.summary)}')
    with open('savedInfo/generatorSummary.json', mode = 'w') as saveModel:
        saveModel.write(model.to_json())
    #save model: figure out why its not working
    #keras.utils.plot_model(model = model, to_file = 'savedInfo/generator.png', show_shapes = True, show_layer_names = True, show_layer_activations = True)

'''
helper function for makeGenerator. Creates the residual blocks
imgDim: an int that represents the length of one axis of an image (should be square!!!)
kernelDim2: a tuple/list that represents the dimensions of each kernel
stride2: a tuple/list that represents how far the kernel moves over the input
padding: a string that represents the 'buffer' around the inputs as they get downsampled
epsilon: a float that represents the learning rate of the model
input:   a keras.layers.Layer instance
returns: a keras.layers.Layer instance that is equal to the input + the res block
'''
def getResNetBlock(imgDim:int, kernelDim:tuple|list, stride:tuple|list, padding:str, epsilon:float, input):
    logger.info(f'getResNetBlock called. values:\timgDim: {imgDim}\tkernelDim: {kernelDim}\tstride: {stride}\tpadding: {padding}\tinput: {input}')
    weightInit = keras.initializers.RandomNormal(stddev = 0.02, seed = getRandomSeed()) #TODO: experiment with this
    
    #resLayer 1
    layer = Conv2D(filters = imgDim, kernel_size = kernelDim, strides = stride, padding = padding, kernel_initializer = weightInit)(input)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.Activation('relu')(layer)
    logger.info(f'resLayer0 created. Value: {layer}\tTarget activation size: (256,64,64)')

    #resLayer 2
    layer = Conv2D(filters = imgDim, kernel_size = kernelDim, strides = stride, padding = padding, kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.Activation('relu')(layer)
    logger.info(f'resLayer1 created. Value: {layer}\tTarget activation size: (256,64,64)')

    return keras.layers.Add()((input, layer))

'''
makes a discriminator CNN for CycleGAN
imgDim: a tuple/list that represents the dimensions of the input. (should be x,y,RGB)
hyperparameterData: a tuple/list that represents an amount of hyperparameters for experimentation
'''
def makeDiscriminator(imgDim:tuple|list, hyperparameterData:tuple|list):
    logger.info(f'makeDiscriminator called.\timgDim: {imgDim}\thyperparameterData: {hyperparameterData}')

    filters    = hyperparameterData[0]
    kernelDim1 = hyperparameterData[1] #kernel size for first hidden layer
    kernelDim2 = hyperparameterData[2] #kernel size for all other layers
    padding    = hyperparameterData[4]
    stride1    = hyperparameterData[5] #stride for final 2 layers
    stride3    = hyperparameterData[7] #stride for all other layers
    alpha      = hyperparameterData[8] #hyperparameter that decreases gradient in LeakyReLU if x < 0
    epsilon    = hyperparameterData[9] #learning rate

    weightInit = keras.initializers.RandomNormal(stddev = 0.02, seed = getRandomSeed()) #TODO: experiment with this

    input = keras.Input(shape = imgDim) #sets the input type

    #first hidden conv layer
    layer = Conv2D(filters = filters, kernel_size = kernelDim1, strides = stride3, padding = padding, kernel_initializer = weightInit)(input)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer) #activation layer TODO experiment with other activation functions

    #second hidden conv layer
    layer = Conv2D(filters = filters*2, kernel_size = kernelDim2, strides = stride3, padding = padding, kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)

   #third hidden conv layer
    layer = Conv2D(filters = filters*4, kernel_size = kernelDim2, strides = stride3, padding = padding, kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)

    #filters = filters*8, as this seems to be the recommended flow of CycleGAN. Experiment/Research required
    layer = Conv2D(filters = filters*8, kernel_size = kernelDim2, strides = stride1, padding = padding, kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.LeakyReLU(alpha = alpha)(layer)

    #Output layer
    layer = Conv2D(filters = 1, kernel_size = kernelDim2, strides = stride1, padding = padding, kernel_initializer = weightInit)(layer)

    #compiles model and returns an untrained discriminator CNN
    model = keras.Model(input, layer) #our model
    model.compile(optimizer = 'Adam', loss = 'mse')
    logging.info(f'Discriminator Summary: {str(model.summary)}')
    with open('savedInfo/discriminatorSummary.json', mode = 'w') as saveModel:
        saveModel.write(model.to_json())
    #save model. Figure out why its not working
    #keras.utils.plot_model(model = model, to_file = 'savedInfo/discriminator.png', show_shapes = True, show_layer_names = True, show_layer_activations = True)
    return model

'''
generates a random seed as an int32
returns: an int32 that represents a pseudo-random seed
'''
def getRandomSeed():
    logger.info('getRandomSeed called.')
    random.seed() #initiate the random number generator
    min = 10**(8) #the length of the seed will be 9 digits, min value = 100...0
    max = 9*min + (min - 1) #max value of seed = 999...9
    seed = random.randint(min, max)
    logger.info(f'seed generated: {seed}')
    return seed

