from tensorflow import keras
from keras.layers import Conv2D
import logging
import random
from ReissLib.debugMessages import createLogger as cl

logger = cl(level = 'debug', dir = 'Log files') #creates the debug messager

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
i: a char, representing if it is generator A or B
returns: a tf.keras.model instance. UNTRAINED
'''
def make_generator(imgDim:tuple|list, hyperparameters:tuple|list, num_res_blocks:int, i:str):
    logger.debug(f'makeGenerator called.\timgDim: {imgDim}')
    filters    = hyperparameters[0]
    kernel_dim1 = hyperparameters[1] #first kernel
    kernel_dim2 = hyperparameters[2] #kernel used for all other layers
    padding    = hyperparameters[4]
    stride1    = hyperparameters[5] #stride for input and output layers
    stride2    = hyperparameters[6] #stride for all other layers
    alpha      = hyperparameters[8] #hyperparameter that decreases gradient in LeakyReLU if x < 0
    epsilon    = hyperparameters[9] #learning rate

    weight_init = keras.initializers.RandomNormal(stddev = 0.02, seed = get_random_seed()) #TODO: experiment with this
    input = keras.Input(shape = imgDim) #sets the input type
    
    #debug warning. Input MUST be 256x256 in RGB
    if(imgDim != (256,256,3)):
        logger.warn(f'input layer is not of the correct target size. current size is {imgDim}, should be (256, 256, 3)')

    #first hidden convolutional layer. Activation size should be (256,256,64)
    layer = make_conv2D(filters=filters, 
                        kernel_size=kernel_dim1, 
                        strides=stride1, 
                        kernel_initializer=weight_init, 
                        prev_layer=input)
    layer = norm_and_activation(epsilon=epsilon, alpha=alpha, prev_layer=layer)
    logger.debug(f'hConvLayer0 created. Value: {layer}\tTarget activation size: (256, 256, 64)')

    #second hidden conv layer. Activation size should be (128,128,128)
    layer = make_conv2D(filters=filters*2, 
                        kernel_size=kernel_dim2, 
                        strides=stride2, 
                        kernel_initializer=weight_init, 
                        prev_layer=layer)
    layer = norm_and_activation(epsilon=epsilon, alpha=alpha, prev_layer=layer)
    logger.debug(f'hConvLayer1 created. Value: {layer}\tTarget activation size: (128,128,128)')

    #third hidden conv layer. Activation size should be (64,64,256)
    layer = make_conv2D(filters=filters*4, 
                        kernel_size=kernel_dim2, 
                        strides=stride2, 
                        kernel_initializer=weight_init, 
                        prev_layer=layer)
    layer = norm_and_activation(epsilon=epsilon, alpha=alpha, prev_layer=layer)
    logger.debug(f'hConvLayer2 created. Value: {layer}\tTarget activation size: (64, 64, 256)')

    #adds x amount of residual layers. 9 for 256x256 TODO: verify math and find evidence to back up. !!!STRIDE MUST BE (1,1) given everything else is standard
    for _ in range(num_res_blocks):
        layer = get_resnet_block(imgDim[0], kernel_dim2, stride1, padding, epsilon, layer) #when finished, layer's activation size should be (64,64,256)
    logger.debug(f'resnet block completed. Layer: {layer}\tTarget activation size: (64, 64, 256)')

    #we transpose to upscale the image. Every conv2D layer decreases the dimension of the image. These two layers restore the original dimensions
    layer = keras.layers.Conv2DTranspose(filters = filters*2, kernel_size = kernel_dim2, strides = stride2, padding = padding, output_padding = (1,1), kernel_initializer = weight_init)(layer)
    layer = norm_and_activation(epsilon=epsilon, alpha=alpha, prev_layer=layer)
    logger.debug(f'first transposed Conv2D layer generated. Layer: {layer}') #TODO get target size

    layer = keras.layers.Conv2DTranspose(filters = filters, kernel_size = kernel_dim2, strides = stride2, padding = padding, output_padding = (1,1), kernel_initializer = weight_init)(layer)
    layer = norm_and_activation(epsilon=epsilon, alpha=alpha, prev_layer=layer)
    logger.debug(f'final transposed Conv2D layer generated. Layer: {layer}') #TODO get target size

    #final output layer
    layer = make_conv2D(filters=imgDim[2],
                        kernel_size=kernel_dim2,
                        strides=stride1,
                        kernel_initializer=weight_init,
                        prev_layer=layer)
    logger.debug(f'final layer of generator made. layer: {layer}') #TODO get target activation size

    #compiles the untrained model and returns it to driver.py
    model = keras.Model(input, layer)
    model.compile(optimizer = 'adam', loss = 'mse')
    logger.info(f'Generator Summary: {str(model.summary)}')
    with open(f'savedInfo/generator{i}_summary.json', mode = 'w') as saveModel:
        saveModel.write(model.to_json())
        logger.debug('Saved model to json file')
    #save model: figure out why its not working
    #keras.utils.plot_model(model = model, to_file = 'savedInfo/generator.png', show_shapes = True, show_layer_names = True, show_layer_activations = True)
    return model
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
def get_resnet_block(imgDim:int, kernelDim:tuple|list, stride:tuple|list, padding:str, epsilon:float, input):
    logger.debug(f'getResNetBlock called. values:\timgDim: {imgDim}\tkernelDim: {kernelDim}\tstride: {stride}\tpadding: {padding}\tinput: {input}')
    weightInit = keras.initializers.RandomNormal(stddev = 0.02, seed = get_random_seed()) #TODO: experiment with this
    
    #resLayer 1
    layer = Conv2D(filters = imgDim, kernel_size = kernelDim, strides = stride, padding = padding, kernel_initializer = weightInit)(input)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.Activation('relu')(layer)
    logger.debug(f'resLayer0 created. Value: {layer}\tTarget activation size: (256,64,64)')

    #resLayer 2
    layer = Conv2D(filters = imgDim, kernel_size = kernelDim, strides = stride, padding = padding, kernel_initializer = weightInit)(layer)
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(layer)
    layer = keras.layers.Activation('relu')(layer)
    logger.debug(f'resLayer1 created. Value: {layer}\tTarget activation size: (256,64,64)')

    return keras.layers.Add()((input, layer))

'''
makes a discriminator CNN for CycleGAN
imgDim: a tuple/list that represents the dimensions of the input. (should be x,y,RGB)
hyperparameterData: a tuple/list that represents an amount of hyperparameters for experimentation
'''
def make_discriminator(imgDim:tuple|list, hyperparameters:tuple|list, i:str):
    logger.debug(f'makeDiscriminator called.\timgDim: {imgDim}\thyperparameterData: {hyperparameters}')

    filters    = hyperparameters[0]
    kernelDim1 = hyperparameters[1] #kernel size for first hidden layer
    kernelDim2 = hyperparameters[2] #kernel size for all other layers
    padding    = hyperparameters[4]
    stride1    = hyperparameters[5] #stride for final 2 layers
    stride3    = hyperparameters[7] #stride for all other layers
    alpha      = hyperparameters[8] #hyperparameter that decreases gradient in LeakyReLU if x < 0
    epsilon    = hyperparameters[9] #learning rate

    weightInit = keras.initializers.RandomNormal(stddev = 0.02, seed = get_random_seed()) #TODO: experiment with this

    input = keras.Input(shape = imgDim) #sets the input type

    #first hidden conv layer
    

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
    with open(f'savedInfo/discriminator{i}_summary.json', mode = 'w') as saveModel:
        saveModel.write(model.to_json())
    #save model. Figure out why its not working
    #keras.utils.plot_model(model = model, to_file = 'savedInfo/discriminator.png', show_shapes = True, show_layer_names = True, show_layer_activations = True)
    return model

'''
generates a random seed as an int32
returns: an int32 that represents a pseudo-random seed
'''
def get_random_seed():
    logger.debug('getRandomSeed called.')
    random.seed() #initiate the random number generator
    min = 10**(8) #the length of the seed will be 9 digits, min value = 100...0
    max = 9*min + (min - 1) #max value of seed = 999...9
    seed = random.randint(min, max)
    logger.info(f'seed generated: {seed}')
    return seed

'''
loads in saved model jsons and trains them for a single iteration
saves the trained models for the next step
'''

def make_conv2D(filters, kernel_size, strides, kernel_initializer prev_layer):
    layer = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same', kernel_initializer = kernel_initializer)(prev_layer)
    return layer

def norm_and_activation(epsilon, alpha, prev_layer, is_leaky=True):
    layer = keras.layers.BatchNormalization(axis = -1, epsilon = epsilon)(prev_layer)
    
    if is_leaky:
        layer = keras.layers.LeakyReLU(alpha = alpha)(layer) #activation layer TODO experiment with other activation functions
    else:
        layer = keras.layers.Activation('relu')(layer)

    return layer

def do_training_step():
    pass
