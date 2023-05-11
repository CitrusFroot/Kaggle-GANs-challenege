import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import GANTraining as GAN

def main():
  monetData = loadData('gan-getting-started\monet_jpg')
  photoData = loadData('gan-getting-started\photo_jpg')

  visualizeData(monetData, 1)
  visualizeData(photoData, 0)

  discriminatorA = GAN.makeDiscriminator(IMAGE_SIZE, HYPERPARAMETER_DATA) #monet
  generatorA     = GAN.makeGenerator(IMAGE_SIZE, HYPERPARAMETER_DATA, NUM_RES_BLOCKS)

  discriminatorB = GAN.makeDiscriminator(IMAGE_SIZE, HYPERPARAMETER_DATA) #normal
  generatorB     = GAN.makeGenerator(IMAGE_SIZE, HYPERPARAMETER_DATA, NUM_RES_BLOCKS)

#loads image data from files
#returns: 2 tf.data.Dataset objects, one holding the training data and the other holding images to be transformed
def loadData(address):
    dataset = tf.keras.utils.image_dataset_from_directory(directory = address,
                                                          labels = None,
                                                          color_mode = 'rgb',
                                                          image_size = (256,256), #we dont use IMAGE_SIZE as this only needs img dimensions, and color channel has been given
                                                          batch_size = BATCH_SIZE,
                                                          shuffle = False,
                                                          validation_split = None)
    dataset = dataset.map(lambda x: x/255) #normalizes the data
    return dataset

#visualizes the first 9 entries of a dataset
#dataset: a tf.data.Dataset
#saves a png of 9 examples of data
def visualizeData(dataset, isMonet):
    fig = plt.figure(figsize = [5,5]) #sets size of figure
    plt.title('9 Entries in data')
    
    for i, element in enumerate(next(iter(dataset))):
        if(i < 9):
            fig.add_subplot(3,3,i+1)
            plt.imshow(element)
            plt.axis('off')

    plt.savefig('savedInfo/nineExamples{}.png'.format(isMonet))

def getPCA(dataset):
   #TODO
   pass

def generateArt():
   pass

def train():
   pass


if __name__ == "__main__":
    IMAGE_SIZE = (256,256,3)
    BATCH_SIZE = 64
    NUM_BATCHES = np.ceil((300/BATCH_SIZE))
    EPOCHS = 100
    VALIDATION_SPLIT = 0.3

    FILTERS       = 64
    KERNEL_SIZE_1 = (7,7)
    KERNEL_SIZE_2 = (3,3)
    KERNEL_SIZE_3 = (4,4)
    PADDING       = 'same' #prevents data loss
    STRIDE_1      = (1,1)
    STRIDE_2      = (2,2)
    STRIDE_3      = (4,4)
    ALPHA         = 0.2
    EPSILON       = 1e-3
    NUM_RES_BLOCKS= 9

    HYPERPARAMETER_DATA = [FILTERS, KERNEL_SIZE_1, KERNEL_SIZE_2, KERNEL_SIZE_3, PADDING, STRIDE_1, STRIDE_2, STRIDE_3, ALPHA, EPSILON]

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()

    print('Number of replicas:', strategy.num_replicas_in_sync)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    main()


'''
============== CREDITS =============
Starting architecture: https://towardsdatascience.com/overview-of-cyclegan-architecture-and-training-afee31612a2f
get random seed: RMPR, stack overflow. https://stackoverflow.com/questions/58468532/generate-random-seed-in-python
'''