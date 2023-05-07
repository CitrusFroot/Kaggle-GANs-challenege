import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
  monetData = loadData('gan-getting-started\monet_jpg')
  photoData = loadData('gan-getting-started\photo_jpg')

  visualizeData(monetData)

#loads image data from files
#returns: 2 tf.data.Dataset objects, one holding the training data and the other holding images to be transformed
def loadData(address):
    dataset = tf.keras.utils.image_dataset_from_directory(directory = address,
                                                          labels = None,
                                                          color_mode = 'rgb',
                                                          image_size = IMAGESIZE,
                                                          batch_size = BATCHSIZE,
                                                          shuffle = False,
                                                          validation_split = None)
    dataset = dataset.map(lambda x: x/255) #normalizes the data
    return dataset

#visualizes the first 9 entries of a dataset
#dataset: a tf.data.Dataset
#saves a png of 9 examples of data
def visualizeData(dataset):
    fig = plt.figure(figsize = [5,5]) #sets size of figure
    plt.title('9 Entries in data')
    
    for i, element in enumerate(next(iter(dataset))):
        if(i < 9):
            fig.add_subplot(3,3,i+1)
            plt.imshow(element)
            plt.axis('off')

    plt.savefig('savedInfo/nineExamples.png')

def getPCA(dataset):
   #TODO
   pass

def generateArt():
   pass

def train():
   pass


if __name__ == "__main__":
    IMAGESIZE = (256,256)
    BATCHSIZE = 64
    NUMBATCHES = np.ceil((300/BATCHSIZE))
    EPOCHS = 100
    VALIDATIONSPLIT = 0.3

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

'''