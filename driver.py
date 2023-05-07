import tensorflow as tf
import numpy as np
import csv


def main():
  xy, apply = loadData()
  visualizeData(xy)
  train()

#loads image data from tfrec files
#useJPG: boolean, determines if we use jpgs instead of tfrec files
#returns: 2 tf.data.Dataset objects, one holding the training data and the other holding images to be transformed
def loadData(useJPG = 0):
    if(useJPG == 0): #loads in the dataset from .tfrec files
        trainingDataset = tf.data.TFRecordDataset('gan-getting-started\monet_tfrec').shuffle(buffer_size = 300).batch(BATCHSIZE)
        trainingDataset = (trainingDataset.take(np.int64((1 - VALIDATIONSPLIT) * 300)),   #training split
                           trainingDataset.skip(np.int64((1 - VALIDATIONSPLIT) * 300)))   #testing split
        transformDataset = tf.data.TFRecordDataset('gan-getting-started\photo_tfrec').batch(BATCHSIZE)
        return trainingDataset, transformDataset
    
    else: #we want to use jpgs
        trainingDataset = tf.keras.utils.image_dataset_from_directory(directory= 'gan-getting-started\monet_jpg',
                                                                      labels = 'inferred',
                                                                      color_mode = 'rgb',
                                                                      batch_size = BATCHSIZE,
                                                                      image_size = IMAGESIZE,
                                                                      shuffle = True,
                                                                      seed = 1912,
                                                                      validation_split= VALIDATIONSPLIT)
        
        transformDataset = tf.keras.utils.image_dataset_from_directory(directory= 'gan-getting-started\photo_jpg',
                                                                      labels = 'inferred',
                                                                      color_mode = 'rgb',
                                                                      batch_size = BATCHSIZE,
                                                                      image_size = IMAGESIZE,
                                                                      shuffle = False,
                                                                      validation_split= None)
        return trainingDataset, transformDataset


def visualizeData(dataset):
   print(tf.)

def generateArt():
   pass

def train():
   pass


if __name__ == "__main__":
    IMAGESIZE = 0
    BATCHSIZE = 64
    EPOCHS = 100
    VALIDATIONSPLIT = 0.3
    main()