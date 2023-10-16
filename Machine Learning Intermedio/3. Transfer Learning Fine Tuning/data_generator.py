from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import tifffile
import numpy as np


class DataGenerator(keras.utils.Sequence):
  def __init__(self,                
               path_images, 
               labels,
               batch_size,               
               n_classes, 
               target_size=256,
               augmentation=None,
               shuffle=True):
    """Constructor.

    Args:
        path_images (np.array): array with images path
        labels (np.array): array with the corresponding labels
        batch_size (int): number of samples per batch
        n_classes (int): number of classes
        target_size (int): size of each image in a batch. 
          Defaults to 256.
        augmentation (bool, optional): If True, data augmentation is
         applied to each sample.
        shuffle (bool, optional): If True, all samples are shuffled
          after each epoch. Defaults to True.
    """
    self.path_images = path_images
    self.labels = labels
    self.batch_size = batch_size
    self.n_classes = n_classes
    self.targe_size = target_size
    self.augmentation = augmentation
    self.shuffle= shuffle
      
  def on_epoch_end(self):
    """Executes at the end of an epoch.
    """
    if self.shuffle:
      self.path_images, self.labels = shuffle(self.path_images, self.labels)

  def __len__(self):
    """Computes the number of batches.

    Returns:
        int: number of batches
    """
    return np.ceil(self.path_images.shape[0]/self.batch_size).astype("int")
  
  def __get_image(self, path_image):
    """Function to read an image and preprocess it.

    Args:
        path_image (string): image path

    Returns:
        np.array: image array with shape [batch_size,width,height,channels]
    """
    x_sample = tifffile.imread(path_image)    
    x_sample = np.resize(x_sample, (self.targe_size,self.targe_size,3))

    if self.augmentation is not None:
        augmented = self.augmentation()(image=x_sample)
        x_sample = augmented["image"]

    x_sample = np.expand_dims(x_sample, axis=0)
    x_sample = x_sample.astype("float")
    return x_sample

  def __get_label(self, label):
    """Function to read a label and preprocess it.

    Args:
        label (int): image label for classification

    Returns:
        np.array: array with shape [batch_size,n_classes]
    """
    y_sample = to_categorical(label, 
                              num_classes=self.n_classes)
    y_sample = np.expand_dims(y_sample, axis=-1)
    return y_sample
     
  def __getitem__(self, idx):
    """Function that provides a batch of data.

    Args:
        idx (int): batch index

    Returns:
        tuple: tuple of np.arrays with the image and its label.
    """
    i = idx * self.batch_size
      
    current_batch_size = self.batch_size
    if (idx+1) == self.__len__():
      current_batch_size = len(self.path_images[i:])

    batch_images = self.path_images[i : i + current_batch_size]
    batch_labels = self.labels[i : i + current_batch_size]

    x = np.zeros((current_batch_size, 
                  self.targe_size, 
                  self.targe_size, 
                  3), 
                  dtype=np.float32)
    
    y = np.zeros((current_batch_size,
                  self.n_classes,
                  1),
                  dtype=np.float32)
    
    # read data
    for j, (path_image,label) in enumerate(zip(batch_images,batch_labels)):
      # Reading each image
      x_sample = self.__get_image(path_image)        
      # Get the label
      y_sample = self.__get_label(label)
        
      x[j,...] = x_sample
      y[j,...] = y_sample
    
    return x, y