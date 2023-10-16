import os
from os.path import join, exists
from glob import glob

import subprocess
import sys

from natsort import natsorted

import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd


def read_ucmerced(path_data, SEED):
  """Function to read images paths of the UC Merced dataset

  Args:
    path_data (string): path to the UC Merced dataset.
    SEED (int): seed to shuffle image paths.

  Returns:
    df: dataframe with information about the image paths and their
      corresponding classes as string and int.
    n_classes: number of classes available in the dataset.
  """
  # List with all images in the folder
  list_img = glob(join(path_data, "**", "*.tif"), recursive=True)
  list_img = natsorted(list_img, key=lambda y: y.lower())

  # Dataframe for better management of image paths
  df = pd.DataFrame(list_img, columns=["path_image"])

  # Getting class name from filename path
  df["class_str"] = df["path_image"].apply(lambda x: x.split(os.sep)[-2])

  classes = np.unique(df["class_str"].values)
  n_classes = len(classes)

  classes_int = np.arange(len(classes))
  classes_dict = dict(zip(classes, classes_int))

  # Applying the dictionary to the column "class"
  df["class_int"] = df["class_str"].apply(lambda x: classes_dict[x])

  # Shuffle the dataframe rows without keeping the old index
  df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

  return df, n_classes


def download_ucmerced(save_dir):  
  """Function to download the UC Merced dataset

  Args:
    save_dir (string): path to save the dataset
  """
  url_dataset = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
  filename = "UCMerced_LandUse.zip"

  if not exists("UCMerced_LandUse"):
    try:
      import wget
    except ImportError:
      subprocess.check_call([sys.executable, "-m", "pip", "install", 'wget'])
    finally:
      import wget

    f = wget.download(url_dataset, save_dir)
    import zipfile
    with zipfile.ZipFile(filename, "r") as zip_ref:
      zip_ref.extractall(".")
    os.remove(join(save_dir, filename))

def train_val_test_split(df, val_size, test_size, SEED):
  """Function to create three disjoint sets for train, validation
  and test.

  Args:
    df (dataframe): pandas dataframe with information about the
      images paths and their corresponding class.
    val_size (float): percentage of the dataset used for validation.
    test_size (float): percentage of the dataset used for test.
    SEED (int): seed to split the dataset.

  Returns:
    splits: dictionary with the images and classes for train,
      validation and test.
  """
  splits = dict()
  # Train and test split
  # 80% for training, 20% for testing
  x_train, x_test, y_train, y_test = train_test_split(df["path_image"].values, 
                                                      df["class_int"].values,                                                     
                                                      test_size=test_size,
                                                      stratify=df["class_int"].values,
                                                      random_state=SEED)
  # Train and validation split
  # 80% for training: 75% for training, 25% for validation
  val_size_relative = val_size/(1-test_size)
  x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                    y_train,
                                                    test_size=val_size_relative, 
                                                    stratify=y_train,
                                                    random_state=SEED)
  splits["x_train"] = x_train
  splits["y_train"] = y_train
  splits["x_val"] = x_val
  splits["y_val"] = y_val
  splits["x_test"] = x_test
  splits["y_test"] = y_test
  
  return splits