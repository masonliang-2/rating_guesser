import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple

DATASET_PATH = ""
DATASET_TESTING_PATH = ""

def setup_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    return train_dataset, val_dataset
