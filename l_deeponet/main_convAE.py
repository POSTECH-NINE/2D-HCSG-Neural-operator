import os
import tensorflow as tf
import random
import numpy as np
import arguments_structure
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # GPU 메모리 증가 방식 사용
        print("GPU memory growth mode set")
    except RuntimeError as e:
        print(e)
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
seed = arguments_structure.DatasetParams.seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
from sklearn.model_selection import train_test_split
from typing import Tuple
import convAE
import utils
import time

def split_dataset(
    x_y_norm: np.ndarray, test_size: float = 0.1, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the normalized dataset into training and testing sets, and reshapes them
    based on the `nx` and `ny` parameters specified in `arguments.DatasetParams`.

    Args:
        x_y_norm (np.ndarray): The normalized dataset to split.
        test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.1.
        random_state (int, optional): Controls the shuffling applied to the data before splitting. Default is 42.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - x_y_train (np.ndarray): Reshaped training data.
            - x_y_test (np.ndarray): Reshaped testing data.
    """
    x_y_norm = x_y_norm.swapaxes(1, 2)
    x_y_norm = np.expand_dims(x_y_norm, axis = -1)
    x_y_train, x_y_test = train_test_split(
        x_y_norm, test_size=test_size, random_state=random_state
    )
    return x_y_train, x_y_test

if __name__ == "__main__":
    inputs_re, x_y_data = utils.load_data_structure()
    n_samples = inputs_re.shape[0]
    print(f"Number of samples: {n_samples}")
    print(f"x_y_data shape: {x_y_data.shape}")

    x_y_norm = utils.normalize_diff_press(x_y_data)
    x_y_train, x_y_test = split_dataset(x_y_norm)
    print(f"Training data shape: {x_y_train.shape}")
    print(f"Testing data shape: {x_y_test.shape}")
    start_time = time.time()
    model, _ = convAE.train_model(x_y_train, x_y_test, save=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # l2, mse = metrics.calculate_errors(x_y_test, model)
