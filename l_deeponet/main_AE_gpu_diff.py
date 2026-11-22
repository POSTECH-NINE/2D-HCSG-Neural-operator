import os
import tensorflow as tf
import random
import numpy as np
import arguments_node_based
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
seed = arguments_node_based.DatasetParams.seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
from sklearn.model_selection import train_test_split
from typing import Tuple
import ae_utils_gpu
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
    x_y_train, x_y_test = train_test_split(
        x_y_norm, test_size=test_size, random_state=random_state
    )
    x_y_train = x_y_train.reshape(
        -1, arguments_node_based.DatasetParams.n_node)
    x_y_test = x_y_test.reshape(
        -1, arguments_node_based.DatasetParams.n_node)
    return x_y_train, x_y_test

def split_dataset_ic1(
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
    
    x_y_train = x_y_norm.reshape(
        -1, arguments_node_based.DatasetParams.n_node)
    x_y_test = x_y_train
    return x_y_train, x_y_test



if __name__ == "__main__":
    inputs_re, x_y_data = utils.load_data_node_based()
    n_samples = inputs_re.shape[0]
    print(f"Number of samples: {n_samples}")
    
    # save minmax values to normalize data
    # save_minmax_path = arguments_node_based.AETrainingParams.minmax_path
    # utils.save_minmax_values(inputs_re, x_y_data, save_minmax_path)

    # normalize data
    # minmax_path = arguments_node_based.AETrainingParams.minmax_path
    # x_y_norm = utils.normalize_outputs_with_saved_minmax(x_y_data, minmax_path=minmax_path)

    x_y_norm = utils.normalize_diff_vel(x_y_data)
    # split data
    x_y_train, x_y_test = split_dataset(x_y_norm)
    
    # train model
    print(f"Training data shape: {x_y_train.shape}")
    print(f"Testing data shape: {x_y_test.shape}")
    start_time = time.time()
    model, _ = ae_utils_gpu.train_model(x_y_train, x_y_test, save=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")