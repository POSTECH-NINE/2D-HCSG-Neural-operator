import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 이 줄을 맨 위에 추가
import utils 
import numpy as np
import arguments_structure
import tensorflow as tf
import deeponet_utils_gpu_structure 
import pickle
import convAE 
import random
from tensorflow.keras.models import Model
import time
from typing import Tuple

seed = arguments_structure.DatasetParams.seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def load_ae_weights(save_path: str) -> object:
    """
    Load the weights of an autoencoder model.

    Args:
        save_path (str): Path to the saved weights file.

    Returns:
        object: The autoencoder model with loaded weights.
    """
    autoencoder = convAE.convAE_HCSG(256)
    autoencoder = tf.keras.models.load_model(
    save_path,
    custom_objects={"convAE_HCSG": convAE.convAE_HCSG}
)
    return autoencoder

def get_deeponet_targets(ae_model: Model, outputs_norm: tf.Tensor) -> tf.Tensor:
    return ae_model.encoder(outputs_norm)


def get_deeponet_predictions(X_func: tf.Tensor, X_loc: tf.Tensor, model: tf.keras.Model) -> tf.Tensor:
    """ 
    Returns the predictions of the DeepONet model.
    
    Args:
        X_func (np.ndarray): The input function for the DeepONet model of shape (batch_size, ld_sqrt, ld_sqrt, 1).
        X_loc (np.ndarray): The input location for the DeepONet model of shape (nt, 1).
        model (tf.keras.Model): The pre-trained DeepONet model.

    Returns:
        np.ndarray: The predictions of the DeepONet model with shape (batch_size, nt, latent_dim).
    """ 
    y_pred = model(X_func, X_loc, training=False)
    return y_pred

def get_ae_predictions(latent_data: tf.Tensor, model: tf.keras.Model) -> list:
    """ 
    Decodes the latent data into the original space using the Autoencoder model.
    
    Args:
        latent_data (np.ndarray): The latent space data to be decoded, shape (batch_size, nt, latent_dim).
        model (tf.keras.Model): The Autoencoder model with a decoder.

    Returns:
        list: A list of decoded outputs where each element corresponds to one prediction from the Autoencoder model.
    """ 
    ae_output_array = []
    for i in range(latent_data.shape[0]):
        curr_latent_data = latent_data[i]
        print(f"Current latent data shape: {curr_latent_data.shape}")
        curr_output = model.decoder(curr_latent_data)
        ae_output_array.append(curr_output)
    return ae_output_array

def load_deeponet(save_path: str, Par: dict) -> tf.keras.Model:
    
    """ 
    Loads the DeepONet model from saved weights.
    
    Args:
        save_path (str): The file path where the DeepONet weights are stored.
        Par (dict): The parameter dictionary containing the model's configuration details.

    Returns:
        tf.keras.Model: The loaded DeepONet model.
    """ 
    
    deeponet_model = tf.keras.models.load_model(save_path)
    return deeponet_model

def preprocess_2dcnn(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:    
    # 파라미터 설정
    n = np.shape(x)[0]
    ld = arguments_structure.DatasetParams.latent_dim
    nt = arguments_structure.DatasetParams.nt
    ld_sqrt = int(np.sqrt(ld))  # numpy sqrt로 통일
    
    # X_func: (n, ld_sqrt, ld_sqrt, 1)로 reshape
    X_func = np.reshape(x, [n, ld_sqrt, ld_sqrt, 1])
    
    # X_loc: 시간 좌표 생성
    X_loc = np.linspace(0.0, 1.0, nt).reshape(nt, 1)  # 한 줄로 간단히
    return X_func, X_loc

def preprocess_mlp(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:    
    n = tf.shape(x)[0]
    ld = arguments_structure.DatasetParams.latent_dim
    X_func = tf.reshape(x, [n, ld])
    X_loc = tf.linspace(0, 1, arguments_structure.DatasetParams.nt)
    X_loc = tf.reshape(X_loc, [arguments_structure.DatasetParams.nt, 1]) # 
    return X_func, X_loc

def preprocess_mlp_diff(x: np.ndarray) -> Tuple:
    n = tf.shape(x)[0]
    ld = arguments_structure.DatasetParams.latent_dim
    X_func = tf.reshape(x, [n, ld])
    X_loc = tf.linspace(0, 1, arguments_structure.DatasetParams.nt-1)
    X_loc = tf.reshape(X_loc, [arguments_structure.DatasetParams.nt-1, 1])
    return X_func, X_loc

def preprocess_3dcnn(x: np.ndarray):
    n = x.shape[0]
    ld = (3,6,256) # 9,24,256
    nt = arguments_structure.DatasetParams.nt

    # X_func: (n, 9,24,256)
    X_func = np.reshape(x, (n, *ld), order='C')

    # X_loc: (nt,1)
    X_loc = np.linspace(0.0, 1.0, nt).reshape(nt, 1)

    return X_func, X_loc

def get_Par() -> dict:
    """ 
    Loads the parameter dictionary from a pickle file.
    
    Returns:
        dict: A dictionary containing the model's hyperparameters and configuration details.
    """ 
    pickle_path = arguments_structure.PredictionParams.load_pickle_path
    with open(pickle_path, "rb") as f:
        Par = pickle.load(f)
    return Par


if __name__ == "__main__":
    start_time = time.time()
    test_data_path = arguments_structure.PredictionParams.test_data_path
    test_data =  np.load(test_data_path)  ##### interpolation data #####
    initial_condition = test_data['inputs']
    initial_condition = np.nan_to_num(initial_condition, nan=0.0)
    ic = np.expand_dims(initial_condition, axis=-1)
    ic = np.swapaxes(ic, 1, 2)

    initial_condition_norm = utils.normalize_diff_press(ic) 
    ae_model_path = arguments_structure.PredictionParams.AE_model_path
    ae_model = load_ae_weights(ae_model_path)
    inputs = get_deeponet_targets(ae_model, initial_condition_norm)
    print(f"Initial condition shape: {initial_condition_norm.shape}")
    print(f"PCA transformed data shape: {inputs.shape}")
    
    # Load DeepONet model
    deeponet_save_path = arguments_structure.PredictionParams.DON_model_path
    Par = get_Par()
    print(f"Parameters: {Par}")
    deeponet_model = load_deeponet(deeponet_save_path, Par)   ### 모델 불러오기 확인! ### 현재 mlp
    print(f"Loaded DeepONet model class: {type(deeponet_model).__name__}")    
    # Preprocess test inputs
    test_inputs = inputs
    print(f"Test inputs shape: {test_inputs.shape}")
    X_func, X_loc = preprocess_mlp_diff(test_inputs)  ### 데이터 전처리 확인! ###
    print(f"Preprocessed data shapes: {X_func.shape}, {X_loc.shape}")
    
    # Get predictions from DeepONet
    predictions = get_deeponet_predictions(X_func, X_loc, deeponet_model) ### deeponet model 확인! ### 
    print(f"Predictions shape: {predictions.shape}")
    
    # Get predictions from Autoencoder
    ae_outputs = get_ae_predictions(predictions, ae_model)
    final_outputs = np.array(ae_outputs)
    # final_outputs_inversed = utils.denormalize_outputs(final_outputs)
    final_outputs_inversed = utils.de_normalize_diff_press(final_outputs)
    print(f"final_outputs: {final_outputs.shape}")
    final_outputs_inversed = np.swapaxes(final_outputs_inversed, 2,3)
    final_outputs_inversed = np.squeeze(final_outputs_inversed,axis=-1)
    
    # Add initial condition (inputs) to predicted difference
    reshape_initial_condition = initial_condition[:, np.newaxis, :]
    restored_outputs = reshape_initial_condition + final_outputs_inversed
    print(f"reshape_initial_condition: {reshape_initial_condition.shape}")
    print(f"final_outputs_inversed: {final_outputs_inversed.shape}")
    print(f"restored_outputs: {restored_outputs.shape}")

    np.save(arguments_structure.PredictionParams.prediction_save_path, restored_outputs)
    end_time = time.time()
    print(f"Code Time taken: {end_time - start_time} seconds")
    print(f"Loaded DeepONet model class: {type(deeponet_model).__name__}")


    # np.save(arguments_structure.PredictionParams.prediction_save_path, final_outputs_inversed)
    # end_time = time.time()
    # print(f"Code Time taken: {end_time - start_time} seconds")
    # print(f"Loaded DeepONet model class: {type(deeponet_model).__name__}")    
    

