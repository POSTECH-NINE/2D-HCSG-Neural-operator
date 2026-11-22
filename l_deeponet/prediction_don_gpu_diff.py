import utils 
import numpy as np
import arguments_node_based
import tensorflow as tf
import deeponet_utils_gpu 
import pickle
import ae_utils_gpu 
import random
from tensorflow.keras.models import Model
import time

seed = arguments_node_based.DatasetParams.seed
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
    autoencoder = ae_utils_gpu.MLAE_HCSG(arguments_node_based.DatasetParams.latent_dim)
    autoencoder = tf.keras.models.load_model(
    save_path,
    custom_objects={"MLAE_HCSG": ae_utils_gpu.MLAE_HCSG}
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
    
    deeponet_model = deeponet_utils_gpu.DeepONet_Model_multiscale_custom(Par)
    deeponet_model = tf.keras.models.load_model(save_path, 
                                               custom_objects={"DeepONet_Model_multiscale_custom": deeponet_utils_gpu.DeepONet_Model_multiscale_custom})
    return deeponet_model

def preprocess_inputs_2dcnn(x: tf.Tensor) -> tuple:
    """ 
    Preprocesses the inputs for the DeepONet model.
    
    Args:
        x (np.ndarray): Input feature array of shape (n, latent_dim).

    Returns:
        tuple: 
            - X_func (np.ndarray): Reshaped input features for the DeepONet model of shape (n, ld_sqrt, ld_sqrt, 1).
            - X_loc (np.ndarray): Linearly spaced array of location inputs of shape (nt, 1).
    """ 
    n = tf.shape(x)[0]
    ld = arguments_node_based.DatasetParams.latent_dim
    ld_sqrt = int(tf.math.sqrt(tf.cast(ld, tf.float32)))

    X_func = tf.reshape(x, [n, ld_sqrt, ld_sqrt, 1])
    nt = arguments_node_based.DatasetParams.nt
    X_loc = tf.linspace(0.0, 1.0, nt)
    X_loc = tf.reshape(X_loc, [nt, 1])    
    return X_func, X_loc

def preprocess_inputs_mlp(x: np.ndarray) -> tuple:
    """ 
    Preprocesses the inputs for the DeepONet model.
    
    Args:
        x (np.ndarray): Input feature array of shape (n, latent_dim).

    Returns:
        tuple: 
            - X_func (np.ndarray): Reshaped input features for the DeepONet model of shape (n, ld_sqrt, ld_sqrt, 1).
            - X_loc (np.ndarray): Linearly spaced array of location inputs of shape (nt, 1).
    """ 
    n = tf.shape(x)[0]
    ld = arguments_node_based.DatasetParams.latent_dim
    X_func = tf.reshape(x, [n, ld])
    X_loc = tf.linspace(0.0, 1.0, arguments_node_based.DatasetParams.nt)
    X_loc = tf.reshape(X_loc, [arguments_node_based.DatasetParams.nt, 1])
    return X_func, X_loc

def preprocess_inputs_mlp_diff(x: np.ndarray) -> tuple:
    """ 
    Preprocesses the inputs for the DeepONet model.
    
    Args:
        x (np.ndarray): Input feature array of shape (n, latent_dim).

    Returns:
        tuple: 
            - X_func (np.ndarray): Reshaped input features for the DeepONet model of shape (n, ld_sqrt, ld_sqrt, 1).
            - X_loc (np.ndarray): Linearly spaced array of location inputs of shape (nt, 1).
    """ 
    n = tf.shape(x)[0]
    ld = arguments_node_based.DatasetParams.latent_dim
    X_func = tf.reshape(x, [n, ld])
    X_loc = tf.linspace(0.0, 1.0, arguments_node_based.DatasetParams.nt-1)
    X_loc = tf.reshape(X_loc, [arguments_node_based.DatasetParams.nt-1, 1])
    return X_func, X_loc

def get_Par() -> dict:
    """ 
    Loads the parameter dictionary from a pickle file.
    
    Returns:
        dict: A dictionary containing the model's hyperparameters and configuration details.
    """ 
    pickle_path = arguments_node_based.PredictionParams.load_pickle_path
    with open(pickle_path, "rb") as f:
        Par = pickle.load(f)
    return Par


if __name__ == "__main__":
    test_data_path = arguments_node_based.PredictionParams.test_data_path
    test_data =  np.load(test_data_path)  ##### interpolation data #####
    initial_condition = test_data['inputs']
    # initial_condition_norm = utils.normalize_inputs(initial_condition)
    # 오토인코더 학습에 outputs 사용
    # minmax_path = arguments_node_based.PredictionParams.minmax_path
    initial_condition_norm = utils.normalize_diff_press(initial_condition) 
    ae_model_path = arguments_node_based.PredictionParams.AE_model_path
    ae_model = load_ae_weights(ae_model_path)
    inputs = get_deeponet_targets(ae_model, initial_condition_norm)
    print(f"Initial condition shape: {initial_condition_norm.shape}")
    print(f"PCA transformed data shape: {inputs.shape}")
    
    # Load DeepONet model
    deeponet_save_path = arguments_node_based.PredictionParams.DON_model_path
    Par = get_Par()
    print(f"Parameters: {Par}")
    deeponet_model = load_deeponet(deeponet_save_path, Par)   ### 모델 불러오기 확인! ### 현재 mlp
    print(f"DeepONet model loaded successfully")
    
    start_time = time.time()
    # Preprocess test inputs
    test_inputs = inputs
    print(f"Test inputs shape: {test_inputs.shape}")
    X_func, X_loc = preprocess_inputs_mlp_diff(test_inputs)  ### 데이터 전처리 확인! ###
    print(f"Preprocessed data shapes: {X_func.shape}, {X_loc.shape}")
    
    # Get predictions from DeepONet
    predictions = get_deeponet_predictions(X_func, X_loc, deeponet_model) ### deeponet model 확인! ### 
    print(f"Predictions shape: {predictions.shape}")
    
    # Get predictions from Autoencoder
    ae_outputs = get_ae_predictions(predictions, ae_model)
    final_outputs = np.array(ae_outputs)
    final_outputs_inversed = utils.de_normalize_diff_press(final_outputs)

    # --- 초기 조건 더하기 ---
    # initial_condition shape: (n, 28877)
    # final_outputs_inversed shape: (n, 99, 28877)
    initial_condition_expanded = initial_condition[:, np.newaxis, :]  # (n, 1, 28877)
    final_outputs_inversed = final_outputs_inversed + initial_condition_expanded
    # --- ---
    
    print(f"final_outputs: {final_outputs_inversed.shape}")
    np.save(arguments_node_based.PredictionParams.prediction_save_path, final_outputs_inversed)
    end_time = time.time()
    print(f"Code Time taken: {end_time - start_time} seconds")