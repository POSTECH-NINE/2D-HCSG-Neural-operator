import numpy as np
import arguments_node_based
import json
import arguments_structure
import arguments_structure_press
import arguments_structure_vel

def load_data_node_based() -> tuple[np.ndarray, np.ndarray]:
    n = arguments_node_based.DatasetParams.n
    nt = arguments_node_based.DatasetParams.nt
    n_node = arguments_node_based.DatasetParams.n_node
    data_path = arguments_node_based.DatasetParams.dataset_path
    file = np.load(data_path, allow_pickle=True)
    inputs, outputs = file["inputs"], file["outputs"]
    outputs = outputs.reshape(n * nt, n_node)
    return inputs, outputs


def load_data_node_based_diff() -> tuple[np.ndarray, np.ndarray]:
    n = arguments_node_based.DatasetParams.n
    nt = arguments_node_based.DatasetParams.nt
    n_node = arguments_node_based.DatasetParams.n_node
    data_path = arguments_node_based.DatasetParams.dataset_path
    file = np.load(data_path, allow_pickle=True)
    inputs, outputs = file["inputs"], file["outputs"]
    outputs = outputs[:,1:,:]  
    outputs = outputs.reshape(n * (nt-1), n_node)  
    return inputs, outputs

def load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and processes data from a specified dataset path.

    Reads input and output data from a .npz file, reshapes the outputs
    and inputs based on parameters provided in `arguments.DatasetParams`, and returns them.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - inputs (np.ndarray): Reshaped input data of shape (n, nx * ny).
            - outputs (np.ndarray): Reshaped output data of shape (n * nt, nx * ny).
    """
    data_path = arguments_node_based.DatasetParams.dataset_path
    file = np.load(data_path)
    inputs, outputs = file["inputs"], file["outputs"]
    return inputs, outputs

def load_data_structure() -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and processes data from a specified dataset path.

    Reads input and output data from a .npz file, reshapes the outputs
    and inputs based on parameters provided in `arguments.DatasetParams`, and returns them.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - inputs (np.ndarray): Reshaped input data of shape (n, nx * ny).
            - outputs (np.ndarray): Reshaped output data of shape (n * nt, nx, ny).
    """
    data_path = arguments_structure.DatasetParams.dataset_path
    file = np.load(data_path)
    inputs, outputs = file["inputs"], file["outputs"]

    # fill the nan value 
    inputs = np.nan_to_num(inputs, nan=0.0)
    outputs = np.nan_to_num(outputs, nan=0.0)

    n = arguments_structure.DatasetParams.n
    nt = arguments_structure.DatasetParams.nt
    nx = arguments_structure.DatasetParams.nx
    ny = arguments_structure.DatasetParams.ny
    
    outputs = outputs.reshape(n * nt, nx, ny)
    
    return inputs, outputs

def load_data_structure_vel() -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and processes data from a specified dataset path.

    Reads input and output data from a .npz file, reshapes the outputs
    and inputs based on parameters provided in `arguments.DatasetParams`, and returns them.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - inputs (np.ndarray): Reshaped input data of shape (n, nx * ny).
            - outputs (np.ndarray): Reshaped output data of shape (n * nt, nx, ny).
    """
    data_path = arguments_structure_vel.DatasetParams.dataset_path
    file = np.load(data_path)
    inputs, outputs = file["inputs"], file["outputs"]

    # fill the nan value 
    inputs = np.nan_to_num(inputs, nan=0.0)
    outputs = np.nan_to_num(outputs, nan=0.0)

    n = arguments_structure_vel.DatasetParams.n
    nt = arguments_structure_vel.DatasetParams.nt
    nx = arguments_structure_vel.DatasetParams.nx
    ny = arguments_structure_vel.DatasetParams.ny
    
    outputs = outputs.reshape(n * nt, nx, ny)
    
    return inputs, outputs

def load_data_structure_press() -> tuple[np.ndarray, np.ndarray]:
    """
    Loads and processes data from a specified dataset path.

    Reads input and output data from a .npz file, reshapes the outputs
    and inputs based on parameters provided in `arguments.DatasetParams`, and returns them.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - inputs (np.ndarray): Reshaped input data of shape (n, nx * ny).
            - outputs (np.ndarray): Reshaped output data of shape (n * nt, nx, ny).
    """
    data_path = arguments_structure_press.DatasetParams.dataset_path
    file = np.load(data_path)
    inputs, outputs = file["inputs"], file["outputs"]

    # fill the nan value 
    inputs = np.nan_to_num(inputs, nan=0.0)
    outputs = np.nan_to_num(outputs, nan=0.0)

    n = arguments_structure_press.DatasetParams.n
    nt = arguments_structure_press.DatasetParams.nt
    nx = arguments_structure_press.DatasetParams.nx
    ny = arguments_structure_press.DatasetParams.ny
    
    outputs = outputs.reshape(n * nt, nx, ny)
    
    return inputs, outputs


def load_data_structure_diff() -> tuple[np.ndarray, np.ndarray]:
    data_path = arguments_structure.DatasetParams.dataset_path
    file = np.load(data_path)
    inputs, outputs = file["inputs"], file["outputs"]
    inputs = np.nan_to_num(inputs, nan=0.0)
    outputs = np.nan_to_num(outputs, nan=0.0)
    n = arguments_structure.DatasetParams.n
    nt = arguments_structure.DatasetParams.nt
    nx = arguments_structure.DatasetParams.nx
    ny = arguments_structure.DatasetParams.ny
    outputs = outputs[:,1:,:]  
    outputs = outputs.reshape(n * (nt-1), nx, ny)  
    return inputs, outputs

# 0.8m/s 유동장을 기준으로 정규화
# absolute 값 정규화
# def normalize_data_big_scale(data: np.ndarray) -> np.ndarray:
#     min_value = np.float32(-1952.538701) 
#     max_value = np.float32(602.8355304)
#     return (data - min_value) / (max_value - min_value)

# def inverse_normalize_data_big_scale(data: np.ndarray) -> np.ndarray:
#     min_value = np.float32(-1952.538701)
#     max_value = np.float32(602.8355304)
#     return data * (max_value - min_value) + min_value

# def normalize_abs_structure_press(data: np.ndarray) -> np.ndarray:
#     min_value = np.float32(-1906.293192) 
#     max_value = np.float32(633.319154)
#     return (data - min_value) / (max_value - min_value)

# def de_normalize_abs_structure_press(data: np.ndarray) -> np.ndarray:
#     min_value = np.float32(-1906.293192)
#     max_value = np.float32(633.319154)
#     return data * (max_value - min_value) + min_value

# def normalize_abs_structure_vel(data: np.ndarray) -> np.ndarray:
#     min_value = np.float32(-0.03131836869)
#     max_value = np.float32(2.079053836)
#     return (data - min_value) / (max_value - min_value)
# def de_normalize_abs_structure_vel(data: np.ndarray) -> np.ndarray:
#     min_value = np.float32(-0.03131836869)
#     max_value = np.float32(2.079053836)
#     return data * (max_value - min_value) + min_value


# 0.8m/s 유동장을 기준으로 정규화
# press 변화량 정규화
def normalize_diff_press(data: np.ndarray) -> np.ndarray:
    min_value = np.float32(-531.18798447)
    max_value = np.float32(285.4357259)
    return (data - min_value) / (max_value - min_value)

def de_normalize_diff_press(data: np.ndarray) -> np.ndarray:
    min_value = np.float32(-531.18798447)
    max_value = np.float32(285.4357259)
    return data * (max_value - min_value) + min_value

# vel 변화량 정규화
def normalize_diff_vel(data: np.ndarray) -> np.ndarray:
    min_value = np.float32(-0.8876490654)
    max_value = np.float32(0.9381583025)
    return (data - min_value) / (max_value - min_value)

def de_normalize_diff_vel(data: np.ndarray) -> np.ndarray:
    min_value = np.float32(-0.8876490654)
    max_value = np.float32(0.9381583025)
    return data * (max_value - min_value) + min_value




