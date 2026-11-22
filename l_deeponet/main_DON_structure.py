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
        print("✅ GPU 메모리 증가 방식 설정 완료")
    except RuntimeError as e:
        print(e)
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
seed = arguments_structure.DatasetParams.seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
import convAE
import utils
from tensorflow.keras import Model
from typing import Tuple
import time
import tqdm
from tensorflow.keras.layers import Input
import pandas as pd
import pickle
import deeponet_utils_gpu_structure


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


def get_deeponet_targets(ae_model: Model, outputs_norm: tf.Tensor, batch_size: int = 64) -> tf.Tensor:
    """
    학습 시와 동일한 배치 크기로 처리
    """
    total_samples = outputs_norm.shape[0]
    all_targets = []
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch_data = outputs_norm[i:end_idx]
        batch_targets = ae_model.encoder(batch_data)
        all_targets.append(batch_targets)
    
    return tf.concat(all_targets, axis=0)


def preprocess_2dcnn(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:    
    # 파라미터 설정
    n = np.shape(x)[0]
    ld = arguments_structure.DatasetParams.latent_dim
    nt = arguments_structure.DatasetParams.nt
    ld_sqrt = int(np.sqrt(ld))  # numpy sqrt로 통일
    
    # X_func: (n, ld_sqrt, ld_sqrt, 1)로 reshape
    X_func = np.reshape(x, [n, ld_sqrt, ld_sqrt, 1])
    
    # X_loc: 시간 좌표 생성
    X_loc = np.linspace(0.0, 1.0, nt).reshape(nt, 1)  # 한 줄로 간단히
    
    # y: (n, nt, ld)로 reshape
    y = np.reshape(y, [n, nt, ld])
    
    return X_func, X_loc, y

def preprocess_mlp(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:    
    # 파라미터 설정
    n = np.shape(x)[0]
    ld = arguments_structure.DatasetParams.latent_dim # 1024
    nt = arguments_structure.DatasetParams.nt
    
    # X_func: (n, ld)로 reshape
    X_func = np.reshape(x, [n, ld])
    
    # X_loc: 시간 좌표 생성
    X_loc = np.linspace(0.0, 1.0, nt).reshape(nt, 1)  # 한 줄로 간단히
    
    # y: (n, nt, ld)로 reshape
    y = np.reshape(y, [n, nt, ld])
    
    return X_func, X_loc, y

def preprocess_mlp_diff(x: np.ndarray, y: np.ndarray) -> Tuple:
    n = tf.shape(x)[0]
    ld = arguments_structure.DatasetParams.latent_dim
    X_func = tf.reshape(x, [n, ld])
    X_loc = tf.linspace(0, 1, arguments_structure.DatasetParams.nt-1)
    X_loc = tf.reshape(X_loc, [arguments_structure.DatasetParams.nt-1, 1])
    y = tf.reshape(y, [-1, arguments_structure.DatasetParams.nt-1, arguments_structure.DatasetParams.latent_dim])
    return X_func, X_loc, y

def preprocess_3dcnn(x: np.ndarray, y: np.ndarray):
    n = x.shape[0]
    ld = (3,6,256)
    nt = arguments_structure.DatasetParams.nt

    # X_func: (n, 9,24,256)
    X_func = np.reshape(x, (n, *ld), order='C')

    # X_loc: (nt,1)
    X_loc = np.linspace(0.0, 1.0, nt).reshape(nt, 1)

    # y: (n, nt, 9,24,256)
    Y = np.reshape(y, (n, nt, *ld), order='C')

    return X_func, X_loc, Y

@tf.function(jit_compile=True)
def train(don_model, X_func, X_loc, y):
    with tf.GradientTape() as tape:
        y_hat = don_model(X_func, X_loc, training=True)
        loss = don_model.mse_loss(y_hat, y)

    gradients = tape.gradient(loss, don_model.trainable_variables)
    # gradient clipping
    don_model.optimizer.apply_gradients(zip(gradients, don_model.trainable_variables))
    return loss

def split_dataset_5(X_func: tf.Tensor, y: tf.Tensor, 
                  val_indices=None) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    
    # 총 샘플 수
    num_samples = tf.shape(X_func)[0]
    idx = tf.range(num_samples)

    # validation index 결정
    if val_indices is None:
        # 특정 속도값에 해당하는 인덱스: 0.10, 0.20, 0.30, 0.50, 0.60 m/s
        # 0.10(idx=0), 0.20(idx=10), 0.30(idx=20), 0.50(idx=39), 0.60(idx=49)
        val_indices = tf.constant([0, 10, 20, 39, 49], dtype=tf.int32)
    else:
        val_indices = tf.convert_to_tensor(val_indices, dtype=tf.int32)

    # mask 계산
    # idx[:,None] 와 val_indices[None,:] 비교 → (n, k) → any(axis=1)
    val_mask = tf.reduce_any(
        tf.equal(tf.expand_dims(idx, 1), tf.expand_dims(val_indices, 0)),
        axis=1
    )
    train_mask = tf.logical_not(val_mask)

    # 인덱스 리스트 생성
    train_indices = tf.boolean_mask(idx, train_mask)
    test_indices  = tf.boolean_mask(idx, val_mask)

    # gather로 분할
    X_func_train = tf.gather(X_func, train_indices)
    y_train      = tf.gather(y,      train_indices)
    X_func_test  = tf.gather(X_func, test_indices)
    y_test       = tf.gather(y,      test_indices)

    return (X_func_train, y_train), (X_func_test, y_test)

def create_params(X_func_train: tf.Tensor) -> dict:

    Par = {}
    Par["n_channels"] = int(X_func_train.shape[-1])
    mean_tf = tf.reduce_mean(X_func_train)        # tf.Tensor
    std_tf  = tf.math.reduce_std(X_func_train)    # tf.Tensor
    Par["mean"] = float(mean_tf.numpy())
    Par["std"]  = float(std_tf.numpy())

    return Par

def train_model(
    n_epochs, X_func_train, X_loc, y_train,
    X_func_test, y_test, don_model, save=True,
    use_early_stopping=arguments_structure.DonTrainingParams.early_stopping,  # Early Stopping 사용 여부
    patience=arguments_structure.DonTrainingParams.patience,  # Early Stopping 사용 시 patience 값
    save_checkpoints=True  # 체크포인트 저장 여부 추가
):
    train_losses = []
    val_losses   = []
    r2_scores    = []
    bs = arguments_structure.DonTrainingParams.batch_size

    y_test_mean = np.mean(y_test, axis=0)
    ss_tot = np.sum(np.square(y_test - y_test_mean))
    
    # Early stopping variables (Early Stopping 사용 시에만 의미 있음)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None
    best_epoch = 0  # 최고 성능 에폭 기록

    # Checkpoint saving setup (체크포인트 설정 추가)
    checkpoint_path = None
    if save_checkpoints and save:
        save_path = arguments_structure.DonTrainingParams.save_path_don
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        checkpoint_path = os.path.join(save_path, "best_deeponet_checkpoint.keras")

    for epoch in tqdm.trange(1, n_epochs + 1, desc="Training", ncols=100):
        # 1) 배치별 손실을 모아 에폭 평균 학습 손실 계산
        batch_losses = [
            train(don_model,
                  X_func_train[i : i + bs],
                  X_loc,
                  y_train[i : i + bs])[0]
            for i in range(0, len(X_func_train), bs)
        ]
        train_losses.append(float(np.mean(batch_losses)))

        # 2) 에폭 끝에서 검증 손실 한 번 계산
        y_hat = don_model(X_func_test, X_loc, training=False)
        current_val_loss = float(don_model.mse_loss(y_hat, y_test)[0])
        val_losses.append(current_val_loss)
        
        # 3) R² 결정계수 계산
        ss_res = np.sum(np.square(y_test - y_hat.numpy()))
        r2_score = 1 - (ss_res / ss_tot)
        r2_scores.append(r2_score)

        # Model improvement check (모델 개선 확인 및 체크포인트 저장)
        improved = False
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_epoch = epoch
            improved = True
            
            if use_early_stopping:
                patience_counter = 0
                # Save the best model weights
                best_model_weights = don_model.get_weights()
            
            # Save checkpoint (best model) - 체크포인트 저장
            if save_checkpoints and checkpoint_path:
                don_model.save(checkpoint_path)
                
        else:
            # Early stopping 사용시에만 patience 카운터 증가
            if use_early_stopping:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    # Restore best weights
                    don_model.set_weights(best_model_weights)
                    break

        # 4) 로그 출력
        if epoch % arguments_structure.DonTrainingParams.n_logs == 0:
            log_message = f"Epoch {epoch}/{n_epochs} — Train: {train_losses[-1]:.3e} | Val: {val_losses[-1]:.3e}"
            if use_early_stopping:
                log_message += f" | Best Val: {best_val_loss:.3e}"
            if improved:
                log_message += " | Best Model"
            tqdm.tqdm.write(log_message)

    # 5) 결과 저장 및 리턴
    loss_df = pd.DataFrame({
        "loss":  train_losses,
        "val_loss":    val_losses,
        "R2_score": r2_scores,
    })
    loss_path = arguments_structure.DonTrainingParams.DON_loss_path
    loss_df.to_csv(loss_path, index=False)
    
    # 학습 완료 요약 출력
    final_epoch = len(train_losses)
    final_val_loss = val_losses[-1]
    early_stopped = final_epoch < n_epochs
    
    print(f"Total epochs: {final_epoch}/{n_epochs}")
    if early_stopped:
        print(f"Early stopping triggered")
    print(f"Final validation loss: {final_val_loss:.12f}")
    print(f"Best validation loss: {best_val_loss:.12f} (Epoch {best_epoch})")
    
    if save:
        print(f"Saving final model")
        save_path = arguments_structure.DonTrainingParams.save_path_don
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, "deeponet.keras")
        don_model.save(filename)
        print(f"Final model saved: {filename}")
        
        if save_checkpoints and checkpoint_path:
            print(f"Best model checkpoint saved: {checkpoint_path}")
            
    return don_model

if __name__ == "__main__":
    inputs, outputs = utils.load_data_structure_diff() 
    inputs_norm = utils.normalize_diff_press(inputs)
    outputs_norm = utils.normalize_diff_press(outputs)
    print(f"Inputs shape: {inputs_norm.shape}")
    print(f"Outputs shape: {outputs_norm.shape}")
    
    inputs_norm = np.expand_dims(inputs_norm, axis=-1)
    outputs_norm = np.expand_dims(outputs_norm, axis=-1)
    inputs_norm = np.swapaxes(inputs_norm, 1, 2)
    outputs_norm = np.swapaxes(outputs_norm, 1, 2)
    print(f"Inputs shape for AE: {inputs_norm.shape}")
    print(f"Outputs shape for AE: {outputs_norm.shape}")

    ae_model_path = arguments_structure.DonTrainingParams.load_AE_model_path
    ae_model = load_ae_weights(ae_model_path)    
    x_red = get_deeponet_targets(ae_model, inputs_norm)
    targets = get_deeponet_targets(ae_model, outputs_norm, batch_size=64)
    print(f"Reduced data shape: {x_red.shape}")
    print(f"Targets shape: {targets.shape}")

    X_func, X_loc, y = preprocess_mlp_diff(x_red, targets) ### change preprocess function to 1d ###
    (X_func_train, y_train), (X_func_test, y_test) = split_dataset_5(X_func, y)
    print(f"Training data shapes: {X_func_train.shape},{X_loc.shape}, {y_train.shape}")
    print(f"Testing data shapes: {X_func_test.shape},{X_loc.shape}, {y_test.shape}")
    print(create_params(X_func_train))
    params = create_params(X_func_train)
   
    with open(arguments_structure.DonTrainingParams.save_pickle_path, "wb") as f: ######################### change!!!!!!!!!!!!!!!!!!
        pickle.dump(params, f)
    don_model = deeponet_utils_gpu_structure.DeepONet_multiscale_HCSG(params) ### change don model to 1d ###
    trained_model = train_model(
        arguments_structure.DonTrainingParams.epochs,
        X_func_train,
        X_loc,
        y_train,
        X_func_test,
        y_test,
        don_model,
        use_early_stopping=arguments_structure.DonTrainingParams.early_stopping,  # Early Stopping 사용 여부 설정
        patience=arguments_structure.DonTrainingParams.patience,  # Early Stopping 사용 시 patience 값 설정
        save_checkpoints=True  # 체크포인트 저장 활성화
    )
    print(f"Trained model: Status: Trained")
