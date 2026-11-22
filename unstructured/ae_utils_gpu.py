import os
import tensorflow as tf
import random
import numpy as np
import arguments_node_based
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) 
        print("✅ GPU 메모리 증가 방식 설정 완료")
    except RuntimeError as e:
        print(e)
# os.environ['PYTHONHASHSEED'] = '0'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
seed = arguments_node_based.DatasetParams.seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
from tensorflow.keras import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import ModelCheckpoint  # 추가
from typing import Tuple
import pandas as pd
from tqdm.keras import TqdmCallback

class MLAE(Model):
    """
    Multi-Layer Autoencoder (MLAE) implementation.

    This autoencoder consists of an encoder and a decoder. The encoder compresses
    the input into a latent representation of specified dimensionality, and the
    decoder reconstructs the input from the latent representation.

    Args:
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        encoder (tf.keras.Sequential): The encoder network.
        decoder (tf.keras.Sequential): The decoder network.
    """

    def __init__(self, latent_dim, **kwargs):
        super(MLAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        # Encoder: 5 layers with batch normalization
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(8192, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(4096, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(1024, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(512, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(latent_dim, activation="relu", kernel_initializer='he_normal'),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(512, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(1024, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(4096, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(8192, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(
                    arguments_node_based.DatasetParams.n_node,
                    activation="sigmoid")
            ]
        )

    def call(self, x):
        """
        Forward pass of the MLAE model.

        Encodes the input tensor into a latent representation and reconstructs
        it using the decoder.

        Args:
            x (tf.Tensor): Input tensor of shape `(batch_size, nx, ny)`.

        Returns:
            tf.Tensor: Reconstructed tensor of shape `(batch_size, nx, ny)`.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_config(self):
        config = super(MLAE, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config    

class MLAE_linear(Model):
    def __init__(self, latent_dim, **kwargs):
        super(MLAE_linear, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        # Encoder: 5 layers with batch normalization
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(8192, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(4096, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(1024, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(512, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(latent_dim, activation="relu", kernel_initializer='he_normal'),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(512, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(1024, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(4096, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(8192, kernel_initializer='he_normal'),
                layers.Activation('relu'),
                layers.Dense(
                    arguments_node_based.DatasetParams.n_node,
                    activation="linear")
            ]
        )

    def call(self, x):
        """
        Forward pass of the MLAE model.

        Encodes the input tensor into a latent representation and reconstructs
        it using the decoder.

        Args:
            x (tf.Tensor): Input tensor of shape `(batch_size, nx, ny)`.

        Returns:
            tf.Tensor: Reconstructed tensor of shape `(batch_size, nx, ny)`.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_config(self):
        config = super(MLAE_linear, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config  


'''class MLAE_HCSG(Model):
    def __init__(self, latent_dim, **kwargs):
        super(MLAE_HCSG, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        # Encoder: 5 layers with batch normalization
        self.encoder = tf.keras.Sequential(
            [
                layers.Dense(8192),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(4096),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(1024),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(512),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(512),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(1024),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(4096),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(8192),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Dense(arguments_node_based.DatasetParams.n_node),
                layers.Activation('sigmoid')
            ]
        )

    def call(self, x):
        """
        Forward pass of the MLAE model.

        Encodes the input tensor into a latent representation and reconstructs
        it using the decoder.

        Args:
            x (tf.Tensor): Input tensor of shape `(batch_size, nx, ny)`.

        Returns:
            tf.Tensor: Reconstructed tensor of shape `(batch_size, nx, ny)`.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_config(self):
        config = super(MLAE_HCSG, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config
'''
def get_compiled_model(optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4), 
                       loss: str = "mse") -> MLAE:
    """
    Creates and compiles an instance of the MLAE model.

    The function initializes the MLAE autoencoder with the latent dimension
    specified in `arguments.DatasetParams.latent_dim`, compiles it with the
    provided optimizer and loss function, and returns the compiled model.

    Args:
        optimizer (str | Optimizer, optional): The optimizer to use for model compilation.
            Default is "adam".
        loss (str, optional): The loss function to use for model compilation.
            Default is "mse" (mean squared error).

    Returns:
        MLAE: The compiled MLAE model.
    """
    autoencoder = MLAE(arguments_node_based.DatasetParams.latent_dim)
    # input_shape = (arguments_node_based.DatasetParams.n_node)
    autoencoder.build(input_shape=(None, arguments_node_based.DatasetParams.n_node))
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder


def train_model(
    x_y_train: tf.Tensor, x_y_test: tf.Tensor, save: bool = False
) -> Tuple[MLAE, tf.keras.callbacks.History]:
    """
    Trains the MLAE model using the provided training and testing datasets.

    The model is compiled using default parameters from `get_compiled_model`,
    and it is trained for the number of epochs and batch size specified in
    `arguments.AETrainingParams`.

    Args:
        x_y_train (tf.Tensor): Training data, where inputs and targets are the same.
        x_y_test (tf.Tensor): Testing data, where inputs and targets are the same.

    Returns:
        Tuple[MLAE_linear, tf.keras.callbacks.History]: A tuple containing:
            - The trained MLAE model.
            - The training history object.
    """
    epochs = arguments_node_based.AETrainingParams.epochs
    batch_size = arguments_node_based.AETrainingParams.batch_size
    model = get_compiled_model()
    save_path = arguments_node_based.AETrainingParams.save_path_ae
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ckpt_path = os.path.join(save_path, "best_checkpoints.keras")
    history = model.fit(
        x_y_train,
        x_y_train,
        batch_size=batch_size,
        shuffle=True,
        epochs=epochs,
        verbose=0,
        validation_data=(x_y_test, x_y_test),
        callbacks=[TqdmCallback(verbose=1), 
                   ModelCheckpoint(filepath=ckpt_path,
                                  monitor="val_loss", mode="min",
                                  save_best_only=True, save_weights_only=False, verbose=1)],
    )
    best_epoch = np.argmin(history.history['val_loss']) + 1  # +1은 에폭이 1부터 시작하므로
    best_val_loss = np.min(history.history['val_loss'])

    print(f"🏆 Best 모델 저장 에폭: {best_epoch}/{epochs}")
    print(f"🏆 Best val_loss: {best_val_loss:.6f}")
    print(f"💾 저장 경로: {ckpt_path}")

    loss_path = arguments_node_based.AETrainingParams.AE_loss_path
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(loss_path, index=False)


    if save:
        save_path = arguments_node_based.AETrainingParams.save_path_ae
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, "autoencoder.keras")
        model.save(filename)

    return model, history
