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
            tf.config.experimental.set_memory_growth(gpu, True) 
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
from tensorflow.keras import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.optimizers import Optimizer
from typing import Tuple
import pandas as pd
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import ModelCheckpoint  # 추가

class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.score = tf.keras.layers.Conv2D(1, 1, padding='same')  # 위치별 스칼라 점수

    def call(self, x):  # x: (B,H,W,C)
        s = self.score(x)                      # (B,H,W,1)
        s = tf.reshape(s, [tf.shape(x)[0], -1, 1])  # (B,HW,1)
        a = tf.nn.softmax(s, axis=1)           # (B,HW,1) → 가중치
        x_flat = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])  # (B,HW,C)
        z = tf.reduce_sum(a * x_flat, axis=1)  # (B,C)
        return z
        
class convAE(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim # 16*16 = 256
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),

            layers.Conv2D(16, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(32, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(128, (3,3), strides=(2,2), padding='same'), # 18,48, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(256, (3,3), strides=(2,2), padding='same'), # 9,24, 256
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(self.latent_dim, (1, 1), padding='same'), # 9,24,256
            layers.BatchNormalization(),
            layers.ReLU(),
            AttentionPooling(), # (256,) - 공간 평균으로 압축 ##  layers.GlobalAveragePooling2D() for abs
        ])

        self.decoder = tf.keras.Sequential([
            layers.Reshape((1, 1, self.latent_dim)), # (256,) -> (1,1,256)
            layers.Conv2DTranspose(256, (9, 24), padding='valid'), # (9,24,256)
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'), # 18,48, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(1, (3,3), strides=(2,2), activation='sigmoid', padding='same'), # 288,768, 1
            layers.Cropping2D(cropping=((3,3),(6,6)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config
    
class convAE_MaxPool(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE_MaxPool, self).__init__(**kwargs)
        self.latent_dim = latent_dim # 256
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),
            
            # Encoder with MaxPooling (이미지 구조 참고)
            layers.Conv2D(16, (3, 3), padding='same'),  # 288,768,16
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),  # 144,384,16
            
            layers.Conv2D(8, (3, 3), padding='same'),   # 144,384,8
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),  # 72,192,8
            
            layers.Conv2D(8, (3, 3), padding='same'),   # 72,192,8
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),  # 36,96,8
            
            layers.Conv2D(4, (3, 3), padding='same'),   # 36,96,4
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2)),  # 18,48,4
            
            layers.Conv2D(4, (3, 3), padding='same'),   # 18,48,4
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((2, 2), name="CODE"),  # 9,24,4 = 864
            
            layers.Conv2D(latent_dim, (1, 1), padding='same'),  # 9,24,256 (직접 변환)
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # (256,) - 공간 평균으로 압축
        ])

        self.decoder = tf.keras.Sequential([
            # 256에서 바로 공간 복원
            layers.Dense(9 * 24 * 4, activation='relu'),  # 256 -> 864 (원래 크기)
            layers.Reshape((9, 24, 4)),  # 인코더 마지막과 동일한 크기
            
            # 이제 4채널부터 시작
            layers.Conv2D(4, (3, 3), padding='same'),  # 9,24,4
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.UpSampling2D((2, 2)),  # 18,48,4
            
            layers.Conv2D(4, (3, 3), padding='same'),  # 18,48,4
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.UpSampling2D((2, 2)),  # 36,96,4

            layers.Conv2D(8, (3, 3), padding='same'),  # 36,96,8
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.UpSampling2D((2, 2)),  # 72,192,8

            layers.Conv2D(8, (3, 3), padding='same'),  # 72,192,8
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.UpSampling2D((2, 2)),  # 144,384,8

            layers.Conv2D(16, (3, 3), padding='same'),  # 144,384,16
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.UpSampling2D((2, 2)),  # 288,768,16

            layers.Conv2D(1, (3, 3), padding='same', name="OUTPUT"),  # 288,768,1
            layers.Cropping2D(cropping=((3,3),(6,6)))  # 282,756,1
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE_MaxPool, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config

class convAE_test(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE_test, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.ny = arguments_structure.DatasetParams.ny
        self.nx = arguments_structure.DatasetParams.nx
        self.input_shape = (self.ny, self.nx, 1)
        
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),
            
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),  # 288,768,16
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),  # 144,384,16
            
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),  # 144,384,16
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),  # 72,192,16
            
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),  # 72,192,8
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),  # 36,96,8
            
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),  # 36,96,8
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),  # 18,48,8
            
            layers.Conv2D(4, (3, 3), activation='relu', padding='same'),  # 18,48,4
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),  # 9,24,4

            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),  # 256
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(9 * 24 * 4, activation='relu'),  # 256 -> 864 (원래 크기)
            layers.Reshape((9, 24, 4)),  # 인코더 마지막과 동일한 크기
            
            layers.Conv2D(4, (3, 3), activation='relu', padding='same'),  # 9,24,4
            layers.UpSampling2D(pool_size=(2, 2), strides=2),  # 18,48,4
            
            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),  # 18,48,8
            layers.UpSampling2D(pool_size=(2, 2), strides=2),  # 36,96,8

            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),  # 36,96,8
            layers.UpSampling2D(pool_size=(2, 2), strides=2),  # 72,192,8

            layers.Conv2D(8, (3, 3), activation='relu', padding='same'),  # 72,192,8
            layers.UpSampling2D(pool_size=(2, 2), strides=2),  # 144,384,8

            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),  # 144,384,16
            layers.UpSampling2D(pool_size=(2, 2), strides=2),  # 288,768,16

            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),  # 288,768,1
            layers.Cropping2D(cropping=((3,3),(6,6)))  # 282,756,1
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  
    
    def get_config(self):
        config = super(convAE_test, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config

class convAE_v2(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE_v2, self).__init__(**kwargs)
        self.latent_dim = (9,24,256) # 9*24*256
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),

            layers.Conv2D(16, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(32, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(128, (3,3), strides=(2,2), padding='same'), # 18,48, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(256, (3,3), strides=(2,2), padding='same'), # 9,24, 256
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(256, (3, 3), padding='same'), # (9,24,256)
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'), # 18,48, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(1, (3,3), strides=(2,2), activation='sigmoid', padding='same'), # 288,768, 1
            layers.Cropping2D(cropping=((3,3),(6,6)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE_v2, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config

class convAE_v3(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE_v3, self).__init__(**kwargs)
        self.latent_dim = (9,24,256) # 9*24*256
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),

            layers.Conv2D(16, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(32, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(128, (3,3), strides=(2,2), padding='same'), # 18,48, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(256, (3,3), strides=(2,2), padding='same'), # 9,24, 256
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2D(256, (3,3), padding='same'), # (9,24,256)
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 18,48
            layers.Conv2D(128, (3,3), padding='same'),  # 18,48, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 36,96
            layers.Conv2D(64, (3,3), padding='same'),  # 36,96, 64
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 72,192
            layers.Conv2D(32, (3,3), padding='same'),  # 72,192, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 144,384
            layers.Conv2D(16, (3,3), padding='same'),  # 144,384, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 288,768
            layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'),  # 288,768, 1
            layers.Cropping2D(cropping=((3,3),(6,6)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE_v3, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config

class convAE_v4(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE_v4, self).__init__(**kwargs)
        self.latent_dim = (3,6,256) # 3*6*256
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),

            layers.Conv2D(4, (3,3), strides=(2,2), padding='same'), # 144,384, 4
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(8, (3,3), strides=(2,2), padding='same'), # 72,192, 8
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(16, (3,3), strides=(2,2), padding='same'), # 36,96, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(32, (3,3), strides=(2,2), padding='same'), # 18,48, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same'), # 9,24, 64
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(128, (3,3), strides=(1,1), padding='same'), # 9,24, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((3,2)), # 3,12, 128

            layers.Conv2D(256, (3,3), strides=(1,1), padding='same'), # 3,12, 256
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D((1,2)), # 3,6, 256
        ])

        self.decoder = tf.keras.Sequential([
            layers.UpSampling2D((1,2)), # 3,12, 256
            layers.Conv2D(128, (3,3), padding='same'), # 3,12, 128
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.UpSampling2D((3,2)), # 9,24, 128
            layers.Conv2D(64, (3,3), padding='same'), # (9,24,64)
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 18,48
            layers.Conv2D(32, (3,3), padding='same'),  # 18,48, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 36,96
            layers.Conv2D(16, (3,3), padding='same'),  # 36,96, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 72,192
            layers.Conv2D(8, (3,3), padding='same'),  # 72,192, 8
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 144,384
            layers.Conv2D(4, (3,3), padding='same'),  # 144,384, 4
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.UpSampling2D((2, 2)),  # 288,768
            layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'),  # 288,768, 1
            layers.BatchNormalization(),
            layers.Cropping2D(cropping=((3,3),(6,6)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE_v4, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config


class convAE_v5(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE_v5, self).__init__(**kwargs)
        self.latent_dim = 1024
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),

            layers.Conv2D(32, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(128, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(256, (3,3), strides=(2,2), padding='same'), # 18,48, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(512, (3,3), strides=(2,2), padding='same'), # 9,24, 256
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv2D(self.latent_dim, (1, 1), padding='same'), # 9,24,256
            layers.BatchNormalization(),
            layers.ReLU(),
            AttentionPooling(), # (1024,) - 공간 평균으로 압축 ##  layers.GlobalAveragePooling2D() for abs
        ])

        self.decoder = tf.keras.Sequential([
            layers.Reshape((1, 1, self.latent_dim)), # (1024,) -> (1,1,1024)
            layers.Conv2DTranspose(512, (9,24), strides=(2,2), padding='valid'), # (9,24,512)
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding='same'), # 18,48, 256
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'), # 36,96, 128
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'), # 72,192, 64
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same'), # 144,384, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(1, (3,3), strides=(2,2), activation='sigmoid', padding='same'), # 288,768, 1
            layers.Cropping2D(cropping=((3,3),(6,6)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE_v5, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config


# convAE의 디코더 마지막 층에 conv2D 레이어 삽입 : 체커보드 완화
class convAE_v6(Model): 
    def __init__(self, latent_dim, **kwargs):
        super(convAE_v6, self).__init__(**kwargs)
        self.latent_dim = 256 # 16*16 = 256
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),

            layers.Conv2D(16, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(32, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(128, (3,3), strides=(2,2), padding='same'), # 18,48, 128
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(256, (3,3), strides=(2,2), padding='same'), # 9,24, 256
            layers.BatchNormalization(),
            layers.Activation('gelu'), 

            layers.Conv2D(self.latent_dim, (1, 1), padding='same'), # 9,24,256
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.GlobalAveragePooling2D(), # (256,) - 공간 평균으로 압축 ##  layers.GlobalAveragePooling2D() for abs
        ])

        self.decoder = tf.keras.Sequential([
            layers.Reshape((1, 1, self.latent_dim)), # (256,) -> (1,1,256)

            layers.Conv2DTranspose(256, (9, 24), padding='valid'), # (3,6,256)
            layers.BatchNormalization(),
            layers.Activation('gelu'), 

            layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'), # 18,48, 128
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.Conv2D(32, (3,3), padding='same'), # CNN layer
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.Conv2D(16, (3,3), padding='same'), # CNN layer
            layers.BatchNormalization(),
            layers.Activation('gelu'), 

            layers.Conv2DTranspose(1, (3,3), strides=(2,2), padding='same'), # 288,768, 1
            layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'),
            layers.Cropping2D(cropping=((3,3),(6,6)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE_v6, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config

class convAE_v7(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE_v7, self).__init__(**kwargs)
        self.latent_dim = 256
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),

            layers.Conv2D(8, (3,3), strides=(2,2), padding='same'), # 144,384, 8
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(16, (3,3), strides=(2,2), padding='same'), # 72,192, 16
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(32, (3,3), strides=(2,2), padding='same'), # 36,96, 32
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same'), # 18,48, 64
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(128, (3,3), strides=(2,2), padding='same'), # 9,24, 128
            layers.BatchNormalization(),
            layers.Activation('gelu'), 

            layers.Conv2D(self.latent_dim, (1, 1), padding='same'), # 9,24,256
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.GlobalAveragePooling2D(), # (256,) - 공간 평균으로 압축 ##  layers.GlobalAveragePooling2D() for abs
        ])

        self.decoder = tf.keras.Sequential([
            layers.Reshape((1, 1, self.latent_dim)), # (256,) -> (1,1,256)
            
            layers.Conv2DTranspose(256, (9,24), padding='valid'), # (9,24,256)
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.Conv2D(256, (3,3), padding='same'), # CNN layer
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same'), # (18,48,128)
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.Conv2D(128, (3,3), padding='same'), # CNN layer
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'), # 36,96, 64
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.Conv2D(64, (3,3), padding='same'), # CNN layer
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same'), # 72,192, 32
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.Conv2D(32, (3,3), padding='same'), # CNN layer
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'), # 144,384, 16
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.Conv2D(16, (3,3), padding='same'), # CNN layer
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2DTranspose(8, (3,3), strides=(2,2), padding='same'), # 288,768, 8
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            layers.Conv2D(8, (3,3), padding='same'), # CNN layer
            layers.BatchNormalization(),
            layers.Activation('gelu'), 
            
            layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'), # 288,768, 1
            layers.Cropping2D(cropping=((3,3),(6,6)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE_v7, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config


class convAE_HCSG(Model):
    def __init__(self, latent_dim, **kwargs):
        super(convAE_HCSG, self).__init__(**kwargs)
        self.latent_dim = latent_dim # 인자 사용 (latent-dim 스윕용; 기존 main 은 256 전달 → 동작 동일)
        self.ny = arguments_structure.DatasetParams.ny # 282
        self.nx = arguments_structure.DatasetParams.nx # 756
        self.input_shape = (self.ny, self.nx, 1)   

        # input data shape: 282, 756 -> 288, 768 with padding ((3,3), (6,6))
        self.encoder = tf.keras.Sequential([
            layers.ZeroPadding2D(padding=((3,3),(6,6)), input_shape=self.input_shape),

            layers.Conv2D(4, (3,3), strides=(2,2), padding='same'), # 144,384, 4
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(8, (3,3), strides=(2,2), padding='same'), # 72,192, 8
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(16, (3,3), strides=(2,2), padding='same'), # 36,96, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(32, (3,3), strides=(2,2), padding='same'), # 18,48, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same'), # 9,24, 64
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Flatten(),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Dense(self.latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Dense(9 * 24 * 64),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((9, 24, 64)),  # 9,24, 64
            
            layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same'), # 18,48, 32
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(16, (3,3), strides=(2,2), padding='same'), # 36,96, 16
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(8, (3,3), strides=(2,2), padding='same'), # 72,192, 8
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(4, (3,3), strides=(2,2), padding='same'), # 144,384, 4
            layers.BatchNormalization(),
            layers.ReLU(),
            
            layers.Conv2DTranspose(1, (3,3), strides=(2,2), activation='sigmoid', padding='same'), # 288,768, 1
            layers.Cropping2D(cropping=((3,3),(6,6)))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  

    def get_config(self):
        config = super(convAE_HCSG, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
        })
        return config


def get_compiled_model(optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4), 
                       loss: str = "mse") -> convAE_HCSG:
    """
    Creates and compiles an instance of the convAE model.

    The function initializes the convAE autoencoder with the latent dimension
    specified in `arguments.DatasetParams.latent_dim`, compiles it with the
    provided optimizer and loss function, and returns the compiled model.

    Args:
        optimizer (str | Optimizer, optional): The optimizer to use for model compilation.
            Default is "adam".
        loss (str, optional): The loss function to use for model compilation.
            Default is "mse" (mean squared error).

    Returns:
        convAE: The compiled convAE model.
    """
    autoencoder = convAE_HCSG(arguments_structure.DatasetParams.latent_dim)
    autoencoder.build(input_shape=(None, arguments_structure.DatasetParams.ny, 
                                   arguments_structure.DatasetParams.nx, 1))
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return autoencoder


def train_model(
    x_y_train: tf.Tensor, x_y_test: tf.Tensor, save: bool = False
) -> Tuple[convAE_HCSG, tf.keras.callbacks.History]:
    """
    Trains the MLAE model using the provided training and testing datasets.

    The model is compiled using default parameters from `get_compiled_model`,
    and it is trained for the number of epochs and batch size specified in
    `arguments.AETrainingParams`.

    Args:
        x_y_train (tf.Tensor): Training data, where inputs and targets are the same.
        x_y_test (tf.Tensor): Testing data, where inputs and targets are the same.

    Returns:
        Tuple[convAE_HCSG, tf.keras.callbacks.History]: A tuple containing:
            - The trained convAE model.
            - The training history object.
    """
    epochs = arguments_structure.AETrainingParams.epochs
    batch_size = arguments_structure.AETrainingParams.batch_size
    model = get_compiled_model()
    save_path = arguments_structure.AETrainingParams.save_path_ae
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

    loss_path = arguments_structure.AETrainingParams.AE_loss_path
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(loss_path, index=False)


    if save:
        save_path = arguments_structure.AETrainingParams.save_path_ae
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, "autoencoder.keras")
        model.save(filename)

    return model, history


