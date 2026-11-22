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
from tensorflow.keras.layers import (
    Conv2D,
    Conv1D,
    Activation,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    Add
)
from tensorflow.keras import regularizers
from keras.saving import register_keras_serializable
from tf_siren import SinusodialRepresentationDense

def siren_uniform(in_dim, w0=30.0, first_layer=False):
    if first_layer:              # y = sin(w0 · (Wx+b)) → W범위 = 1 / n_in
        limit = 1. / in_dim
    else:                        # 깊은 층:  √6 / (n_in·w0)
        limit = np.sqrt(6. / in_dim) / w0
    return tf.keras.initializers.RandomUniform(-limit, limit)



@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_2DCNN(tf.keras.Model):
    def __init__(self, Par, scales=(1.0, 2.0, 3.0, 4.0, 5.0, 10, 20, 25, 30, 50), **kwargs):
        super(DeepONet_Model_2DCNN, self).__init__(**kwargs)

        # model parameters
        self.p = 16
        self.m = arguments_node_based.DatasetParams.latent_dim
        self.Par = Par

        # multi-scale parameters
        self.scales = tf.constant(scales, dtype=tf.float32)  # shape: (S,)
        self.S = len(scales)

        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list  = []

        # learning rate & optimizer
        # self.lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        #     initial_learning_rate=1e-3,
        #     first_decay_steps=2000,
        #     t_mul=2.0,
        #     m_mul=0.95,
        #     alpha=1e-6)
        # self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=10 ** -5,
        #     decay_steps=10000,
        #     staircase=False,
        #     decay_rate=0.99,
        # )
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
        self.branch_net_ls = self.build_branch_net()
        self.trunk_subnet  = self.build_trunk_net()
        self.alpha = tf.Variable(1.0, trainable=True)

    def build_branch_net(self):
        ld_sqrt = int(np.sqrt(arguments_node_based.DatasetParams.latent_dim))
        layers = tf.keras.Sequential([
            Conv2D(
                32,
                (3, 3),
                name="conv1",
                input_shape=[ld_sqrt, ld_sqrt, self.Par["n_channels"]],),
            Activation(tf.nn.relu),
            Conv2D(16, (3, 3), name="conv2"),  # [10,10,16]
            Activation(tf.nn.relu),

            Conv2D(16, (3, 3), name="conv3"),  # [8,8,16]
            Activation(tf.nn.relu),

            Conv2D(16, (3, 3), name="conv4"),  # [8,8,16]
            Activation(tf.nn.relu),
            
            Flatten(),
            Dense(self.m * self.p * self.S),  # multi-scale output
        ])
        return layers

    def build_trunk_net(self):
        layers = [
            Dense(100, input_shape=(1,)),
            Activation(tf.math.sin),

            Dense(100),
            Activation(tf.math.sin),

            Dense(100),
            Activation(tf.math.sin),

            Dense(self.m * self.p),
        ]
        return tf.keras.Sequential(layers)

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc, training=None):
        y_func = (X_func - self.Par["mean"]) / self.Par["std"]
        y_func = self.branch_net_ls(y_func, training=training)
        y_func = tf.reshape(y_func, [-1, self.S, self.m, self.p])

        y_loc0 = 10.0 * (X_loc - 0.5)  

        trunk_outs = []
        for s in range(self.S):
            beta = self.scales[s]                     # 스케일 β_s
            x_scaled = beta * y_loc0                  # (batch, 1) → (batch, 1)
            out_s   = self.trunk_subnet(x_scaled, training=training)  # (batch, m*p)
            out_s = tf.reshape(out_s, [-1, self.m, self.p])
            trunk_outs.append(out_s)

        y_loc = tf.stack(trunk_outs, axis=1)
        Y = tf.einsum("ijkl,pjkl->ipk", y_func, y_loc)
        return Y
    
    @tf.function(jit_compile=True)
    def custum_loss(self, y_pred, y_train):

        # mse = MeanSquaredError()
        # Total Loss
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        # mse(y_pred, y_train)

        return [train_loss]
   
    def get_config(self):
        config = super(DeepONet_Model_2DCNN, self).get_config()
        config.update({
            "Par": self.Par,
            "scales": tuple(self.scales.numpy()),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        Par    = config.pop("Par")
        scales = config.pop("scales")
        return cls(Par, scales=scales, **config)    

@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_MLP(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_Model_MLP, self).__init__(**kwargs)
        self.p = 16 # 16
        self.m = arguments_node_based.DatasetParams.latent_dim  # e.g., 256
        self.Par = Par
        self.nt = arguments_node_based.DatasetParams.nt
        self.exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,  
            decay_steps=10000,            
            staircase=False,
            decay_rate=0.99)   
        self.cosine_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3,
            first_decay_steps=100000,
            t_mul=2.0,
            m_mul=0.95,
            alpha=1e-6)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.exp_lr)
        
        self.branch_net_ls = self.build_branch_net()
        self.trunk_net_ls = self.build_trunk_net()

    def build_branch_net(self):
        layers = [
            Dense(64, input_shape=(self.m,)),
            Activation(tf.nn.relu),

            Dense(64),
            Activation(tf.nn.relu),

            Dense(64),
            Activation(tf.nn.relu),

            Dense(64),
            Activation(tf.nn.relu),

            Dense(64),
            Activation(tf.nn.relu),

            Dense(64),
            Activation(tf.nn.relu),

            Dense(self.m * self.p)
        ]
        return tf.keras.Sequential(layers)

    def build_trunk_net(self):
        layers = [
            Dense(100, input_shape=(1,)),
            Activation(tf.math.sin),

            Dense(100),
            Activation(tf.math.sin),

            Dense(100),
            Activation(tf.math.sin),

            Dense(self.m* self.p,),
        ]
        return tf.keras.Sequential(layers)

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc, training=None):
        # y_func = (X_func - self.Par["mean"]) / self.Par["std"]
        y_func = X_func
        y_func = self.branch_net_ls(y_func, training=training)
        
        y_loc = (X_loc - 0.5) # 30.0
        y_loc = self.trunk_net_ls(y_loc, training=training)
        
        y_func = tf.reshape(y_func, [-1, self.m, self.p])
        y_loc = tf.reshape(y_loc, [-1, self.m, self.p])
        Y = tf.einsum("ijk,pjk->ipj", y_func, y_loc)
        return Y

    @tf.function(jit_compile=True)
    def mse_loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        return [train_loss]
    
    def get_config(self):
        config = super(DeepONet_Model_MLP, self).get_config()
        config.update({
            "Par": self.Par,
            "p": self.p,
            "m": self.m,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        Par = config.pop("Par")
        config.pop("p", None)
        config.pop("m", None)
        return cls(Par, **config)  

@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_multiscale_custom(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_Model_multiscale_custom, self).__init__(**kwargs)
        self.p  = 16
        self.m  = arguments_node_based.DatasetParams.latent_dim  # e.g., 256
        self.Par = Par
        self.nt = arguments_node_based.DatasetParams.nt

        # multi-scale parameters
        self.scales = tf.constant([
                                1.0, 10, 20, 30, 40, 50, 100, 200, 300,500], dtype=tf.float32)  # shape: (S,)
        self.S = len(self.scales)
        self.w0 = 1
        
        self.exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,  
            decay_steps=10000,            
            staircase=False,
            decay_rate=0.99)   
        self.cosine_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3,
            first_decay_steps=100000,
            t_mul=2.0,
            m_mul=0.95,
            alpha=1e-6)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.exp_lr)

        # Branch Net output : (m * p * S)
        self.branch_net_ls = self.build_branch_net()
        # one trunk network
        # self.trunk_subnet = self.build_trunk_net()  

        # independent trunk networks
        self.trunk_subnets = [self.build_trunk_net() for i in range(self.S)]


    def build_branch_net(self):
        layers = [
            Dense(64, input_shape=(self.m,)),
            Activation(tf.nn.relu),
            # Dropout(0.1),
            Dense(64),
            Activation(tf.nn.relu),
            # Dropout(0.2),
            Dense(64),
            Activation(tf.nn.relu),
            # Dropout(0.3),
            Dense(64),
            Activation(tf.nn.relu),
            # Dropout(0.3),
            Dense(64),
            Activation(tf.nn.relu),
            # Dropout(0.2),
            Dense(64),
            Activation(tf.nn.relu),

            Dense(self.m * self.p * self.S)
        ]
        return tf.keras.Sequential(layers)

    def build_trunk_net(self):
        layers = [
            Dense(100, input_shape=(1,)),
            Activation(tf.math.sin),
            # Dropout(0.1),
            Dense(100),
            Activation(tf.math.sin),
            # Dropout(0.1),
            Dense(100),
            Activation(tf.math.sin),

            Dense(self.m * self.p)
        ]
        return tf.keras.Sequential(layers)

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc, training=None):
  
        # y_func = (X_func - self.Par["mean"]) / self.Par["std"]
        y_func = X_func
        y_func = self.branch_net_ls(y_func, training=training)
        y_func = tf.reshape(y_func, [-1, self.S, self.m, self.p])

        y_loc0 = (X_loc - 0.5)  
        # y_loc0 = X_loc
        trunk_outs = []
        for s, subnet in enumerate(self.trunk_subnets):  # 독립적 사용
            beta = self.scales[s]
            x_scaled = beta * y_loc0    
            out_s = subnet(x_scaled, training=training)
            out_s = tf.reshape(out_s, [-1, self.m, self.p])
            trunk_outs.append(out_s)

        y_loc = tf.stack(trunk_outs, axis=1)
        Y = tf.einsum("ijkl,pjkl->ipk", y_func, y_loc)
        return Y  

    @tf.function(jit_compile=True)
    def mae_loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.abs(y_pred - y_train))
        return [train_loss]
    
    @tf.function(jit_compile=True)
    def mse_loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        return [train_loss]


    @tf.function(jit_compile=True)
    def frequency_loss(self, y_pred, y_train):
        y_pred = tf.transpose(y_pred, perm=[0, 2, 1])
        y_train = tf.transpose(y_train, perm=[0, 2, 1])
        pred_fft = tf.signal.rfft(y_pred)
        true_fft = tf.signal.rfft(y_train)
        
        freq_loss = tf.reduce_mean(tf.square((tf.abs(pred_fft - true_fft))))
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        
        return [0.8 * mse_loss + 0.2 * freq_loss]

    @tf.function(jit_compile=True)
    def gradient_hybrid_loss(self, y_pred, y_train):
        # 기본 MAE 손실
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        
        # 공간 축(axis=2)에 대한 경사도 계산
        pred_grads = tf.experimental.numpy.gradient(y_pred, axis=2)
        true_grads = tf.experimental.numpy.gradient(y_train, axis=2)
        
        # 경사도에 대한 MAE 손실 계산
        grad_loss = tf.reduce_mean(tf.square(pred_grads - true_grads))

        # 두 손실을 결합 (가중치는 튜닝 대상)
        return 0.8 * mse_loss + 0.2 * grad_loss    
    
    def get_config(self):
        config = super(DeepONet_Model_multiscale_custom, self).get_config()
        config.update({
            "Par": self.Par,
            "scales": tuple(self.scales.numpy()),
        })
        return config

    @classmethod
    def from_config(cls, config):
        Par    = config.pop("Par")
        scales = config.pop("scales")
        return cls(Par, **config)

