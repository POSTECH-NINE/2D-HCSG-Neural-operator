import os
import tensorflow as tf
import random
import numpy as np
import arguments_structure
import soap
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import soap
seed = arguments_structure.DatasetParams.seed
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
    Add,
    Reshape,
    LayerNormalization,
)
from keras.saving import register_keras_serializable

@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_2DCNN(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_Model_2DCNN, self).__init__(**kwargs)

        # model parameters
        self.p = 16
        self.m = arguments_structure.DatasetParams.latent_dim
        self.Par = Par

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            staircase=False,
            decay_rate=0.99,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.branch_net_ls = self.build_branch_net()
        self.trunk_net_ls  = self.build_trunk_net()

    def build_branch_net(self):
        ld_sqrt = int(np.sqrt(arguments_structure.DatasetParams.latent_dim))
        layers = tf.keras.Sequential([
            Conv2D(
                32,
                (3, 3),
                name="conv1",
                input_shape=[ld_sqrt, ld_sqrt, self.Par["n_channels"]],),
            Activation(tf.math.sin),
            Conv2D(16, (3, 3), name="conv2"),  # [10,10,16]
            Activation(tf.math.sin),

            Conv2D(16, (3, 3), name="conv3"),  # [8,8,16]
            Activation(tf.math.sin),

            Conv2D(16, (3, 3), name="conv4"),  # [8,8,16]
            Activation(tf.math.sin),
            
            Flatten(),
            Dense(self.m * self.p),  # multi-scale output
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
        
        # X_func -> [BS*n_t, k*n_f]
        # X_loc  -> [n_t, 1]

        y_func = (X_func - self.Par["mean"]) / self.Par["std"]
        y_func = self.branch_net_ls(y_func, training=training)
        y_func = tf.reshape(y_func, [-1, self.m, self.p])

        y_loc0 = 10.0 * (X_loc - 0.5)  
        y_loc = self.trunk_net_ls(y_loc0, training=training)  # (batch, m*p)
        y_loc = tf.reshape(y_loc, [-1, self.m, self.p])

        Y = tf.einsum("ijk,pjk->ipj", y_func, y_loc)
        return Y
    
    @tf.function(jit_compile=True)
    def mse_loss(self, y_pred, y_train):

        # mse = MeanSquaredError()
        # Total Loss
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        # mse(y_pred, y_train)

        return [train_loss]
   
    def get_config(self):
        config = super(DeepONet_Model_2DCNN, self).get_config()
        config.update({
            "Par": self.Par,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        Par    = config.pop("Par")
        return cls(Par, **config)    



@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_MLP(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_Model_MLP, self).__init__(**kwargs)

        # model parameters
        self.p = 16
        self.m = arguments_structure.DatasetParams.latent_dim
        self.Par = Par

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=10000,
            staircase=False,
            decay_rate=0.99,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.branch_net_ls = self.build_branch_net()
        self.trunk_net_ls  = self.build_trunk_net()

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

            Dense(self.m * self.p),
        ]
        return tf.keras.Sequential(layers)

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc, training=None):
        
        # X_func -> [BS*n_t, k*n_f]
        # X_loc  -> [n_t, 1]

        y_func = (X_func - self.Par["mean"]) / self.Par["std"]
        y_func = self.branch_net_ls(y_func, training=training)
        y_func = tf.reshape(y_func, [-1, self.m, self.p])

        y_loc0 = 10.0 * (X_loc - 0.5)  
        y_loc = self.trunk_net_ls(y_loc0, training=training)  # (batch, m*p)
        y_loc = tf.reshape(y_loc, [-1, self.m, self.p])

        Y = tf.einsum("ijk,pjk->ipj", y_func, y_loc)
        return Y
    
    @tf.function(jit_compile=True)
    def mse_loss(self, y_pred, y_train):

        # mse = MeanSquaredError()
        # Total Loss
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        # mse(y_pred, y_train)

        return [train_loss]
   
    def get_config(self):
        config = super(DeepONet_Model_MLP, self).get_config()
        config.update({
            "Par": self.Par,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        Par    = config.pop("Par")
        return cls(Par, **config)    



@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_multiscale(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_Model_multiscale, self).__init__(**kwargs)
        self.p  = 16
        self.m  = arguments_structure.DatasetParams.latent_dim  # e.g., 256
        self.Par = Par
        self.nt = arguments_structure.DatasetParams.nt

        # multi-scale parameters
        self.scales = tf.constant([
                                1.0, 10, 20, 30, 40, 50, 100, 200, 300, 500], dtype=tf.float32)  # shape: (S,)
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
    def mse_loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        return [train_loss] 
    
    def get_config(self):
        config = super(DeepONet_Model_multiscale, self).get_config()
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

@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_multiscale_v3(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_Model_multiscale_v3, self).__init__(**kwargs)
        self.p  = 1
        self.m  = arguments_structure.DatasetParams.latent_dim  # e.g., 256
        self.input_shape = (9,24,256)
        self.Par = Par
        self.nt = arguments_structure.DatasetParams.nt

        # multi-scale parameters
        self.scales = tf.constant([
                                1.0, 10, 20, 30, 40, 50, 100, 200, 300, 500], dtype=tf.float32)  # shape: (S,)
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
        layers = tf.keras.Sequential([
            # 입력: (batch_size, 9, 24, 256)
            # 특징 추출을 위한 합성곱 층들 - 크기 유지
            Conv2D(256, (3, 3), padding='same', 
                   input_shape=(9, 24, 256)),
            Activation(tf.nn.relu),
            
            Conv2D(256, (3, 3), padding='same'),
            Activation(tf.nn.relu),
            
            # 다중 스케일 특징 추출
            Conv2D(256, (5, 5), padding='same'),
            Activation(tf.nn.relu),
            
            Conv2D(256, (1, 1), padding='same'),  # 1x1 conv for channel mixing
            Activation(tf.nn.relu),
            
            # 잔차 연결을 위한 추가 층
            Conv2D(256, (3, 3), padding='same'),
            Activation(tf.nn.relu),
            
            # 최종 출력: (batch_size, 9, 24, 256)
            Conv2D(2560, (3, 3), padding='same', activation='linear'),
            Reshape((9,24,256,self.S,self.p))
        ])
        return layers

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

            Dense(9*24*256*self.p)
        ]
        return tf.keras.Sequential(layers)

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc, training=None):
  
        # y_func = (X_func - self.Par["mean"]) / self.Par["std"]
        y_func = X_func
        y_func = self.branch_net_ls(y_func, training=training)
        y_func = tf.reshape(y_func, [-1, self.S, 9, 24, 256, self.p])

        y_loc0 = (X_loc - 0.5)  
        # y_loc0 = X_loc
        trunk_outs = []
        for s, subnet in enumerate(self.trunk_subnets):  # 독립적 사용
            beta = self.scales[s]
            x_scaled = beta * y_loc0    
            out_s = subnet(x_scaled, training=training)
            out_s = tf.reshape(out_s, [-1, 9, 24, 256, self.p])
            trunk_outs.append(out_s)

        y_loc = tf.stack(trunk_outs, axis=1)
        Y = tf.einsum("ijkqlm,pjkqlm->ipkql", y_func, y_loc)
        return Y  
    
    @tf.function(jit_compile=True)
    def mse_loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        return [train_loss] 
    
    def get_config(self):
        config = super(DeepONet_Model_multiscale_v3, self).get_config()
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


@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_multiscale_v4(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_Model_multiscale_v4, self).__init__(**kwargs)
        self.p  = 4
        self.m  = arguments_structure.DatasetParams.latent_dim  # e.g., 256
        self.input_shape = (3,6,256)
        self.Par = Par
        self.nt = arguments_structure.DatasetParams.nt

        # multi-scale parameters
        self.scales = tf.constant([
                                1.0, 10, 20, 30, 40, 50], dtype=tf.float32)  # shape: (S,)
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

        self.branch_net_ls = self.build_branch_net()
        self.trunk_subnets = [self.build_trunk_net() for i in range(self.S)]

    def build_branch_net(self):
        layers = [
            Flatten(input_shape=(3,6,256)),
            Dense(256, input_shape=(self.m,)),
            Activation(tf.nn.relu),
            Dense(256),
            Activation(tf.nn.relu),
            Dense(256),
            Activation(tf.nn.relu),
            Dense(256),
            Activation(tf.nn.relu),
            Dense(self.input_shape[0] * self.input_shape[1]* self.input_shape[2] * self.S * self.p)
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

            Dense(self.input_shape[0] * self.input_shape[1]* self.input_shape[2] * self.p)
        ]
        return tf.keras.Sequential(layers)

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc, training=None):
  
        # y_func = (X_func - self.Par["mean"]) / self.Par["std"]
        y_func = X_func
        y_func = self.branch_net_ls(y_func, training=training)
        y_func = tf.reshape(y_func, [-1, self.S, 3, 6, 256, self.p])

        y_loc0 = (X_loc - 0.5)  
        # y_loc0 = X_loc
        trunk_outs = []
        for s, subnet in enumerate(self.trunk_subnets):  # 독립적 사용
            beta = self.scales[s]
            x_scaled = beta * y_loc0    
            out_s = subnet(x_scaled, training=training)
            out_s = tf.reshape(out_s, [-1, 3, 6, 256, self.p])
            trunk_outs.append(out_s)

        y_loc = tf.stack(trunk_outs, axis=1)
        Y = tf.einsum("ijkqlm,pjkqlm->ipkql", y_func, y_loc)
        return Y  
    
    @tf.function(jit_compile=True)
    def mse_loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        return [train_loss] 
    
    def get_config(self):
        config = super(DeepONet_Model_multiscale_v4, self).get_config()
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
@register_keras_serializable(package="YourPackageName")
class DeepONet_Model_multiscale_v5(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_Model_multiscale_v5, self).__init__(**kwargs)
        self.p  = 16
        self.m  = 1024  # e.g., 256
        self.Par = Par
        self.nt = arguments_structure.DatasetParams.nt

        # multi-scale parameters
        self.scales = tf.constant([
                                1.0, 10, 20, 30, 40, 50, 100, 200, 300, 500], dtype=tf.float32)  # shape: (S,)
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
    def mse_loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        return [train_loss] 
    
    def get_config(self):
        config = super(DeepONet_Model_multiscale_v5, self).get_config()
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



@register_keras_serializable(package="YourPackageName")
class DeepONet_multiscale_HCSG(tf.keras.Model):
    def __init__(self, Par, **kwargs):
        super(DeepONet_multiscale_HCSG, self).__init__(**kwargs)
        self.p  = 16
        self.m  = 256  # e.g., 1024
        self.Par = Par
        self.nt = arguments_structure.DatasetParams.nt

        # multi-scale parameters
        self.scales = tf.constant([
                                1.0, 10, 20, 30, 40, 50, 100, 200, 300], dtype=tf.float32)  # shape: (S,)
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
    def mse_loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_pred - y_train))
        return [train_loss] 
    
    def get_config(self):
        config = super(DeepONet_multiscale_HCSG, self).get_config()
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
