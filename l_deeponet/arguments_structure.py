from dataclasses import dataclass

@dataclass
class DatasetParams:
    nt: int = 100  # How many timesteps from the dataset you want to keep
    n = 50  # All Data
    nx : int = 756
    ny : int = 282
    seed: int = 42
    dataset_path: str = (
        "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/scale_3x/diff_geo2_structured_press_train.npz" ## AE와 DeepONet 데이터 분리
    ) ############################ change ##########################
    num_train: int = 40
    latent_dim: int = 256 # original is 16 and 18, 20
    
@dataclass
class AETrainingParams:
    epochs = 5000
    batch_size = 64
    AE_loss_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/model_loss/CAE_2D_HCSG_pressure_loss"
    save_path_ae: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/CAE_2D_HCSG_pressure"

@dataclass
class DonTrainingParams: 
    early_stopping = False
    patience = 1000
    epochs = 5000
    batch_size = 8   
    n_logs = 100
    DON_loss_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/model_loss/CAE_LDON_2D_HCSG_pressure_loss_test" 

    load_AE_model_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/CAE_2D_HCSG_pressure/autoencoder.keras"

    save_path_don = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/CAE_LDON_2D_HCSG_pressure_test"
    save_pickle_path = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/params/params_CAE_LDON_2D_HCSG_pressure_test.pkl"

@dataclass
class PredictionParams:
    test_data_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/scale_3x/geo2_structured_press_test_ic040.npz"
    AE_model_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/CAE_2D_HCSG_pressure/autoencoder.keras"
    DON_model_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/CAE_LDON_2D_HCSG_pressure_test/deeponet.keras"

    load_pickle_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/params/params_CAE_LDON_2D_HCSG_pressure_test.pkl"
    prediction_save_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/test_inference_cae.npy"