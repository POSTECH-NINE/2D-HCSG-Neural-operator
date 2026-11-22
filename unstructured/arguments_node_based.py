from dataclasses import dataclass

@dataclass
class DatasetParams:
    nt: int = 100  # How many timesteps from the dataset you want to keep
    n = 50  # All Data
    n_node = 28877 # geo1 : 28212, geo2 : 28875
    seed: int = 42
    dataset_path: str = (
        "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/training_data_don_fluent/geo2_vsmr_press_ic010_to_060_except040_diff.npz" ## AE와 DeepONet 데이터 분리
    ) ############################ change ##########################
    num_train: int = 40
    latent_dim: int = 256 # original is 16 and 18, 20
    
@dataclass
class AETrainingParams:
    epochs = 10000
    batch_size = 64
    AE_loss_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/model_loss/AE_2D_HCSG_pressure_loss"
    save_path_ae: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/AE_2D_HCSG_pressure"

@dataclass
class DonTrainingParams: 
    early_stopping = False
    patience = 1000
    epochs = 5000
    batch_size = 4   
    n_logs = 100
    DON_loss_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/model_loss/AE_LDON_2D_HCSG_pressure" 

    load_AE_model_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/AE_2D_HCSG_pressure/autoencoder.keras"

    save_path_don = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/AE_LDON_2D_HCSG_pressure"
    save_pickle_path = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/params/params_AE_LDON_2D_HCSG_pressure.pkl"

@dataclass
class PredictionParams:
    test_data_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/training_data_don_fluent/geo2_vsmr_press_ic040_diff.npz"
    AE_model_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/AE_2D_HCSG_pressure/autoencoder.keras"
    DON_model_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/results/AE_LDON_2D_HCSG_pressure/deeponet.keras"

    load_pickle_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/params/params_AE_LDON_2D_HCSG_pressure.pkl"
    prediction_save_path: str = "/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/Prediction_AE_LDON_2D_HCSG_pressure_inlet040.npy"
