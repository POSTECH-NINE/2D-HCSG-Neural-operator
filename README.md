# Neural operator-based surrogate model for CFD: Helical coil steam generator in small modular reactors

Four neural operator surrogates predict the transient velocity and pressure fields of the two-dimensional cross-flow in the helical coil steam generator (HCSG) tube bundle of SMART.

| Model | Data type | Framework |
|---|---|---|
| Multi-scale L-DeepONet with MLP-based autoencoder | Unstructured mesh (28,877 nodes) | TensorFlow |
| Multi-scale L-DeepONet with convolutional autoencoder (CAE) | Structured grid (756 × 282) | TensorFlow |
| FNO | Structured grid (756 × 282) | PyTorch |
| MscaleFNO | Structured grid (756 × 282) | PyTorch |

## Repository structure

```
unstructured/
├── arguments_node_based.py           # configuration
├── ae_utils_gpu.py                   # MLP-based autoencoder (class MLAE)
└── deeponet_utils_gpu.py             # multi-scale DeepONet (class DeepONet_Model_multiscale_custom)
structured/
├── arguments_structure.py            # configuration
├── convAE.py                         # convolutional autoencoder (class convAE_HCSG)
├── deeponet_utils_gpu_structure.py   # multi-scale DeepONet (class DeepONet_multiscale_HCSG)
└── soap.py                           # SOAP optimizer imported by deeponet_utils_gpu_structure.py
fno/
├── fno_trainer.py                    # FNO (neuraloperator) and its hyperparameters
└── msFNO/msfno.py                    # MscaleFNO (class MultiScaleFNO2d) and its configuration
requirements.txt
```

Other classes in these files are variants examined during development and are not used in the paper.
The file paths in the configuration files point to the authors' server. Change them to your local directories before use.

## Environment

The code was tested with Python 3.10.

```bash
pip install -r requirements.txt
```

## Data

The preprocessed datasets used to train and test the surrogate models are available at: (Google Drive link)

```
data/
├── training_data_don_fluent/   # unstructured mesh (MLP-AE L-DeepONet)
└── scale_3x/                   # structured grid obtained by IDW interpolation (CAE L-DeepONet, FNO, MscaleFNO)
```

Each `.npz` file contains the following arrays.

| Key | Unstructured | Structured | Description |
|---|---|---|---|
| `inputs` | (N, 28877) | (N, 756, 282) | initial field u0 |
| `outputs` | (N, 100, 28877) | (N, 100, 756, 282) | temporal difference field at 100 time steps |
| `x_coords`, `y_coords` | (28877,), (28877,) | (756,), (282,) | node or grid coordinates |
| `nan_mask` | – | (756, 282) | grid points inside the tubes |

- `press` and `vel` denote the pressure and the velocity magnitude.
- `outputs` stores the temporal difference field Δu(x, t) = u(x, t) − u0(x). The full field is recovered by adding `inputs` (u0) to `outputs`.
- `ic010_to_060_except040` contains 50 inlet velocities from 0.10 to 0.60 m/s except 0.40 m/s (training and validation).
- `ic040`, `ic0455`, `ic070`, `ic075`, and `ic080` are the test inlet velocities of 0.40 and 0.455 m/s (interpolation) and 0.70, 0.75, and 0.80 m/s (extrapolation).

## License

See `LICENSE`.
