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
l_deeponet/                           # multi-scale L-DeepONet with the MLP-based AE and the CAE
├── arguments_node_based.py           # configuration (unstructured mesh)
├── arguments_structure.py            # configuration (structured grid)
├── arguments_structure_vel.py        # configuration imported by utils.py
├── arguments_structure_press.py      # configuration imported by utils.py
├── utils.py                          # data loading and normalization
├── ae_utils_gpu.py                   # MLP-based autoencoder (class MLAE)
├── deeponet_utils_gpu.py             # multi-scale DeepONet for the MLP-based AE (class DeepONet_Model_multiscale_custom)
├── convAE.py                         # convolutional autoencoder (class convAE_HCSG)
├── deeponet_utils_gpu_structure.py   # multi-scale DeepONet for the CAE (class DeepONet_multiscale_HCSG)
├── soap.py                           # SOAP optimizer imported by deeponet_utils_gpu_structure.py
├── main_AE_gpu_diff.py               # training of the MLP-based AE
├── main_DON_gpu_diff.py              # training of the multi-scale L-DeepONet with the MLP-based AE
├── prediction_don_gpu_diff.py        # prediction with the MLP-based AE L-DeepONet
├── main_convAE.py                    # training of the CAE
├── main_DON_structure.py             # training of the multi-scale L-DeepONet with the CAE
└── prediction_don_structure.py       # prediction with the CAE L-DeepONet
fno/
├── fno_trainer.py                    # FNO (neuraloperator), its hyperparameters and training
├── dataset.py                        # dataset class for the FNO
├── utils.py                          # normalization utilities for the FNO
└── msFNO/
    ├── msfno.py                      # MscaleFNO (class MultiScaleFNO2d), its configuration and training
    └── predict.py                    # prediction with the MscaleFNO
requirements.txt
```

Other classes in these files are variants examined during development and are not used in the paper.
The file paths in the configuration files point to the authors' server. Change them to your local directories before use.

## Environment

The code was tested with Python 3.10.

```bash
pip install -r requirements.txt
```

## License

See `LICENSE`.
