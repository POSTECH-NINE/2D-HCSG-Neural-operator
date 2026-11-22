"""
MscaleFNO Prediction Script
============================
학습된 MscaleFNO 모델을 로드하여 예측 수행 (pressure + velocity)

사용법:
  python predict_mscalefno_2d_flow.py

흐름:
  1. 학습된 모델 로드
  2. 테스트 데이터 로드 & 정규화
  3. 예측 수행
  4. 역정규화
  5. 예측값 + 초기조건 = 최종 결과 저장
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

# ================================================================
# 설정
# ================================================================
BASE_DIR = '/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/fno_hcsg'
DATA_DIR = f'{BASE_DIR}/data'

PREDICT_CONFIG = {
    'variables': {
        'velocity': {
            'model_path': f'{BASE_DIR}/msFNO/paper_review_msfno/model/best_model_vel_24_reviewer1.pth',
            'test_data_paths': {
                '0.4': f'{DATA_DIR}/geo2_structured_vel_test_ic040.npz',
                '0.7': f'{DATA_DIR}/geo2_structured_vel_test_ic070.npz',
            },
            'label_min': 0.0,
            'label_max': 1.976950764656067,
        },
        'pressure': {
            'model_path': f'{BASE_DIR}/msFNO/paper_review_msfno/model/best_model_press_24_reviewer1.pth',
            'test_data_paths': {
                '0.4': f'{DATA_DIR}/geo2_structured_press_test_ic040.npz',
                '0.7': f'{DATA_DIR}/geo2_structured_press_test_ic070.npz',
            },
            'label_min': -1861.3638916015625,
            'label_max': 588.5082397460938,
        },
    },
    'save_dir': f'{BASE_DIR}/msFNO/paper_review_msfno/prediction',
}


# ================================================================
# 모델 정의 (학습 코드와 동일)
# ================================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, n_layers=4, out_channels=99, include_grid=False):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.include_grid = include_grid

        in_channels = 3 if include_grid else 1
        self.p = nn.Linear(in_channels, self.width)

        self.convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(n_layers)
        ])
        self.mlps = nn.ModuleList([
            MLP2d(self.width, self.width, self.width)
            for _ in range(n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(n_layers)
        ])
        self.q = MLP2d(self.width, out_channels, self.width * 4)

    def forward(self, x):
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = torch.sin(x)
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x


class MultiScaleFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_subnets=6,
                 n_layers=4, out_channels=99, include_grid=False):
        super().__init__()
        self.num_subnets = num_subnets
        self.include_grid = include_grid
        self.subnets = nn.ModuleList([
            FNO2d(modes1, modes2, width, n_layers, out_channels, include_grid)
            for _ in range(num_subnets)
        ])
        self.scaling_factors = nn.Parameter(
            torch.tensor([1.0, 40.0, 80.0, 100.0, 140.0, 200.0])
        )
        self.weights = nn.Parameter(
            torch.tensor([1.0 / num_subnets] * num_subnets)
        )

    def forward(self, x):
        outputs = []
        for i, net in enumerate(self.subnets):
            if self.include_grid:
                grid = self.get_scaled_grid(x.shape, x.device, scale=self.scaling_factors[i])
                x_scaled = torch.cat((self.scaling_factors[i] * x, grid), dim=-1)
            else:
                x_scaled = self.scaling_factors[i] * x
            outputs.append(net(x_scaled))
        weighted_sum = torch.stack(outputs, dim=0)
        weighted_sum = torch.einsum('i,i...->...', self.weights, weighted_sum)
        return weighted_sum

    def get_scaled_grid(self, shape, device, scale=1.0):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device) * scale
        gridx = gridx.reshape(1, size_x, 1, 1).expand(batchsize, -1, size_y, -1)
        gridy = torch.linspace(0, 1, size_y, device=device) * scale
        gridy = gridy.reshape(1, 1, size_y, 1).expand(batchsize, size_x, -1, -1)
        return torch.cat((gridx, gridy), dim=-1)


# ================================================================
# 예측 함수
# ================================================================
def predict(model_path, test_data_path, label_min, label_max, save_path, device):
    """
    학습된 MscaleFNO 모델로 예측 수행

    Args:
        model_path: 학습된 모델 체크포인트 경로
        test_data_path: 테스트 데이터 npz 파일 경로
        label_min, label_max: 정규화 파라미터
        save_path: 예측 결과 저장 경로
        device: torch device

    Returns:
        save_path: 저장된 파일 경로
    """

    # ---- 1. 모델 로드 ----
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    model = MultiScaleFNO2d(
        modes1=cfg['modes'],
        modes2=cfg['modes'],
        width=cfg['width'],
        num_subnets=cfg['num_subnets'],
        n_layers=cfg['num_fourier_layers'],
        out_channels=cfg['time_steps'],
        include_grid=cfg['include_grid']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {model_path} (epoch {checkpoint['epoch']})")

    # ---- 2. 테스트 데이터 로드 ----
    test_data = np.load(test_data_path)
    test_input = test_data['inputs'].astype(np.float32)
    if test_input.ndim == 4:
        test_input = test_input[:, 0, :, :]
    print(f"test_input shape: {test_input.shape}")
    # NaN 처리
    test_input[np.isnan(test_input)] = 0.0

    # ---- 3. 정규화 (label의 min/max로, 학습과 동일) ----
    test_input_normalized = (test_input - label_min) / (label_max - label_min + 1e-8)

    # (N, Nx, Ny) → (N, Nx, Ny, 1)  channel-last 형식
    input_tensor = torch.from_numpy(test_input_normalized).unsqueeze(-1).to(device)

    # ---- 4. 예측 ----
    with torch.inference_mode():
        y_pred = model(input_tensor)  # (N, Nx, Ny, T)

    # ---- 5. 역정규화 ----
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred * (label_max - label_min + 1e-8) + label_min

    # (N, Nx, Ny, T) → (N, T, Nx, Ny)  원래 데이터 형식으로
    y_pred = np.transpose(y_pred, (0, 3, 1, 2))

    # ---- 6. 최종 결과: 예측값(변화량) + 초기조건 ----
    predicted_results = y_pred + test_input[:, np.newaxis, :, :]

    # ---- 7. 저장 ----
    np.save(save_path, predicted_results)
    print(f"Predictions saved: {save_path}")
    print(f"  Shape: {predicted_results.shape}")
    print(f"  Range: [{predicted_results.min():.4f}, {predicted_results.max():.4f}]")

    return save_path


# ================================================================
# 메인
# ================================================================
def main():
    cfg = PREDICT_CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(cfg['save_dir'], exist_ok=True)

    for variable, var_cfg in cfg['variables'].items():
        print(f"\n{'='*60}")
        print(f"Variable: {variable}")
        print(f"  Model: {var_cfg['model_path']}")
        print(f"  Scaler: [{var_cfg['label_min']:.4f}, {var_cfg['label_max']:.4f}]")

        for name, data_path in var_cfg['test_data_paths'].items():
            print(f"\n  --- Predicting: {variable} @ {name} ---")
            print(f"  Data: {data_path}")

            save_path = os.path.join(cfg['save_dir'], f"mscaleFNO_{variable}_predictions_{name}.npy")

            predict(
                model_path=var_cfg['model_path'],
                test_data_path=data_path,
                label_min=var_cfg['label_min'],
                label_max=var_cfg['label_max'],
                save_path=save_path,
                device=device
            )

    print(f"\n{'='*60}")
    print("All predictions complete!")


if __name__ == '__main__':
    main()