"""
MscaleFNO for 2D Transient Flow Field Prediction
=================================================
초기조건(t=0) → 99 타임스텝 변화량 예측

Based on: You, Xu, Cai - "MscaleFNO: Multi-scale Fourier neural operator
learning for oscillatory functions and wave scattering problems" (JCP, 2026)

데이터 구조:
  inputs:  (N, 756, 282) - 초기 유동장
  outputs: (N, 100, 756, 282) - 변화량 (첫 타임스텝 제외 → 99개 사용)
  nan_mask: (756, 282) - 실린더 내부 마스크

데이터 분할:
  매 10번째 샘플(인덱스 0,10,20,30,40) → 테스트 (5개)
  나머지 → 학습 (45개)
  검증 세트 없음 (테스트 세트로 best model 선택)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import gc
import csv
import json
from timeit import default_timer
from datetime import timedelta

# ================================================================
# 설정 (본인 환경에 맞게 수정)
# ================================================================
CONFIG = {
    # 데이터/스케일링(변수별)은 아래 VARIABLES 에서 지정 — press, vel 둘 다 자동 학습
    'scaling_json': None,

    # 모델
    'modes': 24,
    'width': 16,
    'num_subnets': 6,
    'num_fourier_layers': 4,
    'time_steps': 99,
    'include_grid': False,    # True: 좌표(x,y) 포함 (논문 방식), False: 입력 함수만

    # 학습
    'batch_size': 6,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 5000,
    'seed': 20,

    # 저장
    'save_dir': './test',
    'model_name': 'mscalefno_flow',
}

# 변수별 데이터 경로 / 스케일링 상수 (press, vel 둘 다 자동으로 순회 학습)
VARIABLES = {
    'press': {
        'data_path': '/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/scale_3x/diff_geo2_structured_press_train.npz',
        'label_min': -1861.3638916015625,
        'label_max': 588.5082397460938,
    },
    'vel': {
        'data_path': '/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/scale_3x/diff_geo2_structured_vel_train.npz',
        'label_min': 0.0,
        'label_max': 1.976950764656067,
    },
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ================================================================
# 모델 구성요소
# ================================================================
class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → linear transform → IFFT"""

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
    """Conv2d 기반 2층 MLP"""

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
    """
    단일 FNO 서브네트워크.

    include_grid=True:  입력 (batch, Nx, Ny, 3) → a(x,y), x, y
    include_grid=False: 입력 (batch, Nx, Ny, 1) → a(x,y)만
    출력: (batch, Nx, Ny, T)
    """

    def __init__(self, modes1, modes2, width, n_layers=4, out_channels=99, include_grid=False):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.include_grid = include_grid

        # Lifting: 입력 채널 → width
        in_channels = 3 if include_grid else 1
        self.p = nn.Linear(in_channels, self.width)

        # Fourier layers
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

        # Projection: width → out_channels (타임스텝 수)
        self.q = MLP2d(self.width, out_channels, self.width * 4)

    def forward(self, x):
        x = self.p(x)                    # (batch, Nx, Ny, width)
        x = x.permute(0, 3, 1, 2)        # (batch, width, Nx, Ny)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:    # 마지막 레이어 제외
                x = torch.sin(x)         # MscaleFNO 활성화 함수

        x = self.q(x)                    # (batch, out_channels, Nx, Ny)
        x = x.permute(0, 2, 3, 1)        # (batch, Nx, Ny, out_channels)
        return x


class MultiScaleFNO2d(nn.Module):
    """
    MscaleFNO: 논문 식 (28)

    include_grid=True:  u(x) = Σ γ_i * FNO_θi[c_i*a(x), c_i*x, c_i*y]
    include_grid=False: u(x) = Σ γ_i * FNO_θi[c_i*a(x)]
    """

    def __init__(self, modes1, modes2, width, num_subnets=8,
                 n_layers=4, out_channels=99, include_grid=False):
        super().__init__()
        self.num_subnets = num_subnets
        self.include_grid = include_grid
        self.subnets = nn.ModuleList([
            FNO2d(modes1, modes2, width, n_layers, out_channels, include_grid)
            for _ in range(num_subnets)
        ])

        # 논문 스케일링: {1, 40, 80, 100, 120, 140, 180, 200}
        self.scaling_factors = nn.Parameter(
            torch.tensor([1.0, 40.0, 80.0, 100.0, 140.0, 200.0])
        )
        self.weights = nn.Parameter(
            torch.tensor([1.0 / num_subnets] * num_subnets)
        )

    def forward(self, x):
        # x: (batch, Nx, Ny, 1)
        outputs = []
        for i, net in enumerate(self.subnets):
            if self.include_grid:
                grid = self.get_scaled_grid(x.shape, x.device, scale=self.scaling_factors[i])
                x_scaled = torch.cat((self.scaling_factors[i] * x, grid), dim=-1)
            else:
                x_scaled = self.scaling_factors[i] * x
            outputs.append(net(x_scaled))

        # 가중합산
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
# 파라미터 수 카운트
# ================================================================
def count_params(model):
    total = 0
    for p in model.parameters():
        n = p.numel()
        if p.is_complex():
            n *= 2
        total += n
    return total


# ================================================================
# 메인 학습 루프
# ================================================================
def main(cfg):
    set_seed(cfg['seed'])

    # 학습 모델은 paper_review_msfno/model 폴더에 저장 (reviewer1)
    # (스크립트 위치 기준이라 실행 디렉터리와 무관하게 항상 같은 곳에 저장됨)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'paper_review_msfno', 'model')
    os.makedirs(model_dir, exist_ok=True)

    name = f"best_model_{cfg['variable']}_{cfg['modes']}_reviewer1"
    best_model_path = os.path.join(model_dir, f"{name}.pth")
    last_model_path = os.path.join(model_dir, f"last_model_{cfg['variable']}_{cfg['modes']}_reviewer1.pth")
    log_file = os.path.join(model_dir, f"{name}_log.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Include grid: {cfg['include_grid']}")

    # ============================================================
    # 1. 데이터 로드
    # ============================================================
    print("Loading data...")
    data = np.load(cfg['data_path'], allow_pickle=True)

    inputs = data['inputs'].astype(np.float32)                  # (50, 756, 282)
    outputs = data['outputs'][:, 1:, :, :].astype(np.float32)  # (50, 99, 756, 282)
    nan_mask = data['nan_mask']                                  # (756, 282)

    N, Nx, Ny = inputs.shape
    T = outputs.shape[1]
    print(f"Data: {N} samples, grid {Nx}x{Ny}, {T} timesteps")

    # ============================================================
    # 2. NaN 처리: 0으로 채우기
    # ============================================================
    inputs[np.isnan(inputs)] = 0.0
    outputs[np.isnan(outputs)] = 0.0

    # ============================================================
    # 3. Min-Max 스케일링 (label의 min/max로 통일, 기존 FNO와 동일)
    # ============================================================
    label_min, label_max = cfg['label_min'], cfg['label_max']

    inputs = (inputs - label_min) / (label_max - label_min + 1e-8)
    outputs = (outputs - label_min) / (label_max - label_min + 1e-8)

    print(f"After scaling - inputs: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"After scaling - outputs: [{outputs.min():.4f}, {outputs.max():.4f}]")

    # ============================================================
    # 4. 데이터 분할: 매 10번째 → 테스트, 나머지 → 학습
    # ============================================================
    train_mask = np.ones(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    train_mask[::10] = False
    test_mask[::10] = True

    ntrain = int(train_mask.sum())
    ntest = int(test_mask.sum())

    # inputs: (N, Nx, Ny) → (N, Nx, Ny, 1)  (논문 코드의 channel-last 형식)
    x_all = torch.from_numpy(inputs).unsqueeze(-1)            # (N, Nx, Ny, 1)
    # outputs: (N, T, Nx, Ny) → (N, Nx, Ny, T)
    y_all = torch.from_numpy(outputs).permute(0, 2, 3, 1)     # (N, Nx, Ny, T)

    x_train, y_train = x_all[train_mask], y_all[train_mask]
    x_test, y_test = x_all[test_mask], y_all[test_mask]

    print(f"Train: {x_train.shape} ({ntrain} samples)")
    print(f"Test:  {x_test.shape} ({ntest} samples)")
    print(f"Test indices: {np.where(test_mask)[0].tolist()}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=cfg['batch_size'], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=cfg['batch_size'], shuffle=False
    )

    # ============================================================
    # 5. 모델 생성
    # ============================================================
    model = MultiScaleFNO2d(
        modes1=cfg['modes'],
        modes2=cfg['modes'],
        width=cfg['width'],
        num_subnets=cfg['num_subnets'],
        n_layers=cfg['num_fourier_layers'],
        out_channels=T,
        include_grid=cfg['include_grid']
    ).to(device)

    print(f"Model parameters: {count_params(model):,}")

    # ============================================================
    # 6. 옵티마이저 & 스케줄러
    # ============================================================
    iterations = cfg['epochs'] * (ntrain // cfg['batch_size'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    myloss = nn.MSELoss()

    # ============================================================
    # 7. 로그 파일 초기화
    # ============================================================
    header = ['Epoch', 'Time(s)', 'Train_MSE', 'Test_MSE', 'LR']
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    print(f"Log file: {log_file}")

    # ============================================================
    # 8. 학습 루프
    # ============================================================
    best_test_loss = float('inf')
    best_epoch = 0
    train_start = default_timer()

    for ep in range(cfg['epochs']):
        # --- Train ---
        model.train()
        t1 = default_timer()
        train_mse = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)

            loss = myloss(out, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_mse += loss.item() * x_batch.size(0)

        train_mse /= ntrain

        # --- Test ---
        model.eval()
        test_mse = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                out = model(x_batch)
                test_mse += myloss(out, y_batch).item() * x_batch.size(0)
        test_mse /= ntest

        t2 = default_timer()
        current_lr = optimizer.param_groups[0]['lr']

        # --- 로그 ---
        print(f'Epoch {ep+1:>3d}/{cfg["epochs"]} | '
              f'Time: {t2-t1:.1f}s | '
              f'Train MSE: {train_mse:.6f} | '
              f'Test MSE: {test_mse:.6f} | '
              f'LR: {current_lr:.2e}')

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ep + 1, f"{t2-t1:.2f}",
                f"{train_mse:.6f}", f"{test_mse:.6f}",
                f"{current_lr:.2e}"
            ])

        # --- Best model 저장 (test loss 기준) ---
        if test_mse < best_test_loss:
            best_test_loss = test_mse
            best_epoch = ep + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch,
                'best_test_loss': best_test_loss,
                'scaling_factors': model.scaling_factors.data.cpu(),
                'weights': model.weights.data.cpu(),
                'config': cfg,
            }, best_model_path)

    # --- 총 학습 시간 (epoch 루프 전체 wall-clock) ---
    total_train_time = default_timer() - train_start
    print(f"[{cfg['variable']}] Training time: {total_train_time:.1f} s ({cfg['epochs']} epochs)")

    # --- 학습 종료 시점의 최종(마지막) 모델 저장 ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': cfg['epochs'],
        'best_test_loss': best_test_loss,
        'total_train_time_sec': total_train_time,
        'scaling_factors': model.scaling_factors.data.cpu(),
        'weights': model.weights.data.cpu(),
        'config': cfg,
    }, last_model_path)

    # --- 학습 요약(시간/best)을 JSON 으로 저장 — 언제든 확인용 ---
    summary = {
        "model": "msfno",
        "variable": cfg['variable'],
        "mode": cfg['modes'],
        "epochs": cfg['epochs'],
        "best_epoch": best_epoch,
        "best_test_mse": best_test_loss,
        "total_train_time_sec": round(total_train_time, 2),
        "total_train_time_hms": str(timedelta(seconds=int(total_train_time))),
    }
    summary_path = os.path.join(model_dir, f"train_summary_{cfg['variable']}_{cfg['modes']}_reviewer1.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{cfg['variable']}] Summary saved: {summary_path}")

    # ============================================================
    # 9. 최종 결과 출력
    # ============================================================
    print(f'\n{"="*60}')
    print(f'Training complete! (include_grid={cfg["include_grid"]})')
    print(f'Best epoch: {best_epoch}')
    print(f'Best test MSE: {best_test_loss:.6f}')

    checkpoint = torch.load(best_model_path, map_location='cpu')
    print(f'\nLearned scaling factors: {checkpoint["scaling_factors"].numpy()}')
    print(f'Learned weights: {checkpoint["weights"].numpy()}')
    print(f'Best model saved: {best_model_path}')
    print(f'Last model saved: {last_model_path}')


if __name__ == '__main__':
    # press, vel 두 변수 모두 자동으로 순회 학습 (FNO 와 동일)
    for variable, var_cfg in VARIABLES.items():
        cfg = {**CONFIG, **var_cfg, 'variable': variable}
        print(f"\n{'='*60}")
        print(f"Variable: {variable}")
        print(f"{'='*60}")
        main(cfg)
        gc.collect()
        torch.cuda.empty_cache()