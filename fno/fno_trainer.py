import gc
from utils import (
	normalize,
	denormalize, 
	log_metrics, 
	plot_probe_points, 
	plot_loss, 
	plot_variable_drop)
from neuralop.models.fno import FNO

import torch
from pathlib import Path
import numpy as np
import random
import tqdm
import json
from typing import Union

from dataset import FNODataset
from datetime import datetime, timedelta

# from models.FNO2D import FNO2D


def set_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


class FNO2DTrainer:
	def __init__(self, 
			  input_channels=1, 
			  output_channels=99, 
			  modes1=24, modes2=24, 
			  width=256, 
			  depth=4,
			  t_input=10,
			  learning_rate=0.0001,
			  x_coords=None,
			  y_coords=None,
			  include_grid:bool=False,
			  epochs:int=500,
			  activation=torch.functional.F.gelu):
		# self.model = FNO2D(input_channels=input_channels, output_channels=output_channels,
		# 				   modes1=modes1, modes2=modes2, width=width, depth=depth,
		# 				   x_coords=x_coords,
		# 				   y_coords=y_coords,
		# 				   include_grid=include_grid,
		# 				   activation=activation)

		self.model = FNO(
			n_modes=[modes1, modes2],
			hidden_channels=width,
			in_channels=input_channels,
			out_channels=output_channels,
			n_layers=depth,
			norm="instance_norm")
		
		# self.t_input = t_input
		self.epochs = epochs
		self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
		# placeholder — train()에서 총 iteration 수에 맞춰 재설정됨 (per-batch step, msFNO와 동일)
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
		self.criterion = torch.nn.MSELoss()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)

		self.modes1 = modes1
		self.modes2 = modes2

	def predict(self, 
         input_tensor: torch.Tensor, 
         save_dir: Path, 
         metrics_save_path: Union[str, Path],
         suffix: str = "0.4",
         model_dir: Path = None):

		import time

		metrics_save_path = Path(metrics_save_path)
		try: 
			variable = metrics_save_path.stem.split("_")[-1]
			scaler_type = metrics_save_path.stem.split("_")[-2]
		except IndexError:
			raise ValueError("Metrics file name must contain scaler type and variable name")

		if model_dir is None:
			model_dir = save_dir
		model: torch.nn.Module = self.load_model(Path(model_dir) / f"best_model_{variable}_{self.modes1}_reviewer1.pth")
		model.eval()

		input_tensor_normalized = normalize(input_tensor, metrics_save_path, flag="label", type=scaler_type)

		# GPU 워밍업
		if self.device.type == 'cuda':
			dummy = torch.randn_like(input_tensor_normalized[:1]).to(self.device)
			with torch.inference_mode():
				_ = model(dummy)
			torch.cuda.synchronize()

		with torch.inference_mode():
			assert input_tensor_normalized.ndim == 4, "Input tensor must be 4D (B, C, H, W)"
			input_tensor_normalized = input_tensor_normalized.to(self.device)

			# 시간 측정 시작
			if self.device.type == 'cuda':
				torch.cuda.synchronize()
			start_time = time.time()

			y_pred = model(input_tensor_normalized)

			if self.device.type == 'cuda':
				torch.cuda.synchronize()
			end_time = time.time()

			inference_time = end_time - start_time
			n_samples = input_tensor.shape[0]
			print(f"\n{'='*50}")
			print(f"[{suffix}] Inference Time: {inference_time:.4f} s")
			print(f"[{suffix}] Samples: {n_samples}")
			print(f"[{suffix}] Time per sample: {inference_time / n_samples:.4f} s")
			print(f"{'='*50}")

			y_pred = denormalize(y_pred.cpu(), metrics_save_path, flag="label", type=scaler_type)
			predicted_results = y_pred + input_tensor
			np.save(save_dir / f"predictions_{suffix}.npy", predicted_results)

		return save_dir / f"predictions_{suffix}.npy"

	def process_one_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
		inputs = inputs.to(self.device)
		labels = labels.to(self.device)

		predictions = self.model(inputs)
		loss = self.criterion(predictions.reshape(predictions.shape[0], -1), labels.reshape(labels.shape[0], -1))

		if self.model.training:
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			self.scheduler.step()   # per-batch step (msFNO와 동일)

		return loss

	
	def training_loop(self, data_loader:torch.utils.data.DataLoader):
		step_loss = 0.0
		for inputs, labels in data_loader:
			loss = self.process_one_batch(inputs, labels)
			step_loss += loss.item() * inputs.size(0)

		return step_loss / len(data_loader.dataset)
	
	def train(self,
		   train_loader,
		   test_loader,
		   save_dir:Path,
		   variable:str="pressure",
		   model_dir:Path=None):

		if model_dir is None:
			model_dir = save_dir
		model_dir = Path(model_dir)
		best_model_path = model_dir / f"best_model_{variable}_{self.modes1}_reviewer1.pth"
		last_model_path = model_dir / f"last_model_{variable}_{self.modes1}_reviewer1.pth"

		# msFNO 와 동일한 LR 스케줄: per-batch CosineAnnealing, T_max = 총 iteration 수
		iterations = self.epochs * (len(train_loader.dataset) // train_loader.batch_size)
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=iterations)

		import time
		validation_loss = float('inf')
		best_epoch = 0
		train_start = time.time()
		for epoch in tqdm.tqdm(range(self.epochs), desc="Training Epochs"):
			epoch_start = time.time()
			self.model.train()

			train_mse_step_loss = self.training_loop(train_loader)

			self.model.eval()
			with torch.no_grad():
				test_mse_step_loss = self.training_loop(test_loader)
			if test_mse_step_loss < validation_loss:
				self.save_model(best_model_path)
				validation_loss = test_mse_step_loss
				best_epoch = epoch + 1

			# print("\033[4A", end="")
			# print(f"\nEpoch: {epoch},\nTraining loss: {train_mse_step_loss},\nValidation loss: {test_mse_step_loss}\n",  end='', flush=True)

			log_metrics("train_loss", train_mse_step_loss, f"training_{variable}", save_dir)
			log_metrics("validation_loss", test_mse_step_loss, f"training_{variable}", save_dir)
			log_metrics("epoch", epoch, f"training_{variable}", save_dir)
			log_metrics("epoch_time_sec", time.time() - epoch_start, f"training_{variable}", save_dir)

		# 총 학습 시간 기록 (epoch 루프 전체 wall-clock)
		total_train_time = time.time() - train_start
		log_metrics("total_train_time_sec", total_train_time, f"training_{variable}", save_dir)
		print(f"[{variable}] Training time: {total_train_time:.1f} s ({self.epochs} epochs)")

		# 학습 종료 시점의 최종(마지막) 모델 저장
		self.save_model(last_model_path)

		# 학습 요약(시간/best)을 JSON 으로 저장 — 언제든 확인용
		summary = {
			"model": "fno",
			"variable": variable,
			"mode": self.modes1,
			"epochs": self.epochs,
			"best_epoch": best_epoch,
			"best_test_mse": validation_loss,
			"total_train_time_sec": round(total_train_time, 2),
			"total_train_time_hms": str(timedelta(seconds=int(total_train_time))),
		}
		summary_path = model_dir / f"train_summary_{variable}_{self.modes1}_reviewer1.json"
		with open(summary_path, "w") as f:
			json.dump(summary, f, indent=2)
		print(f"[{variable}] Summary saved: {summary_path}")

	def save_model(self, path):
		path.parent.mkdir(parents=True, exist_ok=True)

		# TODO: change this in future, we are hard-coding the mode numbers to the path.
	
		torch.save(
			{"model_dict":self.model.state_dict(),
				"optimizer_dict":self.optimizer.state_dict(),
				"scheduler_dict":self.scheduler.state_dict()
			}, path)
		# print(f"Model saved to {path}")

	def load_model(self, path):
		with torch.serialization.safe_globals(
        [torch._C._nn.gelu]):
			checkpoint = torch.load(path, map_location=self.device, weights_only=False)
		self.model.load_state_dict(checkpoint["model_dict"])
		if "optimizer_dict" in checkpoint:
			self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
		if "scheduler_dict" in checkpoint:
			self.scheduler.load_state_dict(checkpoint["scheduler_dict"])
		self.model.to(self.device)
		print(f"Model loaded from {path}")
		return self.model


	def fit(self, train_loader, test_loader, save_dir:Path, variable:str="pressure", model_dir:Path=None):
		metrics = self.train(train_loader, test_loader, save_dir=save_dir, variable=variable, model_dir=model_dir)
		return metrics
	
def FNO_HCSG_Trainer(
	epochs:int=5000,
	seed:int=20,
	variable:str="velocity",
	training_set_path:str="",
	interpolation_set_path:str="",
	extrapolation_set_path:str="",
	scaler_set_path:str="",
	step_size:int=1,
	use_pretrained_inference:bool=False,
	scaler_type:str="minmax",
	t_output:int=99,
	batch_size:int=6,
	scaler_dimension:tuple=None,
	modes1:int=24,
	modes2:int=24,
	width:int=128,
	depth:int=4,
	learning_rate:float=0.001,
	activation=torch.nn.GELU):
	set_seed(seed)

	hparams = {
		"epochs": epochs,
		"seed": seed,
		"variable": variable,
		"training_set_path": str(training_set_path),
		"interpolation_set_path": str(interpolation_set_path),
		"extrapolation_set_path": str(extrapolation_set_path),
		"scaler_set_path": str(scaler_set_path),
		"step_size": step_size,
		"use_pretrained_inference": use_pretrained_inference,
		"scaler_type": scaler_type,
		"t_output": t_output,
		"batch_size": batch_size,
		"scaler_dimension": scaler_dimension,
		"modes1": modes1,
		"modes2": modes2,
		"width": width,
		"depth": depth,
		"learning_rate": learning_rate,
		"activation": str(activation)	}

	training_set = Path(hparams["training_set_path"])
	interpolation_set = Path(hparams["interpolation_set_path"])
	extrapolation_set = Path(hparams["extrapolation_set_path"])
	scaler_set = Path(hparams["scaler_set_path"])

	root_dir = training_set.parent.parent / "predictions_new"
	date_hr = datetime.now().strftime("%m-%d_%H")
	save_dir = root_dir / date_hr / variable
	model_save_dir = save_dir #TODO: separate model save dir in future if that's convenient

	# 학습 모델은 paper_review_fno/model 폴더에 저장 (reviewer1)
	model_dir = training_set.parent.parent / "paper_review_fno" / "model"

	model_save_dir.mkdir(parents=True, exist_ok=True)
	save_dir.mkdir(parents=True, exist_ok=True)
	model_dir.mkdir(parents=True, exist_ok=True)

	with open(save_dir / "hparams.json", "w") as f:
		json.dump(hparams, f, indent=4)

	cylinderFNO = FNO2DTrainer(
		epochs=epochs,
		include_grid=False,
		modes1=modes1,
		modes2=modes2,
		width=width,
		depth=depth,
		learning_rate=learning_rate,
		activation=activation,
	)

	if use_pretrained_inference:
		model_path = model_dir / f"best_model_{variable}_{cylinderFNO.modes1}_reviewer1.pth"
		cylinderFNO.load_model(model_path)

		try:
			scaler_metrics_path = scaler_set.parent / f"data_scaling_metrics_{scaler_type}_{variable}.json"
			scaler_metrics_path.exists()
		except Exception as e:
			dataset = FNODataset(training_set, t_output=t_output)

			train_loader, test_loader = dataset.get_dataloader(
				batch_size=batch_size,
				step_size=step_size,
				scaler_data_path=scaler_set,
				scaler_type=scaler_type,
				# scaler_dimension=(0,2,3)
				scaler_dimension=scaler_dimension,
				variable=variable
			)

			scaler_metrics_path = dataset.metrics_save_path
		 
	else:
		dataset = FNODataset(training_set, t_output=t_output)

		train_loader, test_loader = dataset.get_dataloader(
			batch_size=batch_size,
			step_size=step_size,
			scaler_data_path=scaler_set,
			scaler_type=scaler_type,
			# scaler_dimension=(0,2,3)
			scaler_dimension=scaler_dimension,
			variable=variable
		)
		cylinderFNO.fit(train_loader,
				  test_loader,
				  save_dir=model_save_dir,
				  variable=variable,
				  model_dir=model_dir)
		
		scaler_metrics_path = dataset.metrics_save_path

	# Testing
	test_data = np.load(interpolation_set)
	test_input = test_data['inputs']

	extrapolation_test = np.load(extrapolation_set)
	extrapolation_test_input = extrapolation_test['inputs']

	prediction_path = cylinderFNO.predict(
		torch.from_numpy(test_input).float(),
		suffix=f"{variable}_0.4",
		save_dir=model_save_dir,
		metrics_save_path=scaler_metrics_path,
		model_dir=model_dir
	)

	prediction_path_extrapolation = cylinderFNO.predict(
		torch.from_numpy(extrapolation_test_input).float(),
		suffix=f"{variable}_0.7",
		save_dir=model_save_dir,
		metrics_save_path=scaler_metrics_path,
		model_dir=model_dir
	)

	predicted_results = np.load(prediction_path)
	predicted_results_extrapolation = np.load(prediction_path_extrapolation)
	
	plot_probe_points(
		ground_truth=test_data["outputs"] + test_data["inputs"],
		predicted=predicted_results,
		save_path=save_dir / f"probe_points_{variable}_0.4.png"
	)

	plot_variable_drop(
		ground_truth=test_data["outputs"] + test_data["inputs"],
		predicted=predicted_results,
		save_path=save_dir / f"{variable}_drop_0.4.png",
		variable=variable
	)

	plot_loss(
		loss_metrics_path= model_save_dir / f"training_{variable}_metrics.ndjson",
		suffix=f"{variable}_0.4",
		save_dir=save_dir
	)

	plot_probe_points(
		ground_truth=test_data["outputs"] + test_data["inputs"],
		predicted=predicted_results,
		save_path=save_dir / f"probe_points_{variable}_0.4_vertical.png",
		locations=[(0, 100), (0, 200), (-1, 150), (-1, 250)]
	)

	plot_probe_points(
		ground_truth=extrapolation_test["outputs"] + extrapolation_test["inputs"],
		predicted=predicted_results_extrapolation,
		save_path=save_dir / f"probe_points_{variable}_0.7.png"
	)

	plot_variable_drop(
		ground_truth=extrapolation_test["outputs"] + extrapolation_test["inputs"],
		predicted=predicted_results_extrapolation,
		save_path=save_dir / f"{variable}_drop_0.7.png",
		variable=variable
	)
	


if __name__ == "__main__":
    import time

    for variable in ["press", "vel"]:
        training_set = f"/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/fno_hcsg/data/diff_geo2_structured_{variable}_train.npz"
        interpolation_set = f"/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/fno_hcsg/data/geo2_structured_{variable}_test_ic040.npz"
        extrapolation_set = f"/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/fno_hcsg/data/geo2_structured_{variable}_test_ic070.npz"
        scaler_set = f"/home/minseo/minseo_data/minseo_oll/2D_HCSG/data/fno_hcsg/data/geo2_structured_{variable}_test_ic080.npz"

        print(f"\n{'='*60}")
        print(f"Variable: {variable}")
        print(f"{'='*60}")

        start_time = time.time()

        FNO_HCSG_Trainer(
            use_pretrained_inference=False,
            epochs=5000,
            variable=variable,
            training_set_path=training_set,
            interpolation_set_path=interpolation_set,
            extrapolation_set_path=extrapolation_set,
            scaler_set_path=scaler_set,
            batch_size=6
        )

        end_time = time.time()
        print(f"\n[{variable}] Total time (train+inference+plot): {end_time - start_time:.4f} s")

        gc.collect()
        torch.cuda.empty_cache()