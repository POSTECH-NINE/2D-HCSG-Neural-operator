import torch
import numpy as np
import json
from typing import Union
from pathlib import Path
import json

from utils import save_scaler_metrics

class FNODataset(torch.utils.data.Dataset):
	def __init__(self, data_path, t_output=99):

		self.data_path = data_path
		self.t_output = t_output
		self.data = np.load(data_path)
		if not isinstance(self.data_path, Path):
			self.data_path = Path(self.data_path)
		self.root_dir = self.data_path.parent
		# self.mask = self.get_mask()

	def get_dataset(self, 
		scaler_data_path: Union[str, Path],
		scaler_type: str="minmax",
		scaler_dimension: tuple=None,
		include_ic_label: bool=False,
		variable:str="pressure"
		):
		'''
		Currently difference value calculated data are used to train the model. 

		Input to the model is true initial condition. 
		Output to the model is difference between true value and initial condition.

		So, include_ic_label selects whether to include initial condition in the output label or not.
		If included the first time step of the output label will be all zeros, so it might not be very useful.
		Hence, by default it is set to False.
		'''

		data = self.data

		inputs: np.ndarray = data['inputs']
		if inputs.ndim == 3:
			inputs = np.expand_dims(inputs, axis=1)  # Add channel dim if missing:
		
		outputs: np.ndarray = data['outputs']
		# if include_ic_label:
		# 	outputs: np.ndarray = data['outputs']
		# elif data["outputs"].shape[1] != self.t_output:
		# 	starting_index = data["outputs"].shape[1] - self.t_output
		# 	starting_index = max(0, starting_index)
		# 	outputs: np.ndarray = data['outputs'][:, starting_index:, ...]
		# else:
		# 	outputs = data['outputs'][:, 1:, ...]  # Exclude first time step

		batchsize = inputs.shape[0]

		train_mask = torch.ones(batchsize, dtype=bool)
		test_mask = torch.zeros(batchsize, dtype=bool)
		train_mask[::10] = False  # Set every 10th sample to False for training
		test_mask[::10] = True    # Set every 10th sample to True for testing

		self.metrics_save_path = save_scaler_metrics(
			data_path=scaler_data_path,
			dimension=scaler_dimension,
			type=scaler_type,
			variable=variable
		)

		with open(self.metrics_save_path, "r") as f:
			metrics = json.load(f)

		if scaler_type == "standard":
			input_mean = np.array(metrics["input_mean"])
			input_std = np.array(metrics["input_std"])
			label_mean = np.array(metrics["label_mean"])
			label_std = np.array(metrics["label_std"])
			inputs_ = (inputs - input_mean) / (input_std + 1e-16)
			outputs_ = (outputs - label_mean) / (label_std + 1e-16)
		else:  # minmax
			input_min = np.array(metrics["input_min"])
			input_max = np.array(metrics["input_max"])
			label_min = np.array(metrics["label_min"])
			label_max = np.array(metrics["label_max"])
			inputs_ = (inputs - label_min) / (label_max - label_min + 1e-16) #TODO: normalized inputs with the labels' min and max
			outputs_ = (outputs - label_min) / (label_max - label_min + 1e-16)

		train_input = inputs_[train_mask]
		train_label = outputs_[train_mask]

		test_input = inputs_[test_mask]
		test_label = outputs_[test_mask]

		return train_input, train_label, test_input, test_label

	def get_dataloader(self, batch_size=32, step_size=1,
					   scaler_data_path: Union[str, Path]="",
					   scaler_type: str="minmax",
					   scaler_dimension: tuple=None,
					   include_ic_label: bool=False,
					   variable:str="pressure"):

		train_input, train_label, test_input, test_label = self.get_dataset(
			scaler_data_path=scaler_data_path,
			scaler_type=scaler_type,
			scaler_dimension=scaler_dimension,
			include_ic_label=include_ic_label,
			variable=variable
		)

		train_dataset = torch.utils.data.TensorDataset(
			torch.tensor(train_input[..., ::step_size, ::step_size], dtype=torch.float32),
			torch.tensor(train_label[..., ::step_size, ::step_size], dtype=torch.float32)
		)
		test_dataset = torch.utils.data.TensorDataset(
			torch.tensor(test_input[..., ::step_size, ::step_size], dtype=torch.float32),
			torch.tensor(test_label[..., ::step_size, ::step_size], dtype=torch.float32)
		)

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

		return train_loader, test_loader
	
	def get_coordinates(self):
		data = self.data
		if 'x_coords' in data and 'y_coords' in data:
			return data['x_coords'], data['y_coords']
		else:
			raise KeyError("Coordinates not found in the dataset.")
	
	def get_mask(self):
		data = self.data
		if 'nan_mask' in data:
			return data['nan_mask']
		else:
			raise KeyError("Mask not found in the dataset.")
		
	def __len__(self):
		return self.data['inputs'].shape[0]
	
	def __getitem__(self, idx):
		inputs = self.data['inputs']
		outputs = self.data['outputs']
		return torch.tensor(inputs[idx], dtype=torch.float32), torch.tensor(outputs[idx], dtype=torch.float32)