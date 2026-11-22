import torch
import numpy as np
import json
from pathlib import Path
from typing import Union, Tuple, Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio



class LpLoss(object):
	def __init__(self, d=2, p=2, size_average=True, reduction=True):
		super(LpLoss, self).__init__()

		#Dimension and Lp-norm type are postive
		assert d > 0 and p > 0

		self.d = d
		self.p = p
		self.reduction = reduction
		self.size_average = size_average

	def abs(self, x, y):
		num_examples = x.size()[0]

		#Assume uniform mesh
		h = 1.0 / (x.size()[1] - 1.0)

		all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

		if self.reduction:
			if self.size_average:
				return torch.mean(all_norms)
			else:
				return torch.sum(all_norms)

		return all_norms

	def rel(self, x, y):
		num_examples = x.size()[0]

		diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
		y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

		if self.reduction:
			if self.size_average:
				return torch.mean(diff_norms/y_norms)
			else:
				return torch.sum(diff_norms/y_norms)

		return diff_norms/y_norms

	def __call__(self, x, y):
		return self.rel(x, y)


def save_scaler_metrics(data_path:Union[str, Path], 
						dimension: Tuple=None,
						type:str="minmax",
						variable:str="pressure")-> Tuple[np.ndarray, np.ndarray]:
	'''
	Save scaling metrics (mean, std or min, max) to a JSON file for later use in normalization/denormalization.

	Currently, the scaling metrics are calculated using the data of velocity 0.8 m/s.
	'''

	if not isinstance(data_path, Path):
		data_path = Path(data_path)

	if dimension is None:
		keep_dims = False
	else:
		keep_dims = True
	
	data = np.load(data_path)
	inputs: np.ndarray = data["inputs"]
	labels: np.ndarray = data["outputs"]

	root_dir = data_path.parent
	metrics_save_path = root_dir / f"data_scaling_metrics_{type}_{variable}.json"

	if type not in ["standard", "minmax"]:
			raise ValueError("Scaler type must be either 'standard' or 'minmax'")
	
	if type == "standard":
		input_mean: np.ndarray = inputs.mean(axis=dimension, keepdims=True)
		input_std: np.ndarray = inputs.std(axis=dimension, keepdims=True)
		label_mean: np.ndarray = labels.mean(axis=dimension, keepdims=True)
		label_std: np.ndarray = labels.std(axis=dimension, keepdims=True)

		with open(metrics_save_path, "w") as f:
			json.dump({
				"input_mean": input_mean.tolist(),
				"input_std": input_std.tolist(),
				"label_mean": label_mean.tolist(),
				"label_std": label_std.tolist()
			}, f)

		return metrics_save_path
	else:  # minmax
		input_min: np.ndarray = inputs.min(axis=dimension, keepdims=keep_dims)
		input_max: np.ndarray = inputs.max(axis=dimension, keepdims=keep_dims)
		label_min: np.ndarray = labels.min(axis=dimension, keepdims=keep_dims)
		label_max: np.ndarray = labels.max(axis=dimension, keepdims=keep_dims)


		with open(metrics_save_path, "w") as f:
			json.dump({
				"input_min": input_min.tolist(),
				"input_max": input_max.tolist(),
				"label_min": label_min.tolist(),
				"label_max": label_max.tolist()
			}, f)

		return metrics_save_path

def log_metrics(key:str, value:int|float, metrics_type:str="training", save_dir:Union[str, Path]=""):

	save_dir = Path(save_dir)
	logging_path = save_dir / f"{metrics_type}_metrics.ndjson"

	# build a single JSON record for this metric
	record = {"key": key, "value": value}
	
	# append one JSON object per line
	with open(logging_path, "a") as f:
		f.write(json.dumps(record) + "\n")


def normalize(data, data_path, flag:str="input", type:str="minmax")-> torch.tensor:
	if not isinstance(data_path, Path):
		data_path = Path(data_path)
	
	with open(data_path, "r") as f:
		metrics = json.load(f)

	if flag == "input" and type == "standard":
		mean = torch.tensor(metrics["input_mean"], dtype=torch.float32)
		std = torch.tensor(metrics["input_std"], dtype=torch.float32)
		return (data - mean) / std
	elif flag == "label" and type == "standard":
		mean = torch.tensor(metrics["label_mean"], dtype=torch.float32)
		std = torch.tensor(metrics["label_std"], dtype=torch.float32)
		return (data - mean) / std
	elif flag == "input" and type == "minmax":
		min_val = torch.tensor(metrics["input_min"], dtype=torch.float32)
		max_val = torch.tensor(metrics["input_max"], dtype=torch.float32)
		return (data - min_val) / (max_val - min_val)
	elif flag == "label" and type == "minmax":
		min_val = torch.tensor(metrics["label_min"], dtype=torch.float32)
		max_val = torch.tensor(metrics["label_max"], dtype=torch.float32)
		return (data - min_val) / (max_val - min_val)
	else:
		raise ValueError("Invalid flag. Use 'input' or 'label'.")

def denormalize(data, data_path, flag:str="input", type:str="minmax")-> torch.tensor:
	if not isinstance(data_path, Path):
		data_path = Path(data_path)
	
	with open(data_path, "r") as f:
		metrics = json.load(f)

	if flag == "input" and type == "standard":
		mean = torch.tensor(metrics["input_mean"], dtype=torch.float32)
		std = torch.tensor(metrics["input_std"], dtype=torch.float32)
		return data * std + mean
	elif flag == "label" and type == "standard":
		mean = torch.tensor(metrics["label_mean"], dtype=torch.float32)
		std = torch.tensor(metrics["label_std"], dtype=torch.float32)
		return data * std + mean
	elif flag == "input" and type == "minmax":
		min_val = torch.tensor(metrics["input_min"], dtype=torch.float32)
		max_val = torch.tensor(metrics["input_max"], dtype=torch.float32)
		return data * (max_val - min_val) + min_val
	elif flag == "label" and type == "minmax":
		min_val = torch.tensor(metrics["label_min"], dtype=torch.float32)
		max_val = torch.tensor(metrics["label_max"], dtype=torch.float32)
		return data * (max_val - min_val) + min_val
	else:
		raise ValueError("Invalid flag. Use 'input' or 'label'.")


def visualize_output(data, timestep:int,mode:str="rgb_array"
):
	fig, ax = plt.subplots(1, 1, figsize=(13, 5))
	ax = np.atleast_1d(ax)
	im = ax[0].imshow(data.T, origin="lower", cmap="coolwarm")
	fig.tight_layout()
	fig.suptitle(f"{timestep}s")
	if mode == "image":
		plt.savefig(f"{timestep}.png")
		plt.tight_layout()
		plt.close()
		return True
	elif mode == "rgb_array":
		fig.canvas.draw()
		rgb_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
			fig.canvas.get_width_height()[::-1] + (4,))
		plt.close()
		return rgb_array
	else:
		plt.close()
		raise ValueError("Invalid mode. Must be either 'image' or 'rgb_array'.")
	
def make_animation(prediction_data_path, ground_truth, type="predictions", suffix="0.4"):
	image_list = []
	for timestep in range(1, ground_truth.shape[1]):
		if type == "predictions":
			prediction_data = np.load(prediction_data_path + f"/predictions_{suffix}.npy")[0, timestep, :, :]
			image_list.append(visualize_output(prediction_data, timestep))
		else:
			image_list.append(visualize_output(ground_truth[:, timestep, :, :], timestep))
		# Optionally, you can also visualize the ground truth if needed
		# visualize_output(ground_truth[:, timestep, :, :], timestep, mode="image")
	save_name = f"prediction_{suffix}.gif" if type == "predictions" else f"ground_truth_{suffix}.gif"
	imageio.mimsave(save_name, image_list, fps=10, loop=0)

def plot_probe_points(ground_truth: np.ndarray,
					  predicted: np.ndarray,
					  locations: tuple = ((120, 141), (220, 141), (320, 141), (420, 141)),
					  save_path: Union[str, Path] = ""):
	"""
	Generates and saves a publication-quality plot comparing ground truth and 
	predicted time-series data at specific probe point locations.
	
	The legend is placed horizontally above the plot.
	Assumes data shape is [Batch, Time, H, W].

	Args:
		ground_truth: Ground truth data array [B, T, H, W].
		predicted: Predicted data array [B, T, H, W].
		locations: A tuple of (x, y) or (h, w) coordinates for probe points.
		save_path: Directory to save the final plot.
	"""
	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	# --- Style Setup (Nature-like) ---
	plt.rcParams['font.family'] = 'sans-serif'
	plt.rcParams['font.size'] = 12
	plt.rcParams['axes.linewidth'] = 1.2
	plt.rcParams['xtick.major.width'] = 1.2
	plt.rcParams['ytick.major.width'] = 1.2

	fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

	# Use a colorblind-safe, distinct color palette
	colors = plt.colormaps['tab10']
	probe_handles = []

	for i, (x, y) in enumerate(locations):
		color = colors(i)
		
		# Plot Ground Truth (solid)
		line, = ax.plot(ground_truth[0, :, x, y], color=color, ls='-', lw=1.5,
						 label=f'Probe ({x}, {y})')
		probe_handles.append(line)
		
		# Plot Predicted (dashed)
		ax.plot(predicted[0, :, x, y], color=color, ls='--', lw=1.5)

	# --- Define all legend handles ---
	style_handles = [
		Line2D([0], [0], color="black", lw=1.5, ls="-", label="Ground Truth"),
		Line2D([0], [0], color="black", lw=1.5, ls="--", label="Predicted"),
	]
	
	# Combine style and probe handles for one legend
	all_handles = style_handles + probe_handles
	
	# --- Place the single legend above the plot ---
	ax.legend(
		handles=all_handles,
		loc='lower center', # Anchor the legend at its bottom-center
		bbox_to_anchor=(0.5, 1.05), # Position it (50% x, 5% y) above the axes
		ncol=len(all_handles), # All entries in one horizontal row
		frameon=False, # No box
		fontsize=9
	)

	# --- Styling ---
	ax.set_title('Probe Point Time Series Comparison', loc='left', fontweight='bold')
	ax.set_ylabel('Value')
	ax.set_xlabel('Time Steps')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	
	# --- Finalize and Save ---
	# Use rect to make space for the legend above the title
	fig.tight_layout(rect=[0, 0, 1, 0.9]) 
	try:
		# save_path = save_dir / "probe_points_comparison.png"
		plt.savefig(save_path)
		print(f"Probe plot saved successfully to {save_path}")
	except Exception as e:
		print(f"Error saving probe plot: {e}")
	finally:
		plt.close(fig) # Always close the figure

def plot_variable_drop(ground_truth: np.ndarray,
						predicted: np.ndarray,
						save_path: Union[str, Path] = "",
						variable: str = "pressure"):
	"""
	Generates and saves a plot comparing the average drop over time.
	
	The legend is placed horizontally above the plot.
	Assumes data shape is [Batch, Time, H, W].

	Args:
		ground_truth: Ground truth data array [B, T, H, W].
		predicted: Predicted data array [B, T, H, W].
		save_dir: Directory to save the final plot.
	"""
	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	# --- Style Setup (Nature-like) ---
	plt.rcParams['font.family'] = 'sans-serif'
	plt.rcParams['font.size'] = 12
	plt.rcParams['axes.linewidth'] = 1.2
	plt.rcParams['xtick.major.width'] = 1.2
	plt.rcParams['ytick.major.width'] = 1.2

	fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

	# --- Pressure Drop Calculation ---
	gt_data = ground_truth[0]  # Shape [T, H, W]
	pred_data = predicted[0] # Shape [T, H, W]
	
	H, W = gt_data.shape[1], gt_data.shape[2]

	
	if H > W: # Shorter walls are top/bottom
		p_drop_gt = np.mean(gt_data[..., 0, :], axis=-1) - np.mean(gt_data[..., -1, :], axis=-1)
		p_drop_pred = np.mean(pred_data[..., 0, :], axis=-1) - np.mean(pred_data[..., -1, :], axis=-1)
	else: # Shorter walls are left/right
		p_drop_gt = np.mean(gt_data[..., :, 0], axis=-1) - np.mean(gt_data[..., :, -1], axis=-1)
		p_drop_pred = np.mean(pred_data[..., :, 0], axis=-1) - np.mean(pred_data[..., :, -1], axis=-1)
		
	time_steps = np.arange(p_drop_gt.shape[0])

	# --- Plot Pressure Drop ---
	ax.plot(time_steps, p_drop_gt, color='blue', ls='-', lw=1.5, label='Ground Truth')
	ax.plot(time_steps, p_drop_pred, color='green', ls='--', lw=1.5, label='Predicted')

	# --- Place the single legend above the plot ---
	ax.legend(
		loc='lower center', # Anchor the legend at its bottom-center
		bbox_to_anchor=(0.5, 1.05), # Position it (50% x, 5% y) above the axes
		ncol=2, # Two entries
		frameon=False, # No box
		fontsize=10
	)

	# --- Styling ---
	ax.set_title(f'Average {variable} Drop Over Time', loc='left', fontweight='bold')
	ax.set_ylabel(rf'{variable} Drop ($\Delta {variable.capitalize()}$)')
	ax.set_xlabel('Time Steps')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	
	# --- Finalize and Save ---
	# Use rect to make space for the legend above the title
	fig.tight_layout(rect=[0, 0, 1, 0.9])
	try:
		plt.savefig(save_path)
		print(f"{variable.capitalize()} drop plot saved successfully to {save_path}")
	except Exception as e:
		print(f"Error saving {variable} drop plot: {e}")
	finally:
		plt.close(fig) # Always close the figure

def load_metrics(metrics_path: Path) -> Dict[str, List[float]]:
	data = defaultdict(list)
	with open(metrics_path, "r") as f:
		if str(metrics_path).endswith(".ndjson"):
			for line in f:
				record = json.loads(line)
				data[record["key"]].append(record["value"])
		else:
			data = json.load(f)
	return data

def plot_loss(loss_metrics_path:Union[str, Path], suffix:str="0.4",
			  save_dir:Union[str, Path]=""):

	save_dir = Path(save_dir)
	metrics = load_metrics(Path(loss_metrics_path))

	plt.plot(metrics["epoch"], metrics["train_loss"], label="Training Loss", color="r")
	plt.plot(metrics["epoch"], metrics["validation_loss"], label="Validation Loss", color="b")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.yscale("log")
	plt.legend()
	plt.savefig(save_dir / f"loss_curves_{suffix}.png")
	plt.close()