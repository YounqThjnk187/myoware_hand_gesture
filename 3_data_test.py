"""View the EMG signals stored in Mov1.mat through Mov5.mat."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.io


DATA_DIR = Path(__file__).parent / "dataset" / "kaggle" / "DS1_EMG_SIGNALS_AVERAGE"
MOVEMENTS = range(1, 6)


def load_movements(data_dir):
	"""Load Mov1 through Mov5 and return them keyed by movement number."""
	movements = {}
	for movement in MOVEMENTS:
		name = f"Mov{movement}"
		path = data_dir / f"{name}.mat"
		if not path.exists():
			raise FileNotFoundError(f"Could not find {path}")

		mat_data = scipy.io.loadmat(path)
		if name not in mat_data:
			raise KeyError(f"{path} does not contain a '{name}' variable")

		signal = mat_data[name]
		if signal.ndim != 3 or signal.shape[1] != 3:
			raise ValueError(
				f"{path} has shape {signal.shape}; expected (samples, 3, trials)"
			)
		movements[movement] = signal
	return movements


def plot_movements(movements, trial=None):
	"""Plot all movements, either for one trial or averaged across trials."""
	figure, axes = plt.subplots(
		nrows=5,
		ncols=1,
		sharex=True,
		figsize=(12, 12),
		constrained_layout=True,
	)
	colors = ("tab:blue", "tab:orange", "tab:green")

	for movement, axis in zip(MOVEMENTS, axes):
		signal = movements[movement]
		if trial is None:
			values = signal.mean(axis=2)
			title = f"Mov{movement} (average of {signal.shape[2]} trials)"
		else:
			values = signal[:, :, trial]
			title = f"Mov{movement} (trial {trial + 1} of {signal.shape[2]})"

		for channel, color in enumerate(colors):
			axis.plot(values[:, channel], color=color, label=f"Channel {channel + 1}")
		axis.set_title(title, loc="left")
		axis.set_ylabel("Signal")
		axis.grid(alpha=0.25)
		axis.legend(loc="upper right", ncols=3)

	axes[-1].set_xlabel("Sample")
	figure.suptitle("EMG movement signals", fontsize=16)
	plt.show()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--trial",
		type=int,
		help="1-based trial number to display; omit to show the average of all trials",
	)
	args = parser.parse_args()

	movements = load_movements(DATA_DIR)
	trial = None
	if args.trial is not None:
		trial_count = next(iter(movements.values())).shape[2]
		if not 1 <= args.trial <= trial_count:
			parser.error(f"--trial must be between 1 and {trial_count}")
		trial = args.trial - 1

	plot_movements(movements, trial)


if __name__ == "__main__":
	main()
