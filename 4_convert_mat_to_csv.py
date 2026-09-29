"""Convert Mov1.mat through Mov5.mat into one CSV file per movement."""

import argparse
import csv
from pathlib import Path

import scipy.io


DEFAULT_DATA_DIR = (
    Path(__file__).parent / "dataset" / "kaggle" / "DS1_EMG_SIGNALS_AVERAGE"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "csv_data"
MOVEMENTS = range(1, 6)


def convert_file(mat_path, csv_path):
    """Convert one MATLAB movement array to a long-format CSV file."""
    movement_name = mat_path.stem
    mat_data = scipy.io.loadmat(mat_path)
    if movement_name not in mat_data:
        raise KeyError(f"{mat_path} does not contain a '{movement_name}' variable")

    signal = mat_data[movement_name]
    if signal.ndim != 3 or signal.shape[1] != 3:
        raise ValueError(
            f"{mat_path} has shape {signal.shape}; expected (samples, 3, trials)"
        )

    sample_count, channel_count, trial_count = signal.shape
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["sample", "trial", "channel_1", "channel_2", "channel_3"])
        for trial_index in range(trial_count):
            for sample_index in range(sample_count):
                values = signal[sample_index, :, trial_index]
                writer.writerow(
                    [
                        sample_index + 1,
                        trial_index + 1,
                        *[f"{value:.10g}" for value in values],
                    ]
                )

    return sample_count, channel_count, trial_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing Mov1.mat through Mov5.mat",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory where Mov1.csv through Mov5.csv will be written",
    )
    args = parser.parse_args()

    for movement in MOVEMENTS:
        name = f"Mov{movement}"
        mat_path = args.data_dir / f"{name}.mat"
        csv_path = args.output_dir / f"{name}.csv"
        if not mat_path.exists():
            parser.error(f"Could not find {mat_path}")

        sample_count, channel_count, trial_count = convert_file(mat_path, csv_path)
        print(
            f"Created {csv_path} ({sample_count} samples x "
            f"{channel_count} channels x {trial_count} trials)"
        )


if __name__ == "__main__":
    main()
