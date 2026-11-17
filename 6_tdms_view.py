"""Select a TDMS file and plot its numeric EMG channels."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "kaggle" / "DS1_EMG_SIGNALS_AVERAGE"
DEFAULT_SAMPLE_RATE_HZ = 1000
MAX_PLOT_POINTS = 20000
CSV_CACHE_DIR = ROOT / "csv_data" / "tdms_cache"


def choose_tdms_file():
    """Open a file picker, with a console path prompt as a fallback."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a TDMS file",
            initialdir=str(DATASET_DIR if DATASET_DIR.exists() else ROOT),
            filetypes=(("TDMS files", "*.tdms"), ("All files", "*.*")),
        )
        root.destroy()
        if path:
            return Path(path)
    except Exception as exc:
        print("Could not open the file picker: {}".format(exc))

    entered = input("Enter the TDMS path (blank to cancel): ").strip().strip('"')
    return Path(entered) if entered else None


def convert_tdms_to_csv(tdms_path, csv_path):
    """Convert numeric TDMS channels to a reusable CSV cache."""
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise RuntimeError(
            "The nptdms package is required. Install project dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    channels = []
    skipped = []

    with TdmsFile.open(tdms_path) as tdms_file:
        for group in tdms_file.groups():
            for channel in group.channels():
                sample_count = len(channel)
                if sample_count == 0:
                    continue

                values = np.asarray(channel[:])
                if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
                    skipped.append("{}/{}".format(group.name, channel.name))
                    continue

                channels.append(
                    (
                        "{}/{}".format(group.name, channel.name),
                        values,
                    )
                )

    if not channels:
        raise ValueError("No non-empty numeric channels were found in the TDMS file.")
    if skipped:
        print("Skipped non-numeric channels: {}".format(", ".join(skipped)))

    sample_count = max(values.size for _, values in channels)
    data = np.full((sample_count, len(channels)), np.nan, dtype=float)
    for column, (_, values) in enumerate(channels):
        data[: values.size, column] = values

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        output.write("# sample_count={}\n".format(sample_count))
        writer = csv.writer(output)
        writer.writerow([name for name, _ in channels])
        np.savetxt(output, data, delimiter=",", fmt="%.10g")


def load_csv_channels(csv_path, group_number):
    """Load one group's channels from the converted CSV cache."""
    with csv_path.open("r", newline="", encoding="utf-8") as source:
        metadata = source.readline().strip()
        names = next(csv.reader([source.readline()]))

    metadata_key, separator, metadata_value = metadata.partition("=")
    if metadata_key != "# sample_count" or not separator:
        raise ValueError("Invalid sample count in CSV cache: {}".format(csv_path))
    try:
        sample_count = int(metadata_value)
    except ValueError as exc:
        raise ValueError("Invalid sample count in CSV cache: {}".format(csv_path)) from exc

    groups = list(dict.fromkeys(name.rsplit("/", 1)[0] for name in names))
    if group_number < 1 or group_number > len(groups):
        raise ValueError(
            "Group {} is unavailable; this file contains {} groups (choose 1-{}).".format(
                group_number, len(groups), len(groups)
            )
        )

    group_name = groups[group_number - 1]
    column_indexes = [
        index for index, name in enumerate(names) if name.rsplit("/", 1)[0] == group_name
    ]
    channel_names = [names[index].rsplit("/", 1)[-1] for index in column_indexes]
    values = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=2,
        usecols=column_indexes,
        ndmin=2,
    )
    return group_name, channel_names, values, sample_count


def get_csv_cache(tdms_path):
    """Reuse a converted CSV unless its source TDMS file has changed."""
    csv_path = CSV_CACHE_DIR / "{}.csv".format(tdms_path.stem)
    if csv_path.exists() and csv_path.stat().st_mtime >= tdms_path.stat().st_mtime:
        print("Using converted CSV: {}".format(csv_path))
        return csv_path

    print("Converting TDMS to CSV. This is only needed once per source file...")
    convert_tdms_to_csv(tdms_path, csv_path)
    print("CSV cache saved: {}".format(csv_path))
    return csv_path


def plot_tdms(tdms_path, sample_rate_hz, group_number):
    csv_path = get_csv_cache(tdms_path)
    group_name, names, data, sample_count = load_csv_channels(csv_path, group_number)
    stride = max(1, int(np.ceil(sample_count / MAX_PLOT_POINTS)))
    plot_data = data[::stride]
    figure, axes = plt.subplots(
        len(names),
        1,
        sharex=True,
        figsize=(12, max(3, 2.8 * len(names))),
        squeeze=False,
        constrained_layout=True,
    )

    for channel_index, (axis, name) in enumerate(zip(axes[:, 0], names)):
        values = plot_data[:, channel_index]
        time_seconds = np.arange(values.size) * stride / sample_rate_hz
        axis.plot(time_seconds, values, linewidth=0.7)
        axis.set_ylabel("Amplitude")
        axis.set_title(
            "{} | {} samples (showing {})".format(name, sample_count, values.size),
            loc="left",
            fontsize=10,
        )
        axis.grid(alpha=0.25)

    axes[-1, 0].set_xlabel("Time (s), sampling rate {} Hz".format(sample_rate_hz))
    figure.suptitle("{} | {} (group {})".format(tdms_path.name, group_name, group_number))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdms", type=Path, help="TDMS path; omit to open a file picker")
    parser.add_argument(
        "--group",
        type=int,
        default=1,
        help="1-based group to plot (default: 1)",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help="sampling rate in Hz (default: {})".format(DEFAULT_SAMPLE_RATE_HZ),
    )
    args = parser.parse_args()

    tdms_path = args.tdms or choose_tdms_file()
    if tdms_path is None:
        print("No TDMS file selected; cancelled.")
        return
    if not tdms_path.is_file():
        parser.error("TDMS file does not exist: {}".format(tdms_path))
    if args.fs <= 0:
        parser.error("--fs must be positive")

    try:
        plot_tdms(tdms_path, args.fs, args.group)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()