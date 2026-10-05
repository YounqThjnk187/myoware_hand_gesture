"""Choose a converted movement CSV and plot or replay one trial's EMG channels."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


ROOT = Path(__file__).parent
CSV_DIR = ROOT / "csv_data"
SAMPLE_RATE_HZ = 1000
CHANNEL_COLUMNS = ("channel_1", "channel_2", "channel_3")


def choose_csv():
    """Open a file picker, with a console path prompt as a fallback."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a converted movement CSV",
            initialdir=str(CSV_DIR if CSV_DIR.exists() else ROOT),
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        root.destroy()
        if path:
            return Path(path)
    except Exception as exc:
        print("Could not open the file picker: {}".format(exc))

    entered = input("Enter the CSV path (blank to cancel): ").strip().strip('"')
    return Path(entered) if entered else None


def read_trial(csv_path, trial_number):
    """Read only the selected trial; return sample indexes and three channel arrays."""
    samples = []
    channels = [[], [], []]

    with csv_path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"sample", "trial", *CHANNEL_COLUMNS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV columns must include: {}".format(", ".join(sorted(required)))
            )

        for row in reader:
            try:
                if int(row["trial"]) != trial_number:
                    continue
                samples.append(int(row["sample"]))
                for values, column in zip(channels, CHANNEL_COLUMNS):
                    values.append(float(row[column]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Invalid CSV value near data row {}".format(reader.line_num)
                ) from exc

    if not samples:
        raise ValueError("Trial {} was not found in {}".format(trial_number, csv_path.name))
    if any(b <= a for a, b in zip(samples, samples[1:])):
        raise ValueError(
            "Sample indexes for trial {} are not strictly increasing".format(trial_number)
        )

    return samples, channels


def plot_trial(csv_path, trial_number, sample_rate_hz=SAMPLE_RATE_HZ, simulate=False, speed=1.0):
    samples, channels = read_trial(csv_path, trial_number)
    time_seconds = [(sample - samples[0]) / sample_rate_hz for sample in samples]

    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8), constrained_layout=True)
    lines = []
    cursors = []
    for channel, (axis, values) in enumerate(zip(axes, channels), start=1):
        if simulate:
            (line,) = axis.plot([], [], linewidth=0.7)
            cursor = axis.axvline(time_seconds[0], color="black", alpha=0.45, linewidth=0.8)
            lines.append(line)
            cursors.append(cursor)
            axis.set_xlim(time_seconds[0], time_seconds[-1])
            low, high = min(values), max(values)
            padding = (high - low) * 0.08 or 0.01
            axis.set_ylim(low - padding, high + padding)
        else:
            axis.plot(time_seconds, values, linewidth=0.7)
        axis.set_ylabel("Channel {}".format(channel))
        axis.grid(alpha=0.25)

    duration = len(samples) / sample_rate_hz
    figure.suptitle(
        "{} | trial {} | {} samples | {:.3f} s".format(
            csv_path.name, trial_number, len(samples), duration
        )
    )
    axes[-1].set_xlabel("Time from trial start (s), {} Hz".format(sample_rate_hz))

    if simulate:
        display_fps = 30
        samples_per_frame = max(1, int(sample_rate_hz * speed / display_fps))
        frame_count = (len(samples) + samples_per_frame - 1) // samples_per_frame
        status = figure.text(0.5, 0.01, "", ha="center")

        def update(frame):
            end = min((frame + 1) * samples_per_frame, len(samples))
            for line, cursor, values in zip(lines, cursors, channels):
                line.set_data(time_seconds[:end], values[:end])
                cursor.set_xdata([time_seconds[end - 1], time_seconds[end - 1]])
            status.set_text(
                "Sample {} / {} | t = {:.3f} s | playback {:.2f}x".format(
                    end, len(samples), time_seconds[end - 1], speed
                )
            )
            return tuple(lines + cursors + [status])

        animation = FuncAnimation(
            figure,
            update,
            frames=frame_count,
            interval=1000.0 / display_fps,
            blit=False,
            repeat=False,
        )
        figure._sampling_animation = animation

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV path; omit to open a file picker")
    parser.add_argument("--trial", type=int, help="1-based repetition/trial number")
    parser.add_argument("--fs", type=int, default=SAMPLE_RATE_HZ, help="sampling rate in Hz")
    parser.add_argument(
        "--simulate", action="store_true", help="replay samples over time with a moving cursor"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="simulation speed multiplier; default is real-time"
    )
    args = parser.parse_args()

    csv_path = args.csv or choose_csv()
    if csv_path is None:
        print("No CSV selected; cancelled.")
        return
    if not csv_path.is_file():
        parser.error("CSV file does not exist: {}".format(csv_path))

    trial_number = args.trial
    if trial_number is None:
        while True:
            entered = input("Trial/repetition number: ").strip()
            try:
                trial_number = int(entered)
                if trial_number < 1:
                    raise ValueError
                break
            except ValueError:
                print("Enter a positive whole number.")

    if args.fs <= 0:
        parser.error("--fs must be positive")
    if args.speed <= 0:
        parser.error("--speed must be positive")

    try:
        plot_trial(csv_path, trial_number, args.fs, args.simulate, args.speed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
