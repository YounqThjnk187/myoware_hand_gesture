"""Record unmodified EMG packets and display them in a monitoring graph."""

import csv
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
REST_TIME_SEC = 5
PLOT_REFRESH_MS = 50
PLOT_BUFFER_SIZE = 2000
RECORDINGS_ROOT = Path(__file__).resolve().parent / 'recordings'


class RawEmgRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title('RAW EMG Data Recorder')
        self.root.geometry('900x620')

        self.serial_port = None
        self.recording_thread = None
        self.stop_event = threading.Event()
        self.plot_samples = deque(maxlen=PLOT_BUFFER_SIZE)
        self.plot_lock = threading.Lock()
        self.csv_file = None
        self.csv_writer = None
        self.sample_count = 0
        self.malformed_count = 0
        self.error_message = ''
        self.recording_pending = False
        self.countdown_after_id = None
        self.countdown_remaining = 0
        self.session_dir = RECORDINGS_ROOT / datetime.now().strftime('session_%Y%m%d_%H%M%S')
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.port_var = tk.StringVar(value=SERIAL_PORT)
        self.rest_time_var = tk.IntVar(value=REST_TIME_SEC)
        self.status_var = tk.StringVar(value='Ready')
        self.stats_var = tk.StringVar(value='Samples: 0 | Malformed packets: 0')

        self._build_controls()
        self._build_plot()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(PLOT_REFRESH_MS, self._refresh_plot)

    def _build_controls(self):
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text='Serial port:').pack(side=tk.LEFT)
        ttk.Entry(controls, textvariable=self.port_var, width=12).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Label(controls, text='Rest before recording (s):').pack(side=tk.LEFT)
        ttk.Spinbox(controls, from_=0, to=60, textvariable=self.rest_time_var, width=5).pack(
            side=tk.LEFT, padx=(6, 12)
        )

        self.start_button = ttk.Button(controls, text='START RECORDING', command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=4)
        self.stop_button = ttk.Button(controls, text='STOP RECORDING', command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=4)

        ttk.Label(self.root, textvariable=self.status_var, padding=(10, 0)).pack(anchor=tk.W)
        ttk.Label(self.root, textvariable=self.stats_var, padding=(10, 0)).pack(anchor=tk.W)

    def _build_plot(self):
        figure = Figure(figsize=(9, 5), dpi=100)
        self.axis = figure.add_subplot(111)
        self.axis.set_title('Raw EMG monitor')
        self.axis.set_xlabel('Sample index')
        self.axis.set_ylabel('Raw EMG value')
        self.axis.grid(True, alpha=0.3)
        self.line, = self.axis.plot([], [], color='#1f77b4', linewidth=1)

        self.canvas = FigureCanvasTkAgg(figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def start_recording(self):
        if self.recording_pending or (self.recording_thread and self.recording_thread.is_alive()):
            return

        try:
            rest_time_sec = max(0, int(self.rest_time_var.get()))
        except (TypeError, ValueError):
            messagebox.showerror('Invalid rest time', 'Enter a whole number from 0 to 60 seconds.')
            return

        port_name = self.port_var.get().strip()
        try:
            self.serial_port = serial.Serial(port_name, BAUD_RATE, timeout=0.1)
        except serial.SerialException as error:
            messagebox.showerror('Serial connection error', str(error))
            self.status_var.set('Connection failed')
            return

        recording_name = datetime.now().strftime('trial_%Y%m%d_%H%M%S.csv')
        recording_path = self.session_dir / recording_name
        try:
            self.csv_file = recording_path.open('w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['timestamp', 'sample_index', 'raw_emg'])
            self.csv_file.flush()
        except OSError as error:
            self.serial_port.close()
            self.serial_port = None
            messagebox.showerror('File error', str(error))
            return

        self.sample_count = 0
        self.malformed_count = 0
        self.error_message = ''
        with self.plot_lock:
            self.plot_samples.clear()
        self.stop_event.clear()
        self.recording_pending = True
        self.countdown_remaining = rest_time_sec
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set(f'Rest before recording: {rest_time_sec} seconds')
        self._run_rest_countdown(recording_path)

    def _run_rest_countdown(self, recording_path):
        if not self.recording_pending:
            return
        if self.countdown_remaining > 0:
            self.status_var.set(f'Rest before recording: {self.countdown_remaining} seconds')
            self.countdown_remaining -= 1
            self.countdown_after_id = self.root.after(
                1000, lambda: self._run_rest_countdown(recording_path)
            )
            return

        try:
            self.serial_port.reset_input_buffer()
        except (serial.SerialException, OSError) as error:
            self.error_message = str(error)
            self.recording_pending = False
            self._close_recording_resources()
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.status_var.set(f'Connection failed: {error}')
            return

        self.recording_pending = False
        self.recording_thread = threading.Thread(target=self._read_serial, daemon=True)
        self.recording_thread.start()
        self.status_var.set(f'Recording to {recording_path}')

    def stop_recording(self):
        if self.recording_pending:
            self.recording_pending = False
            if self.countdown_after_id is not None:
                self.root.after_cancel(self.countdown_after_id)
                self.countdown_after_id = None
            self._close_recording_resources()
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.status_var.set('Recording cancelled during rest period.')
            return

        if not self.recording_thread:
            return

        self.stop_event.set()
        self.recording_thread.join(timeout=1.0)
        if self.recording_thread.is_alive():
            self.status_var.set('Stopping serial reader...')
            self.root.after(100, self.stop_recording)
            return

        self._close_recording_resources()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if self.error_message:
            self.status_var.set(f'Stopped with error: {self.error_message}')
        else:
            self.status_var.set(f'Stopped. Saved {self.sample_count} samples.')

    def _read_serial(self):
        sample_index = 0
        try:
            while not self.stop_event.is_set():
                line = self.serial_port.readline()
                if not line:
                    continue

                try:
                    decoded = line.decode('ascii').strip()
                    fields = decoded.split(',')
                    if len(fields) != 2:
                        raise ValueError('expected timestamp,raw_emg')
                    timestamp = int(fields[0])
                    raw_emg = int(fields[1])
                except (UnicodeDecodeError, ValueError):
                    self.malformed_count += 1
                    continue

                self.csv_writer.writerow([timestamp, sample_index, raw_emg])
                self.csv_file.flush()
                self.sample_count += 1
                sample_index += 1

                with self.plot_lock:
                    self.plot_samples.append((sample_index - 1, raw_emg))
        except (serial.SerialException, OSError) as error:
            self.error_message = str(error)
            self.stop_event.set()

    def _refresh_plot(self):
        with self.plot_lock:
            samples = list(self.plot_samples)

        if samples:
            indices, values = zip(*samples)
            self.line.set_data(indices, values)
            self.axis.set_xlim(max(0, indices[-1] - PLOT_BUFFER_SIZE), max(PLOT_BUFFER_SIZE, indices[-1]))
            lower = min(values)
            upper = max(values)
            margin = max(1, (upper - lower) * 0.05)
            self.axis.set_ylim(lower - margin, upper + margin)
            self.canvas.draw_idle()

        self.stats_var.set(
            f'Samples: {self.sample_count} | Malformed packets: {self.malformed_count}'
        )
        if self.recording_thread and not self.recording_thread.is_alive() and self.stop_button['state'] == tk.NORMAL:
            self.stop_recording()
        self.root.after(PLOT_REFRESH_MS, self._refresh_plot)

    def _close_recording_resources(self):
        if self.serial_port is not None:
            self.serial_port.close()
            self.serial_port = None
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def close(self):
        self.recording_pending = False
        if self.countdown_after_id is not None:
            self.root.after_cancel(self.countdown_after_id)
        if self.recording_thread and self.recording_thread.is_alive():
            self.stop_event.set()
            self.recording_thread.join(timeout=1.0)
        self._close_recording_resources()
        self.root.destroy()


if __name__ == '__main__':
    app_root = tk.Tk()
    RawEmgRecorder(app_root)
    app_root.mainloop()