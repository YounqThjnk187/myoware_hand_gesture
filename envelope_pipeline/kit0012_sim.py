"""Simulate the KIT0012 analog envelope from raw 1 kHz sEMG and quantize it like the Uno ADC."""

import numpy as np
from scipy import signal

from envelope_pipeline import config

_HP = signal.butter(2, config.SIM_HIGHPASS_HZ, "highpass", fs=config.RAW_FS, output="sos")
_LP = signal.butter(2, config.SIM_ENVELOPE_HZ, "lowpass", fs=config.RAW_FS, output="sos")


def raw_to_envelope(x):
    """(..., channels, time) raw at RAW_FS -> envelope at FS with the filter transient removed."""
    y = signal.sosfilt(_LP, np.abs(signal.sosfilt(_HP, x, axis=-1)), axis=-1)
    start = int(config.SIM_TRANSIENT_S * config.RAW_FS)
    return y[..., start :: config.RAW_FS // config.FS]


def to_adc(env, scale):
    """Apply a per-channel gain and quantize to 10-bit ADC counts."""
    return np.clip(np.rint(env * np.asarray(scale)[:, None]), 0, config.ADC_MAX).astype(np.int16)
