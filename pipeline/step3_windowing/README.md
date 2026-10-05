# Step 3 – Sliding Window Segmentation

**Scripts:** `windowing.py` (library), `run_windowing.py` (runner)  
**Run:** `python -m pipeline.step3_windowing.run_windowing [--show]`

| | |
|---|---|
| Input | `artifacts/step2_filtered.npz` |
| Output | `artifacts/step3_windows.npz`: `W` `(M, 3, 200)`, and `y`, `subject`, `rep_uid` per window |

## Parameters

| Parameter | Value | Reason |
|---|---|---|
| Window length $L$ | 200 samples = 200 ms | Long enough for stable amplitude estimates (MAV/RMS variance ∝ 1/L); short enough to react quickly |
| Stride $S$ | 100 samples = 100 ms (50 % overlap) | The system makes a new decision every 100 ms |
| Windows per repetition | $\lfloor (T-L)/S \rfloor + 1 = \lfloor(4200-200)/100\rfloor+1 = 41$ | |

Windows are built with `numpy.lib.stride_tricks.sliding_window_view`. It is a view into the existing array, so nothing is copied until the final `reshape`.

## Latency

Latency has two parts:

1. **Algorithmic latency, fixed by the design.** A new decision is made every $S = 100$ ms. The window centre is $L/2 = 100$ ms behind the newest sample. The filter group delay adds a few ms, printed by the script.
2. **Computation time.** Filter + features + inference for one window. This is measured in Step 8 (Python) and on the MCU (Step 9). It must be far smaller than $S$, otherwise decisions start to pile up.

The ≤ 100 ms budget in the brief refers to part 2. Making $S$ smaller increases the decision rate, but it also increases CPU load.

## Pitfalls

* **Overlapping windows are highly correlated.** Adjacent windows share 50 % of their samples. If a random split puts window *k* in train and window *k+1* in test, accuracy will look better than it really is. That is why every window keeps `rep_uid` and `subject`: Step 6 splits by **repetition** (or subject), never by window.
* **Windows never cross repetition boundaries,** because segmentation is done per repetition before flattening. This means no window mixes two gestures.
* **For a 1D-CNN,** use `W` directly: `(M, 3, 200)` is the network input shape.
