# Step 1 – Data Inspection & Cleaning

**Script:** `inspect_data.py`  
**Run:** `python -m pipeline.step1_data_inspection.inspect_data [--show]`

| | |
|---|---|
| Input | `dataset/kaggle/DS1_EMG_SIGNALS_AVERAGE/Mov1.mat … Mov5.mat` |
| Output | `artifacts/step1_clean.npz` (`X`, `y`, `subject`, `rep_uid`), `artifacts/step1_quality_report.csv`, `artifacts/figures/step1_*.png` |

## What it does

1. **Loads the data with the correct axes.** Each `MovK.mat` is `(400 repetitions, 3 channels, 4500 samples)`: 20 subjects × 20 reps, 4.5 s at 1000 Hz. The loader rejects any array whose time axis is not the last and longest axis, so the axis swap in the old CSVs cannot happen again. Labels are `0..4` = Fist, Thumb flexion, Rest, Finger extension, Middle+ring flexion. `subject = rep_index // 20` is an **assumption**: the dataset does not document the order.
2. **Computes a quality report per repetition:**
   * **DC offset**: the mean of each channel.
   * **RMS**: the standard deviation, i.e. RMS after removing the DC.
   * **Clipping**: plateaus, meaning consecutive identical samples in the top 10 % of the channel's amplitude range. The repetition is rejected if more than 1 % of its samples are plateau samples.
   * **Flatline / dropout**: any 50 ms block with a standard deviation below 1e-6.
   * **Non-finite values**: NaN or Inf.
   * **Outlier repetitions**: a robust z-score of **log**-RMS, computed separately for each gesture and channel, $z = 0.6745\,(x-\tilde x)/\text{MAD}$. The repetition is rejected if $|z| > 3.5$.
3. **Cleans the data:** drops the flagged repetitions and subtracts each repetition's DC offset.
4. **Makes figures:**
   * raw examples, one per gesture;
   * RMS box plots per gesture and channel;
   * the median 100 ms RMS envelope, which shows whether the gesture is held for the whole 4.5 s.

## Why

* **Median/MAD instead of mean/std:** outliers pull the mean and inflate the std, which hides them. The median and MAD are not affected this way.
* **Log-RMS:** RMS varies about 12× between subjects and is right-skewed. The log makes the distribution roughly symmetric.
* **Per gesture:** Rest is *supposed* to have low RMS. A single global threshold would reject valid Rest data.
* **Per-repetition DC removal:** this is only for offline training. On the MCU a 1 Hz EMA tracks the baseline instead: $\hat b[n] = \hat b[n-1] + \alpha(x[n]-\hat b[n-1])$ (see Step 9).

## Findings to keep in mind

* The printed "median RMS per gesture and channel" table shows **Ch3 dominates Thumb flexion and Finger extension**, while **Ch2 (posterior) stays weak during extension**. This does not match the FCR/EDC/brachioradialis placement described in the project report. Any hardware you build must reproduce the *dataset's* placement, or collect its own training data.
* Mov3 is the quietest gesture on every channel, so **Rest = Mov3**. The `EMG_DB` note that says "gesture 5 = rest" is wrong.
* The `.tdms` files contain the "correctly performed" ranges. They are not used here; they could be cross-checked with the flags above.
