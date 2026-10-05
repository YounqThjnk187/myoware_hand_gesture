# Step 5 – Min-Max Normalization

**Script:** `normalizer.py` (the `MinMaxNormalizer` class)  
**Run:** there is no separate command. The normalizer is fitted inside Step 6 (`python -m pipeline.step6_splitting.split_and_normalize`), because the min/max values must come from the **training split only**, and that split does not exist until Step 6.

| | |
|---|---|
| Input | the training features from Step 6 |
| Output | `artifacts/normalizer_profile.json` (`min`, `max`, `feature_names`), which Step 9 exports to C |

## Formula

$$z_k = \operatorname{clip}\!\left(\frac{x_k - x_k^{min}}{x_k^{max}-x_k^{min}},\ 0,\ 1\right)$$

$x_k^{min}$ and $x_k^{max}$ are computed per feature on the training set. If a feature is constant (span = 0), it is divided by 1 instead of 0.

## Why normalize

The features have very different ranges: ZC is a count from 0 to ~100, while MAV is about 0.01–1 V. KNN and SVM-RBF compute Euclidean distances, so without scaling the ZC features would dominate. LDA and RF are scale-invariant, but they are normalized too so the firmware runs one identical path for every model.

## Rules

1. **Fit on train, apply everywhere.** Fitting on all data leaks test-set information: the scaling would already "know" the test extremes, and accuracy would be optimistic.
2. **Clip at inference.** A stronger contraction than any in training gives $z > 1$. Without clipping, distance-based models extrapolate unpredictably.
3. **Save the profile.** The same `min`/`max` must be used on the MCU. Step 9 embeds them as `EMG_FEAT_MIN/MAX`.
4. **Recalibrate for new hardware or users.** A different gain, electrode placement or skin impedance shifts the feature ranges. A short calibration session (record every gesture, refit only `min`/`max`) is the cheapest way to adapt.

## Min-Max vs. Z-score

Min-Max has a fixed [0, 1] output, which is convenient for fixed-point or int8 quantization on MCUs (Step 9 / TFLite Micro). It is, however, sensitive to single outliers. Step 1 removes outlier repetitions first. If it is still a problem, switch to the 1st/99th percentiles instead of min/max.
