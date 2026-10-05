# Step 8 – Evaluation & End-to-End Latency Measurement

**Script:** `evaluate.py`  
**Run:** `python -m pipeline.step8_evaluation.evaluate [--models lda svm] [--latency-reps 20] [--show]`

| | |
|---|---|
| Input | `artifacts/step6_dataset.npz`, `artifacts/models/*.joblib`, `artifacts/normalizer_profile.json`, `artifacts/step1_clean.npz` (raw test repetitions, for streaming) |
| Output | `artifacts/step8_results.json`, `artifacts/figures/step8_confusion_matrices.png` |

## Classification metrics (test split, used only here)

For each gesture class $k$:

$$P_k=\frac{TP_k}{TP_k+FP_k},\qquad R_k=\frac{TP_k}{TP_k+FN_k},\qquad F1_k=\frac{2P_kR_k}{P_k+R_k},\qquad \text{macro-F1}=\frac1K\sum_k F1_k$$

Macro-F1 gives every gesture the same weight, so a model cannot score well by getting only the "easy" gestures right. The **row-normalized confusion matrix** shows recall on the diagonal and highlights which gestures get confused. Likely pairs are Fist ↔ Middle+ring flexion, because both are flexor-dominated (see the Step 1 channel table).

## Latency measurement

`measure_latency` replays raw **test** repetitions the way the device receives them: in 100-sample hops, with a persistent filter state (`StreamingFilter`) and a 200-sample ring buffer. For every decision it times:

| Stage | Work per decision |
|---|---|
| filter | 5 biquads × 3 channels × 100 new samples |
| buffer | shift in the new 100 samples |
| features | MAV, RMS, WL, ZC on 3 × 200 samples |
| normalize | 12 subtractions and divisions + clip |
| inference | `model.predict` on one 12-D vector |

The script reports mean, p95 and **max** for each stage. The budget check uses the max, because real-time guarantees depend on the worst case. The first 5 decisions are discarded as warm-up.

### Interpreting the numbers

* The Python timings include interpreter and NumPy call overhead (tens of µs per call). They are an **upper bound on the algorithm cost**, not MCU timings. The firmware in Step 9 prints its own `micros()` measurement.
* **Total latency perceived by the user** ≈ computation time + waiting for the next hop (≤ 100 ms) + filter group delay (printed; a few ms). Because of the window length, the decision also reflects about 200 ms of past signal.
* KNN and SVM inference get slower as the training set grows. LDA and RF stay roughly constant.

## Honest reporting

* Say whether the split was by `repetition` or by `subject` (stored as `split_by` in the results). The two answer different questions.
* Expect lower accuracy on your own MyoWare hardware than on this dataset. Placement, gain and sampling differ (see Steps 1 and 4).
