# Step 6 – Stratified Dataset Splitting (+ fitting Step 5)

**Script:** `split_and_normalize.py`  
**Run:** `python -m pipeline.step6_splitting.split_and_normalize [--by repetition|subject] [--seed 42]`

| | |
|---|---|
| Input | `artifacts/step4_features.npz` |
| Output | `artifacts/step6_dataset.npz`: `X_/y_/rep_/subject_` × `train/val/test`, with `X` already normalized; `artifacts/normalizer_profile.json` |

## Procedure

1. **Split whole groups, not windows.** Every window keeps its repetition id (`rep_uid`) and subject id.
   * `--by repetition` (default): the repetitions kept by Step 1 are split 70/15/15 with `train_test_split(..., stratify=gesture)`, done twice (70 / 30, then 30 → 15 / 15). Every gesture keeps the same share in each split.
   * `--by subject`: the 20 subjects are split 14 / 3 / 3. Since every subject performs every gesture, the classes stay balanced automatically.
2. **Check for leaks.** An assertion verifies that no repetition appears in two splits.
3. **Fit Min-Max on the training windows only** (Step 5), then normalize train, validation and test with that profile.

## Why grouping matters

With 50 % overlap, neighbouring windows share 100 samples. A random split at the window level puts almost-identical windows in both train and test. The model then partly *memorizes* instead of generalizing, and test accuracy looks inflated, often by 5–15 points on EMG data.

| Split mode | What the test score means |
|---|---|
| repetition | "User-dependent": subjects seen in training, new repetitions |
| subject | "User-independent": a completely new person. This is closest to deploying a pre-trained model on your own arm |

Report the subject-split result as the realistic estimate. Expect it to be clearly lower, because of differences in anatomy, electrode placement and skin impedance between people.

## Roles of the three sets

* **Train (70 %)**: fitting model weights, the Min-Max profile, and the 5-fold CV used for hyperparameter tuning (Step 7).
* **Validation (15 %)**: choosing between models and checking for overfitting.
* **Test (15 %)**: used **once**, in Step 8. Never tune anything based on it.
