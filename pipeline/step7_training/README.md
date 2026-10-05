# Step 7 – Model Training & Hyperparameter Tuning

**Script:** `train.py`  
**Run:** `python -m pipeline.step7_training.train [--models lda knn svm rf] [--max-tune-samples 20000] [--jobs -1] [--show]`

| | |
|---|---|
| Input | `artifacts/step6_dataset.npz` |
| Output | `artifacts/models/{lda,knn,svm,rf}.joblib`, `artifacts/step7_summary.json`, `artifacts/figures/step7_learning_curves.png` |

## Models and search grids

| Model | Grid | Notes |
|---|---|---|
| LDA (`lsqr`) | shrinkage ∈ {None, auto} | Linear, $O(KF)$ inference (5×12 MACs). **Default model for the MCU** (Step 9) |
| KNN | k ∈ {3,5,7,9}, weights ∈ {uniform, distance} | No real training, but inference needs the whole training set: too large for an MCU |
| SVM (RBF) | C ∈ {1,10,100}, γ ∈ {scale, 0.1, 1} | Often the most accurate; inference cost is proportional to the number of support vectors |
| Random Forest | trees ∈ {50,100}, depth ∈ {None,10,20}, min leaf ∈ {1,5} | Robust; flash cost grows with trees × nodes |

## Procedure

1. **Grid search with 5-fold `StratifiedGroupKFold`.** Folds are stratified by gesture and grouped by repetition, so overlapping windows of one repetition never sit on both sides of a fold. The score is **macro-F1**.
2. **Tuning uses a subsample** (default 20 000 windows, set with `--max-tune-samples`). RBF-SVM training is roughly $O(n^2)$ and KNN prediction is $O(n)$ per query, so a full grid on about 60 000 windows is slow. The best configuration is then **refit on the full training set**.
3. **Validation score.** Accuracy and macro-F1 on the validation split pick the best model (`_best_model` in the summary).
4. **Overfitting checks:**
   * the **train-fold vs CV-fold gap** for the best parameters;
   * **learning curves** of train and CV macro-F1 against the number of training windows.

## Reading the learning curves

| Pattern | Diagnosis | Action |
|---|---|---|
| Train ≈ 1.0, CV much lower, gap not closing | Overfitting (typical: RF with depth None, KNN with k = 1–3) | Limit depth, raise min leaf / k, lower C |
| Train ≈ CV, both low | Underfitting / not enough information in the features | More expressive model or features (SSC, WL ratios, 1D-CNN) |
| CV still rising at the right edge | More data would help | Record more repetitions |

## The 1D-CNN option

The brief allows a 1D-CNN on the raw windows (`W` from Step 3, shape `(3, 200)`). It is not included here because it would add a PyTorch/TensorFlow dependency. The same split, early stopping on the validation loss, and train/val loss curves apply. The deployment path would be TFLite Micro with int8 quantization instead of Step 9's generated header.
