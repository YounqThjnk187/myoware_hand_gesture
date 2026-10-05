#include "emg_pipeline.h"

#include <math.h>
#include <string.h>

void emg_init(EmgPipeline *p) { memset(p, 0, sizeof(*p)); }

/* Direct-Form II Transposed biquad cascade; identical to scipy.signal.sosfilt. */
static float biquad_cascade(float x, float state[EMG_NUM_SOS][2]) {
  for (int s = 0; s < EMG_NUM_SOS; ++s) {
    const float *c = EMG_SOS[s];
    float y = c[0] * x + state[s][0];
    state[s][0] = c[1] * x - c[4] * y + state[s][1];
    state[s][1] = c[2] * x - c[5] * y;
    x = y;
  }
  return x;
}

int emg_push_sample(EmgPipeline *p, const float x[EMG_NUM_CHANNELS]) {
  for (int c = 0; c < EMG_NUM_CHANNELS; ++c) {
    if (p->samples_seen == 0) p->baseline[c] = x[c];
    p->baseline[c] += EMG_BASELINE_ALPHA * (x[c] - p->baseline[c]);
    p->window[c][p->write_index] = biquad_cascade(x[c] - p->baseline[c], p->sos_state[c]);
  }
  p->write_index = (uint16_t)((p->write_index + 1) % EMG_WINDOW_SIZE);
  p->samples_seen++;

  const uint32_t first = EMG_TRANSIENT_SAMPLES + EMG_WINDOW_SIZE;
  if (p->samples_seen < first) return 0;
  return ((p->samples_seen - first) % EMG_WINDOW_STEP) == 0;
}

int emg_classify(const EmgPipeline *p, float features_out[EMG_NUM_FEATURES], float scores_out[EMG_NUM_CLASSES]) {
  const int n = EMG_WINDOW_SIZE;
  const int nch = EMG_NUM_CHANNELS;
  float f[EMG_NUM_FEATURES];

  for (int c = 0; c < nch; ++c) {
    const float *w = p->window[c];
    int idx = p->write_index; /* oldest sample */
    float prev = w[idx];
    float sum_abs = fabsf(prev), sum_sq = prev * prev, wl = 0.0f;
    int zc = 0;
    for (int i = 1; i < n; ++i) {
      if (++idx == n) idx = 0;
      const float cur = w[idx];
      const float d = cur - prev;
      sum_abs += fabsf(cur);
      sum_sq += cur * cur;
      wl += fabsf(d);
      if (prev * cur < 0.0f && fabsf(d) >= EMG_ZC_THRESHOLD) ++zc;
      prev = cur;
    }
    f[0 * nch + c] = sum_abs / n;
    f[1 * nch + c] = sqrtf(sum_sq / n);
    f[2 * nch + c] = wl;
    f[3 * nch + c] = (float)zc;
  }

  float z[EMG_NUM_FEATURES];
  for (int k = 0; k < EMG_NUM_FEATURES; ++k) {
    const float span = EMG_FEAT_MAX[k] - EMG_FEAT_MIN[k];
    float v = (f[k] - EMG_FEAT_MIN[k]) / (span > 0.0f ? span : 1.0f);
    z[k] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    if (features_out) features_out[k] = f[k];
  }

  int best = 0;
  float best_score = 0.0f;
  for (int k = 0; k < EMG_NUM_CLASSES; ++k) {
    float s = EMG_B[k];
    for (int j = 0; j < EMG_NUM_FEATURES; ++j) s += EMG_W[k][j] * z[j];
    if (scores_out) scores_out[k] = s;
    if (k == 0 || s > best_score) {
      best = k;
      best_score = s;
    }
  }
  return best;
}
