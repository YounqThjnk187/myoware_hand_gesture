#ifndef EMG_PIPELINE_H
#define EMG_PIPELINE_H

#include <stdint.h>

#include "emg_model.h"

typedef struct {
  float baseline[EMG_NUM_CHANNELS];
  float sos_state[EMG_NUM_CHANNELS][EMG_NUM_SOS][2];
  float window[EMG_NUM_CHANNELS][EMG_WINDOW_SIZE]; /* ring buffer of filtered samples */
  uint16_t write_index;                            /* next slot = oldest sample */
  uint32_t samples_seen;
} EmgPipeline;

void emg_init(EmgPipeline *p);

/* Push one sample per channel (same units as the training data). Returns 1 when a new window is ready. */
int emg_push_sample(EmgPipeline *p, const float x[EMG_NUM_CHANNELS]);

/* Features -> Min-Max -> linear classifier on the current window. Outputs are optional (may be NULL). */
int emg_classify(const EmgPipeline *p, float features_out[EMG_NUM_FEATURES], float scores_out[EMG_NUM_CLASSES]);

#endif /* EMG_PIPELINE_H */
