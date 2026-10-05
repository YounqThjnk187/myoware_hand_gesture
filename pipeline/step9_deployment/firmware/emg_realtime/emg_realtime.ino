/*
 * Real-time sEMG gesture recognition: 3 x MyoWare 2.0 (RAW output) -> A0, A1, A2 at 1000 Hz.
 * Target: Arduino UNO R4 / ESP32 / any board with >= 8 KB RAM (an AVR UNO R3 has only 2 KB).
 * Generate emg_model.h first:  python -m pipeline.step9_deployment.export_c
 * Serial output per decision (every 100 ms): label,compute_us
 */
#include "emg_pipeline.h"

static const uint8_t EMG_PINS[EMG_NUM_CHANNELS] = {A0, A1, A2};
static const float ADC_VREF = 5.0f;
static const float ADC_MAX = 1023.0f;
// Counts -> volts. The dataset gain differs from MyoWare's, so recalibrate the Min-Max profile on your own data.
static const float ADC_TO_SIGNAL = ADC_VREF / ADC_MAX;
static const unsigned long SAMPLE_INTERVAL_US = 1000000UL / EMG_FS_HZ;

static EmgPipeline pipeline;
static unsigned long nextSampleUs;

void setup() {
  Serial.begin(115200);
  emg_init(&pipeline);
  nextSampleUs = micros();
}

void loop() {
  if ((long)(micros() - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_INTERVAL_US;

  float x[EMG_NUM_CHANNELS];
  for (uint8_t c = 0; c < EMG_NUM_CHANNELS; ++c) {
    x[c] = analogRead(EMG_PINS[c]) * ADC_TO_SIGNAL;
  }

  if (emg_push_sample(&pipeline, x)) {
    const unsigned long t0 = micros();
    const int label = emg_classify(&pipeline, NULL, NULL);
    const unsigned long dt = micros() - t0;
    Serial.print(EMG_CLASS_NAMES[label]);
    Serial.print(',');
    Serial.println(dt);
  }
}
