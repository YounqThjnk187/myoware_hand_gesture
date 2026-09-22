/*
 * Single-Channel EMG Reader with DSP Filters (Notch + Low-Pass)
 * Designed for KIT0012 Envelope Signal (Fs = 100 Hz)
 */

#define EMG_PIN          A0
#define SAMPLE_RATE_HZ   100     // ADC 10-bit lấy mẫu đúng 100Hz
#define BAUD_RATE        115200

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

// Biến lưu trạng thái cho Bộ lọc Notch IIR (50Hz @ Fs=100Hz)
float notch_x1 = 0, notch_x2 = 0;
float notch_y1 = 0, notch_y2 = 0;

// Biến lưu trạng thái cho Bộ lọc Low-Pass IIR (10Hz Cutoff, Bậc 2)
float lp_x1 = 0, lp_x2 = 0;
float lp_y1 = 0, lp_y2 = 0;

int16_t baseline = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(EMG_PIN, INPUT);

  // Cân chỉnh Baseline tự động 1 giây đầu
  long sum = 0;
  for (int i = 0; i < 100; i++) {
    sum += analogRead(EMG_PIN);
    delay(10);
  }
  baseline = sum / 100;
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime += SAMPLE_INTERVAL_US;

    // 1. Đọc tín hiệu ADC 10-bit raw từ cảm biến
    int16_t rawADC = analogRead(EMG_PIN);
    
    // Trừ baseline để đưa về mốc 0 khi thả lỏng
    float inputSignal = abs(rawADC - baseline);

    // 2. TẦNG LỌC SỐ 1: Notch Filter IIR (Triệt nhiễu 50Hz)
    // Phương trình sai phân cho Notch 50Hz tại Fs=100Hz: y[n] = x[n] + x[n-2]
    float notchOut = inputSignal + notch_x2;
    notch_x2 = notch_x1;
    notch_x1 = inputSignal;

    // 3. TẦNG LỌC SỐ 2: Butterworth Low-Pass IIR (10Hz Cutoff, Bậc 2)
    // Hệ số IIR Butterworth bão hòa độ mượt, triệt tiêu dao động cơ học
    float b0 = 0.067455, b1 = 0.134910, b2 = 0.067455;
    float a1 = -1.142981, a2 = 0.412802;

    float filteredSignal = b0 * notchOut + b1 * lp_x1 + b2 * lp_x2 - a1 * lp_y1 - a2 * lp_y2;

    // Cập nhật trạng thái bộ lọc Low-Pass
    lp_x2 = lp_x1;
    lp_x1 = notchOut;
    lp_y2 = lp_y1;
    lp_y1 = filteredSignal;

    // Khóa trần dải 0 - 1023
    int outputVal = (int)filteredSignal;
    if (outputVal < 0) outputVal = 0;
    if (outputVal > 1023) outputVal = 1023;

    // Output hiển thị trên Serial Plotter
    Serial.print("Min:0,Max:1023,EMG_Filtered:");
    Serial.println(outputVal);
  }
}