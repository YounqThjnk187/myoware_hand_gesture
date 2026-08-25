/*
 * EMG RAW Data Logger for KIT0012 Sensor
 * Sample Rate: 100 Hz (Fixed Timer via micros)
 * Output Format: timestamp_ms, raw_adc
 */

#define EMG_PIN          A0
#define SAMPLE_RATE_HZ   100
#define BAUD_RATE        115200

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(EMG_PIN, INPUT);
  
  // Chờ kết nối Serial mở
  while (!Serial) { delay(10); }
}

void loop() {
  unsigned long now = micros();
  
  // Lấy mẫu định thời cứng 100Hz (triệt tiêu jitter)
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime += SAMPLE_INTERVAL_US;

    unsigned long ts = millis();
    int16_t rawVal = (int16_t)analogRead(EMG_PIN);

    // Xuất dữ liệu định dạng CSV: timestamp, raw_val
    Serial.print(ts);
    Serial.print(',');
    Serial.println(rawVal);
  }
}