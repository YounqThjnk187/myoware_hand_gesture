/*
 * EMG Filtered Data Streamer for KIT0012 Sensor
 * Algorithm: Moving Average Filter (Window Size = 5)
 * Sampling Rate: 100 Hz (Fixed Timer via micros)
 * Baud Rate: 115200
 * Output Format: filtered_adc
 */

#define EMG_PIN          A0
#define SAMPLE_RATE_HZ   50
#define BAUD_RATE        115200

const int WINDOW_SIZE = 5;
int readings[WINDOW_SIZE];  // Bộ đệm vòng lưu 5 mẫu gần nhất
int readIndex = 0;          // Vị trí con trỏ đệm
long total = 0;             // Tổng giá trị trong cửa sổ trượt
float filteredVal = 0;      // Giá trị sau lọc

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ; // 10,000 us (10ms)

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(EMG_PIN, INPUT);

  // Khởi tạo mảng bộ đệm ban đầu bằng 0
  for (int i = 0; i < WINDOW_SIZE; i++) {
    readings[i] = 0;
  }

  // Chờ kết nối Serial mở hoàn toàn
  while (!Serial) { delay(10); }
}

void loop() {
  unsigned long now = micros();

  // Định thời cứng 100Hz triệt tiêu jitter
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime += SAMPLE_INTERVAL_US;

    // 1. Trừ giá trị cũ nhất ra khỏi tổng
    total = total - readings[readIndex];

    // 2. Đọc giá trị RAW từ cảm biến tại chân A0
    readings[readIndex] = analogRead(EMG_PIN);

    // 3. Cộng giá trị mới vào tổng
    total = total + readings[readIndex];

    // 4. Cập nhật vị trí con trỏ đệm vòng
    readIndex = readIndex + 1;
    if (readIndex >= WINDOW_SIZE) {
      readIndex = 0;
    }

    // 5. Tính trung bình trượt
    filteredVal = (float)total / WINDOW_SIZE;

    // 6. Xuất duy nhất giá trị ADC sau lọc
    Serial.println((int)filteredVal);
  }
}