/*
 * EMG RAW Data Collector for KIT0012 / MyoWare Sensor
 * Sampling Rate: 100 Hz (Fixed Interval = 10,000 us)
 * Baud Rate: 115200
 * Output Format: Single Raw ADC Value (0 - 1023)
 */

#define EMG_PIN          A1       // Chân Analog cắm tín hiệu SIG từ sensor
#define SAMPLE_RATE_HZ   50      // Tần số lấy mẫu 100Hz
#define BAUD_RATE        9600   // Tốc độ truyền Serial

unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ; // 10,000 microseconds

void setup() {
  // Khởi tạo cổng Serial tốc độ cao
  Serial.begin(BAUD_RATE);
  pinMode(EMG_PIN, INPUT);

  // Chờ cổng Serial mở hoàn toàn (đặc biệt cần thiết cho Mac/Linux)
  while (!Serial) { 
    delay(10); 
  }
}

void loop() {
  unsigned long now = micros();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_US) {
    lastSampleTime += SAMPLE_INTERVAL_US;

    int rawVal = analogRead(EMG_PIN);
    
    // Nếu tín hiệu sát 0, ta có thể cộng thêm một mức Offset nền (ví dụ +50)
    // để đồ thị nâng lên giữa màn hình cho dễ nhìn
    int visualVal = rawVal; 

    Serial.println(visualVal);
  }
}