import serial
import time
import pandas as pd

# Thay tên cổng đúng với máy Mac của bạn (ví dụ: '/dev/cu.usbmodem1101')
SERIAL_PORT = '/dev/cu.usbmodem1101' 
BAUD_RATE = 115200

# Chỉ thu thập 2 cử chỉ để test
GESTURES = {
    '0': 'Rest',    # Thả lỏng
    '1': 'Fist'     # Nắm đấm
}

def collect_gesture_data(gesture_name, duration_sec=5):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"Lỗi kết nối Serial: {e}")
        print("Vui lòng kiểm tra lại tên cổng hoặc tắt Serial Plotter/Monitor trên Arduino IDE!")
        return []

    time.sleep(2) # Chờ kết nối Serial ổn định
    
    data = []
    print(f"\n---> CHUẨN BỊ ĐO CỬ CHỈ: [{gesture_name.upper()}] <---")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print(">>> BẮT ĐẦU! (Giữ nguyên cử chỉ trong 5 giây) <<<")
    
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            if len(parts) == 2 and parts[0].isdigit():
                ts, raw = int(parts[0]), int(parts[1])
                data.append([ts, raw, gesture_name])
                
    ser.close()
    print(f"✓ Đã thu thập {len(data)} mẫu cho [{gesture_name}]")
    return data

# Chạy thu thập lần lượt
all_dataset = []
for code, name in GESTURES.items():
    input(f"\nẤn ENTER để bắt đầu đo cử chỉ [{name}]...")
    gesture_data = collect_gesture_data(name, duration_sec=5)
    all_dataset.extend(gesture_data)

if all_dataset:
    # Lưu ra file CSV
    df = pd.DataFrame(all_dataset, columns=['Timestamp_ms', 'RAW_ADC', 'Label'])
    df.to_csv('emg_raw_dataset.csv', index=False)
    print("\n==========================================")
    print("=> THÀNH CÔNG: Đã lưu dữ liệu vào 'emg_raw_dataset.csv'")
    print("==========================================")