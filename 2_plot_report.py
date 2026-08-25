import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Đọc dữ liệu CSV vừa tạo
    df = pd.read_csv('emg_raw_dataset.csv')
except Exception as e:
    print("Chưa tìm thấy file 'emg_raw_dataset.csv'. Hãy chạy file 1_collect_data.py trước!")
    exit()

# Cấu hình giao diện đồ thị
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. ĐỒ THỊ CHUỖI THỜI GIAN (TIME-SERIES)
plt.figure(figsize=(10, 4.5))
plt.plot(df['RAW_ADC'], color='#1f77b4', linewidth=1.2, label='Tín hiệu EMG RAW')

# Tô màu phân vùng Rest và Fist
colors = {'Rest': '#2ecc71', 'Fist': '#e74c3c'}
for label in df['Label'].unique():
    sub_df = df[df['Label'] == label]
    start_idx = sub_df.index[0]
    end_idx = sub_df.index[-1]
    plt.axvspan(start_idx, end_idx, color=colors.get(label, 'gray'), alpha=0.2, label=f'Cử chỉ: {label}')

plt.title('DẠNG SÓNG TÍN HIỆU EMG RAW (REST vs FIST)', fontsize=12, fontweight='bold')
plt.xlabel('Chỉ số mẫu (Sample Index - 100Hz)', fontsize=10)
plt.ylabel('Biên độ RAW ADC (0 - 1023 LSB)', fontsize=10)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('emg_waveform.png', dpi=300)
plt.close()

# 2. BIỂU ĐỒ HỘP (BOXPLOT) SO SÁNH ĐỘ PHÂN TÁCH
plt.figure(figsize=(6, 4.5))
sns.boxplot(x='Label', y='RAW_ADC', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('SO SÁNH PHÂN BỐ BIÊN ĐỘ GIỮA THẢ LỎNG VÀ NẮM ĐẤM', fontsize=11, fontweight='bold')
plt.xlabel('Cử chỉ', fontsize=10)
plt.ylabel('Biên độ RAW ADC (LSB)', fontsize=10)
plt.tight_layout()
plt.savefig('emg_boxplot.png', dpi=300)
plt.close()

print("=> THÀNH CÔNG: Đã xuất 2 file ảnh 'emg_waveform.png' và 'emg_boxplot.png'!")