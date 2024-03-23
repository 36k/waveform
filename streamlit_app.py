import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt,spectrogram
import pywt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Heiti TC']  # 设置为SimSun字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 读取Excel文件
@st.cache_data
def load_data(file_path):
    return pd.read_excel(file_path)


# 重新采样数据
def resample_data(data, sampling_rate):
    total_samples = len(data)
    sampling_interval = 1 / sampling_rate
    timestamps = np.arange(0, total_samples * sampling_interval, sampling_interval)
    data.index = pd.to_timedelta(timestamps, unit='s')
    return data


# 应用低通滤波
# 应用低通滤波
def apply_lowpass_filter(data, cutoff_freq, sampling_rate, filter_order):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data.values.squeeze())
    return pd.DataFrame(filtered_data, index=data.index)


# 平滑滤波
def smooth_filter(data, window_size=11):
    return data.rolling(window_size).mean()


# 高通滤波
def highpass_filter(data, cutoff_freq, sampling_rate):
    nyquist_freq = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data.values.squeeze())
    return pd.DataFrame(filtered_data, index=data.index)



def mad(data):
    median = np.median(data)
    return np.median(np.abs(data - median))
def wavelet_denoising(data, wavelet='db4', level=1):
    denoised_data = pd.DataFrame(index=data.index)
    for column in data.columns:
        # 进行小波变换
        coeff = pywt.wavedec(data[column], wavelet, mode='symmetric', level=level)

        # 计算阈值
        sigma = mad(coeff[-1])
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        # 应用阈值
        coeff[1:] = (pywt.threshold(i, value=threshold, mode="soft") for i in coeff[1:])

        # 重构信号
        denoised_data[column] = pywt.waverec(coeff, wavelet, mode='symmetric')

    return denoised_data


def plot_data(dataList, labels):
    plt.figure(figsize=(10, 6))
    for label, data in zip(labels, dataList):
        if isinstance(data, pd.DataFrame):
            plt.plot(data.index.total_seconds(), data.values, label=label)
        elif isinstance(data, pd.Series):
            plt.plot(data.index.total_seconds(), data.values, label=label)
        elif isinstance(data, np.ndarray):
            plt.plot(np.arange(len(data)), data, label=label)
    # plt.plot(data.index.total_seconds(), data.values, label=title)
    plt.xlabel('时间 (秒)')
    plt.ylabel('电势差')
    plt.title('电势差预处理')
    plt.grid(True)
    plt.legend()
    st.pyplot()

#时频图
def plot_spectrogram(data, sampling_rate, window_size=256, overlap=128):
    f, t, Sxx = spectrogram(data.values.squeeze(), fs=sampling_rate, window='hann', nperseg=window_size, noverlap=overlap)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(0, sampling_rate / 2)
    st.pyplot()

def main():
    st.title('电势差数据展示')

    # 上传Excel文件
    uploaded_file = st.sidebar.file_uploader("上传Excel文件", type=['xls', 'xlsx'])

    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # 用户设置采样率
        sampling_rate = st.sidebar.number_input("请输入采样率 (Hz):", min_value=0.1, max_value=1000.0, value=333.3,
                                                step=0.1)

        # 重新采样数据
        resampled_data = resample_data(data, sampling_rate)

        # 用户选择滤波方案
        filter_options = st.sidebar.multiselect("请选择滤波方案:",
                                                ["低通滤波", "小波阈值去噪", "平滑滤波", "高通滤波"])

        # 存储每个滤波结果
        filtered_data_list = [resampled_data]
        # 存储滤波行为
        filtered_data_label=["原始数据"]
        # 应用多次滤波
        for option in filter_options:
            if option == "低通滤波":
                cutoff_freq = st.sidebar.number_input("请输入低通滤波截止频率 (Hz):", min_value=0.1, max_value=1000.0,
                                                      value=50.0, step=0.1)
                filter_order = st.sidebar.slider("请输入低通滤波器阶数:", min_value=1, max_value=10, value=4, step=1)
                filtered_data = apply_lowpass_filter(filtered_data_list[-1], cutoff_freq, sampling_rate, filter_order)
            elif option == "平滑滤波":
                window_size = st.sidebar.slider("请输入窗口大小:", min_value=3, max_value=101, value=11, step=2)
                filtered_data = smooth_filter(filtered_data_list[-1], window_size)
            elif option == "小波阈值去噪":
                level = st.sidebar.slider("请输入小波变换级数:", min_value=1, max_value=10, value=1, step=1)
                filtered_data = wavelet_denoising(filtered_data_list[-1], level=level)
            elif option == "高通滤波":
                cutoff_freq = st.sidebar.number_input("请输入高通滤波截止频率 (Hz):", min_value=0.1, max_value=1000.0,
                                                      value=50.0, step=0.1)
                filtered_data = highpass_filter(filtered_data_list[-1], cutoff_freq, sampling_rate)

            filtered_data_list.append(filtered_data)
            filtered_data_label.append(option)

        # 显示数据
        st.write("数据处理:")

        # 绘制数据曲线
        plot_data(filtered_data_list, filtered_data_label)
        # 显示时频图
        st.write("时频图:")
        plot_spectrogram(filtered_data_list[-1], sampling_rate)


if __name__ == "__main__":
    main()
