"""
Real-time Neocore EEG plotter with proper ADS1299 data conversion.

pip install bleak matplotlib scipy numpy
python neo_plot.py                 # scan for any QCC5181 / NEOCORE
python neo_plot.py AA:BB:CC:DD:EE  # connect by MAC
python neo_plot.py --test          # enable built-in square-wave
python neo_plot.py --filter        # enable signal filtering (0.5-80Hz + 60Hz notch)
"""

import asyncio
import struct
import sys
from collections import deque
from typing import List, Optional
import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient

# ─────────────────────────── BLE constants ──────────────────────────
RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write
TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify
TARGET_NAMES = ["QCC5181", "QCC5181-LE", "NEOCORE"]

# Feature IDs
NEOCORE_SENSOR_CFG_FEATURE_ID = 0x01
NEOCORE_SENSOR_STREAM_FEATURE_ID = 0x02

# PDU Types
PDU_TYPE_COMMAND = 0
PDU_TYPE_NOTIFICATION = 1

# Command IDs
NEOCORE_CMD_ID_DATA_STREAM_CTRL = 0
NEOCORE_CMD_ID_EEG_TEST_SIGNAL_CTRL = 1

# Notification IDs
NEOCORE_NOTIFY_ID_EEG_DATA = 0

# ─────────────────────────── ADS1299 Configuration ──────────────────────────
# ADS1299 specifications for voltage conversion
ADS1299_VREF = 4.5  # Reference voltage in volts (typical)
ADS1299_GAIN = 24   # Programmable gain (1, 2, 4, 6, 8, 12, 24)
ADS1299_RESOLUTION = 24  # 24-bit ADC

# Calculate LSB value in microvolts
# Formula from datasheet: 1 LSB = (2 × VREF / Gain) / 2^24
LSB_UV = (2 * ADS1299_VREF * 1000000) / (ADS1299_GAIN * (2**ADS1299_RESOLUTION))
print(f"ADS1299 LSB = {LSB_UV:.3f} µV")

def convert_ads1299_to_uv(raw_value):
    """Convert ADS1299 raw value to microvolts"""
    # Handle 24-bit signed integer conversion
    if raw_value > 0x7FFFFF:  # If greater than max positive 24-bit
        raw_value = raw_value - (1 << 24)  # Convert to negative

    return raw_value * LSB_UV

def parse_ads1299_data(data, debug=False):
    """Parse ADS1299 data structure"""
    if debug:
        print(f"Raw data length: {len(data)} bytes")
        print(f"First 24 bytes: {data[:24].hex()}")

    # Try different parsing methods
    results = {}

    # Method 1: 32-bit integers (current method)
    if len(data) % 4 == 0:
        num_samples_32 = len(data) // 4
        try:
            samples_32 = struct.unpack(f"<{num_samples_32}i", data)
            ch1_32 = [convert_ads1299_to_uv(samples_32[i]) for i in range(0, len(samples_32), 2)]
            ch2_32 = [convert_ads1299_to_uv(samples_32[i]) for i in range(1, len(samples_32), 2)]
            results['32bit_alternating'] = (ch1_32, ch2_32)
            if debug:
                print(f"32-bit alternating: {len(ch1_32)} + {len(ch2_32)} samples")
                print(f"Ch1 range: {min(ch1_32):.1f} to {max(ch1_32):.1f} µV")
                print(f"Ch2 range: {min(ch2_32):.1f} to {max(ch2_32):.1f} µV")
        except:
            pass

    # Method 2: 24-bit samples (3 bytes each)
    if len(data) % 3 == 0:
        num_samples_24 = len(data) // 3
        try:
            samples_24 = []
            for i in range(0, len(data), 3):
                # Convert 3 bytes to 24-bit signed integer
                raw_bytes = data[i:i+3]
                # Little endian 24-bit
                value = int.from_bytes(raw_bytes, byteorder='little', signed=False)
                if value > 0x7FFFFF:
                    value = value - (1 << 24)
                samples_24.append(convert_ads1299_to_uv(value))

            # Try alternating pattern
            ch1_24 = [samples_24[i] for i in range(0, len(samples_24), 2)]
            ch2_24 = [samples_24[i] for i in range(1, len(samples_24), 2)]
            results['24bit_alternating'] = (ch1_24, ch2_24)
            if debug:
                print(f"24-bit alternating: {len(ch1_24)} + {len(ch2_24)} samples")
                print(f"Ch1 range: {min(ch1_24):.1f} to {max(ch1_24):.1f} µV")
                print(f"Ch2 range: {min(ch2_24):.1f} to {max(ch2_24):.1f} µV")
        except:
            pass

    # Method 3: Check if data is sequential (all ch1 then all ch2)
    if len(data) % 4 == 0:
        num_samples_32 = len(data) // 4
        try:
            samples_32 = struct.unpack(f"<{num_samples_32}i", data)
            mid_point = len(samples_32) // 2
            ch1_seq = [convert_ads1299_to_uv(samples_32[i]) for i in range(mid_point)]
            ch2_seq = [convert_ads1299_to_uv(samples_32[i]) for i in range(mid_point, len(samples_32))]
            results['32bit_sequential'] = (ch1_seq, ch2_seq)
            if debug:
                print(f"32-bit sequential: {len(ch1_seq)} + {len(ch2_seq)} samples")
                print(f"Ch1 range: {min(ch1_seq):.1f} to {max(ch1_seq):.1f} µV")
                print(f"Ch2 range: {min(ch2_seq):.1f} to {max(ch2_seq):.1f} µV")
        except:
            pass

    # Method 4: Try big endian
    if len(data) % 4 == 0:
        num_samples_32 = len(data) // 4
        try:
            samples_32_be = struct.unpack(f">{num_samples_32}i", data)
            ch1_be = [convert_ads1299_to_uv(samples_32_be[i]) for i in range(0, len(samples_32_be), 2)]
            ch2_be = [convert_ads1299_to_uv(samples_32_be[i]) for i in range(1, len(samples_32_be), 2)]
            results['32bit_bigendian'] = (ch1_be, ch2_be)
            if debug:
                print(f"32-bit big endian: {len(ch1_be)} + {len(ch2_be)} samples")
                print(f"Ch1 range: {min(ch1_be):.1f} to {max(ch1_be):.1f} µV")
                print(f"Ch2 range: {min(ch2_be):.1f} to {max(ch2_be):.1f} µV")
        except:
            pass

    return results

# ─────────────────────────── Signal Processing ──────────────────────────
SAMPLE_RATE = 250  # Hz
WINDOW_DURATION = 1  # seconds
WINDOW_SIZE = int(WINDOW_DURATION * SAMPLE_RATE)

def filter_eeg(data, sps, enable_filtering=True):
    """Apply bandpass filter and 60Hz notch to EEG data"""
    if not enable_filtering or len(data) < 10:
        return data

    try:
        data_array = np.array(data, dtype=float)

        # Bandpass filter: 0.5 Hz to 80 Hz
        nyquist = sps / 2
        low_cutoff = 0.5 / nyquist
        high_cutoff = min(80.0, nyquist - 1) / nyquist

        if high_cutoff > low_cutoff:
            b_bp, a_bp = signal.butter(N=4, Wn=[low_cutoff, high_cutoff], btype='band', analog=False)
            filtered_data = signal.filtfilt(b_bp, a_bp, data_array)
        else:
            b_hp, a_hp = signal.butter(N=4, Wn=low_cutoff, btype='high', analog=False)
            filtered_data = signal.filtfilt(b_hp, a_hp, data_array)

        # 60 Hz notch filter
        if sps > 120:
            notch_freq = 60.0
            Q = 30.0
            b_notch, a_notch = signal.iirnotch(notch_freq, Q, sps)
            filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)

        return filtered_data.tolist()

    except Exception as e:
        print(f"Error in filtering: {e}")
        return data

# Band power calculation functions (same as before)
history_theta = []
history_alpha = []
history_beta = []
moving_average_samples = 10

def pwelch_analysis(data, sps):
    """Perform pwelch analysis and return frequency spectrum and band powers"""
    global history_theta, history_alpha, history_beta, moving_average_samples

    if len(data) < WINDOW_SIZE:
        return None, None, None, None, None, None

    try:
        f_, pxx = signal.welch(data, fs=sps,
                              window=signal.windows.tukey(len(data), sym=False, alpha=.17),
                              nperseg=min(len(data), WINDOW_SIZE),
                              noverlap=int(min(len(data), WINDOW_SIZE) * 0.75),
                              nfft=min(len(data), WINDOW_SIZE * 2),
                              return_onesided=True,
                              scaling='spectrum', axis=-1, average='mean')

        power = pxx * 0.84

        freq_res = f_[1] - f_[0]

        theta_start = int(4 / freq_res)
        theta_end = int(8 / freq_res)
        alpha_start = int(8 / freq_res)
        alpha_end = int(13 / freq_res)
        beta_start = int(13 / freq_res)
        beta_end = int(30 / freq_res)
        gamma_start = int(30 / freq_res)
        gamma_end = int(100 / freq_res)

        theta_end = min(theta_end, len(power))
        alpha_end = min(alpha_end, len(power))
        beta_end = min(beta_end, len(power))
        gamma_end = min(gamma_end, len(power))

        theta = np.sum(power[theta_start:theta_end]) if theta_end > theta_start else 0
        alpha = np.sum(power[alpha_start:alpha_end]) if alpha_end > alpha_start else 0
        beta = np.sum(power[beta_start:beta_end]) if beta_end > beta_start else 0
        gamma = np.sum(power[gamma_start:gamma_end]) if gamma_end > gamma_start else 0

        history_theta.append(theta)
        history_alpha.append(alpha)
        history_beta.append(beta)

        if len(history_theta) > moving_average_samples:
            history_theta.pop(0)
        if len(history_alpha) > moving_average_samples:
            history_alpha.pop(0)
        if len(history_beta) > moving_average_samples:
            history_beta.pop(0)

        attention_index = 0
        if len(history_theta) >= 5 and len(history_beta) >= 5:
            avg_theta = np.mean(history_theta)
            avg_beta = np.mean(history_beta)
            if avg_beta > 0:
                attention_index = avg_theta / avg_beta

        return f_, power, theta, alpha, beta, gamma, attention_index

    except Exception as e:
        print(f"Error in pwelch analysis: {e}")
        return None, None, None, None, None, None, None

# ─────────────────────────── plotting helper ─────────────────────────
class LivePlot:
    TIME_WIDTH = 1000
    FREQ_WIDTH = 100

    def __init__(self, enable_filtering=False):
        self.enable_filtering = enable_filtering
        self.parsing_method = '32bit_alternating'  # Default, will be auto-detected
        self.packet_count = 0

        # Time domain data
        self.ch1 = deque(maxlen=self.TIME_WIDTH)
        self.ch2 = deque(maxlen=self.TIME_WIDTH)
        self.filtered_ch1 = deque(maxlen=self.TIME_WIDTH)
        self.filtered_ch2 = deque(maxlen=self.TIME_WIDTH)

        # Frequency and band power data (same as before)
        self.freq_data = None
        self.power_ch1_history = deque(maxlen=self.FREQ_WIDTH)
        self.power_ch2_history = deque(maxlen=self.FREQ_WIDTH)

        self.theta_ch1 = deque(maxlen=self.FREQ_WIDTH)
        self.alpha_ch1 = deque(maxlen=self.FREQ_WIDTH)
        self.beta_ch1 = deque(maxlen=self.FREQ_WIDTH)
        self.gamma_ch1 = deque(maxlen=self.FREQ_WIDTH)
        self.attention_ch1 = deque(maxlen=self.FREQ_WIDTH)

        self.theta_ch2 = deque(maxlen=self.FREQ_WIDTH)
        self.alpha_ch2 = deque(maxlen=self.FREQ_WIDTH)
        self.beta_ch2 = deque(maxlen=self.FREQ_WIDTH)
        self.gamma_ch2 = deque(maxlen=self.FREQ_WIDTH)
        self.attention_ch2 = deque(maxlen=self.FREQ_WIDTH)

        self.data_updated = False

        # Create plots (same layout as before)
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))

        if self.enable_filtering:
            self.ax_time1 = plt.subplot(3, 2, 1)
            self.ax_time2 = plt.subplot(3, 2, 2)
            self.ax_filt1 = plt.subplot(3, 2, 3)
            self.ax_filt2 = plt.subplot(3, 2, 4)
            self.ax_freq1 = plt.subplot(3, 2, 5)
            self.ax_freq2 = plt.subplot(3, 2, 6)

            self.l_time1, = self.ax_time1.plot([], [], 'g-', linewidth=1, label='Ch1 Raw')
            self.l_time2, = self.ax_time2.plot([], [], 'r-', linewidth=1, label='Ch2 Raw')
            self.l_filt1, = self.ax_filt1.plot([], [], 'b-', linewidth=1, label='Ch1 Filtered')
            self.l_filt2, = self.ax_filt2.plot([], [], 'm-', linewidth=1, label='Ch2 Filtered')

            self.ax_time1.set_title('Channel 1 - Raw EEG (µV)')
            self.ax_time2.set_title('Channel 2 - Raw EEG (µV)')
            self.ax_filt1.set_title('Channel 1 - Filtered EEG (µV)')
            self.ax_filt2.set_title('Channel 2 - Filtered EEG (µV)')

            for ax in [self.ax_time1, self.ax_time2, self.ax_filt1, self.ax_filt2]:
                ax.set_xlim(0, self.TIME_WIDTH)
                ax.set_ylim(-200, 200)  # Typical EEG range in µV
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_ylabel('Amplitude (µV)')
        else:
            self.ax_time1 = plt.subplot(2, 2, 1)
            self.ax_time2 = plt.subplot(2, 2, 2)
            self.ax_freq1 = plt.subplot(2, 2, 3)
            self.ax_freq2 = plt.subplot(2, 2, 4)

            self.l_time1, = self.ax_time1.plot([], [], 'g-', linewidth=1, label='Ch1 Raw')
            self.l_time2, = self.ax_time2.plot([], [], 'r-', linewidth=1, label='Ch2 Raw')

            self.ax_time1.set_title('Channel 1 - Raw EEG (µV)')
            self.ax_time2.set_title('Channel 2 - Raw EEG (µV)')

            for ax in [self.ax_time1, self.ax_time2]:
                ax.set_xlim(0, self.TIME_WIDTH)
                ax.set_ylim(-200, 200)  # Typical EEG range in µV
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_ylabel('Amplitude (µV)')

        # Frequency domain setup (same as before)
        self.l_freq1, = self.ax_freq1.plot([], [], 'g-', linewidth=1, label='Ch1 PSD')
        self.l_freq2, = self.ax_freq2.plot([], [], 'r-', linewidth=1, label='Ch2 PSD')

        self.ax_freq1.set_title('Channel 1 - Power Spectral Density')
        self.ax_freq2.set_title('Channel 2 - Power Spectral Density')

        for ax in [self.ax_freq1, self.ax_freq2]:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1000)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power (µV²/Hz)')
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax.axvline(x=4, color='purple', linestyle='--', alpha=0.5)
            ax.axvline(x=8, color='blue', linestyle='--', alpha=0.5)
            ax.axvline(x=13, color='green', linestyle='--', alpha=0.5)
            ax.axvline(x=30, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=60, color='orange', linestyle=':', alpha=0.7)

        self.text1 = self.ax_freq1.text(0.02, 0.98, '', transform=self.ax_freq1.transAxes,
                                        verticalalignment='top', fontsize=8,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.text2 = self.ax_freq2.text(0.02, 0.98, '', transform=self.ax_freq2.transAxes,
                                        verticalalignment='top', fontsize=8,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show(block=False)

    def push(self, a: List[float], b: List[float]):
        """Add new data to the plot buffers (now in µV)"""
        self.ch1.extend(a)
        self.ch2.extend(b)

        # Apply filtering if enabled
        if self.enable_filtering and len(self.ch1) >= WINDOW_SIZE:
            recent_ch1 = list(self.ch1)[-WINDOW_SIZE:]
            recent_ch2 = list(self.ch2)[-WINDOW_SIZE:]

            filtered_ch1 = filter_eeg(recent_ch1, SAMPLE_RATE, self.enable_filtering)
            filtered_ch2 = filter_eeg(recent_ch2, SAMPLE_RATE, self.enable_filtering)

            new_samples = len(a)
            if len(filtered_ch1) >= new_samples:
                self.filtered_ch1.extend(filtered_ch1[-new_samples:])
            if len(filtered_ch2) >= new_samples:
                self.filtered_ch2.extend(filtered_ch2[-new_samples:])

            if len(filtered_ch1) >= WINDOW_SIZE:
                f, pxx1, theta1, alpha1, beta1, gamma1, att1 = pwelch_analysis(filtered_ch1, SAMPLE_RATE)
                if f is not None:
                    self.freq_data = f
                    self.power_ch1_history.append(pxx1)
                    self.theta_ch1.append(theta1)
                    self.alpha_ch1.append(alpha1)
                    self.beta_ch1.append(beta1)
                    self.gamma_ch1.append(gamma1)
                    self.attention_ch1.append(att1)

            if len(filtered_ch2) >= WINDOW_SIZE:
                f, pxx2, theta2, alpha2, beta2, gamma2, att2 = pwelch_analysis(filtered_ch2, SAMPLE_RATE)
                if f is not None:
                    self.power_ch2_history.append(pxx2)
                    self.theta_ch2.append(theta2)
                    self.alpha_ch2.append(alpha2)
                    self.beta_ch2.append(beta2)
                    self.gamma_ch2.append(gamma2)
                    self.attention_ch2.append(att2)

        elif len(self.ch1) >= WINDOW_SIZE:
            recent_ch1 = list(self.ch1)[-WINDOW_SIZE:]
            recent_ch2 = list(self.ch2)[-WINDOW_SIZE:]

            f, pxx1, theta1, alpha1, beta1, gamma1, att1 = pwelch_analysis(recent_ch1, SAMPLE_RATE)
            if f is not None:
                self.freq_data = f
                self.power_ch1_history.append(pxx1)
                self.theta_ch1.append(theta1)
                self.alpha_ch1.append(alpha1)
                self.beta_ch1.append(beta1)
                self.gamma_ch1.append(gamma1)
                self.attention_ch1.append(att1)

            f, pxx2, theta2, alpha2, beta2, gamma2, att2 = pwelch_analysis(recent_ch2, SAMPLE_RATE)
            if f is not None:
                self.power_ch2_history.append(pxx2)
                self.theta_ch2.append(theta2)
                self.alpha_ch2.append(alpha2)
                self.beta_ch2.append(beta2)
                self.gamma_ch2.append(gamma2)
                self.attention_ch2.append(att2)

        self.data_updated = True

    def update_plot(self):
        """Update all plots with proper µV scaling"""
        if not self.data_updated:
            return

        try:
            # Update time domain plots
            if len(self.ch1) > 0:
                x_data = list(range(len(self.ch1)))
                self.l_time1.set_data(x_data, list(self.ch1))
                self.l_time2.set_data(x_data, list(self.ch2))

                # Auto-scale for typical EEG range
                y1_data = list(self.ch1)
                if y1_data:
                    y1_min, y1_max = min(y1_data), max(y1_data)
                    if y1_max > y1_min:
                        margin = max(50, (y1_max - y1_min) * 0.1)  # At least 50µV margin
                        self.ax_time1.set_ylim(y1_min - margin, y1_max + margin)
                        self.ax_time2.set_ylim(y1_min - margin, y1_max + margin)

                self.ax_time1.set_xlim(max(0, len(x_data) - self.TIME_WIDTH), len(x_data))
                self.ax_time2.set_xlim(max(0, len(x_data) - self.TIME_WIDTH), len(x_data))

            # Update filtered plots if enabled
            if self.enable_filtering and len(self.filtered_ch1) > 0:
                x_filt = list(range(len(self.filtered_ch1)))
                self.l_filt1.set_data(x_filt, list(self.filtered_ch1))
                self.l_filt2.set_data(x_filt, list(self.filtered_ch2))

                y1_filt = list(self.filtered_ch1)
                if y1_filt:
                    y1_min, y1_max = min(y1_filt), max(y1_filt)
                    if y1_max > y1_min:
                        margin = max(50, (y1_max - y1_min) * 0.1)
                        self.ax_filt1.set_ylim(y1_min - margin, y1_max + margin)
                        self.ax_filt2.set_ylim(y1_min - margin, y1_max + margin)

                self.ax_filt1.set_xlim(max(0, len(x_filt) - self.TIME_WIDTH), len(x_filt))
                self.ax_filt2.set_xlim(max(0, len(x_filt) - self.TIME_WIDTH), len(x_filt))

            # Update frequency plots
            if self.freq_data is not None and len(self.power_ch1_history) > 0:
                latest_pxx1 = self.power_ch1_history[-1]
                latest_pxx2 = self.power_ch2_history[-1]

                freq_mask = self.freq_data <= 100
                f_plot = self.freq_data[freq_mask]
                pxx1_plot = latest_pxx1[freq_mask]
                pxx2_plot = latest_pxx2[freq_mask]

                self.l_freq1.set_data(f_plot, pxx1_plot)
                self.l_freq2.set_data(f_plot, pxx2_plot)

                if len(pxx1_plot) > 0:
                    max_power = max(np.max(pxx1_plot), np.max(pxx2_plot))
                    self.ax_freq1.set_ylim(0, max_power * 1.1)
                    self.ax_freq2.set_ylim(0, max_power * 1.1)

                if len(self.theta_ch1) > 0:
                    text1 = f"Method: {self.parsing_method}\nθ: {self.theta_ch1[-1]:.2f}\nα: {self.alpha_ch1[-1]:.2f}\nβ: {self.beta_ch1[-1]:.2f}\nγ: {self.gamma_ch1[-1]:.2f}\nAtt: {self.attention_ch1[-1]:.3f}"
                    text2 = f"Packets: {self.packet_count}\nθ: {self.theta_ch2[-1]:.2f}\nα: {self.alpha_ch2[-1]:.2f}\nβ: {self.beta_ch2[-1]:.2f}\nγ: {self.gamma_ch2[-1]:.2f}\nAtt: {self.attention_ch2[-1]:.3f}"
                    self.text1.set_text(text1)
                    self.text2.set_text(text2)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.data_updated = False

        except Exception as e:
            print(f"Error updating plot: {e}")

# ─────────────────────────── headset driver ──────────────────────────
class Headset:
    def __init__(self, mac: Optional[str], use_test: bool, enable_filtering: bool):
        self.mac = mac
        self.use_test = use_test
        self.enable_filtering = enable_filtering
        self.plot = LivePlot(enable_filtering)
        self.packet_count = 0
        self.parsing_method_locked = False

    async def _scan(self) -> str:
        if self.mac:
            return self.mac
        print("Scanning…")
        for dev in await BleakScanner.discover():
            if dev.name and any(tag in dev.name.upper() for tag in TARGET_NAMES):
                print(f"Found {dev.name} [{dev.address}]")
                return dev.address
        raise RuntimeError("device not found")

    @staticmethod
    def _build_command(feature_id: int, pdu_type: int, pdu_specific_id: int, data: bytes = b"") -> bytes:
        command_id = ((feature_id << 9) | (pdu_type << 7)) | pdu_specific_id
        return command_id.to_bytes(2, "big") + data

    async def _send(self, cli, feature_id, pdu_specific_id, data=b""):
        packet = self._build_command(feature_id, PDU_TYPE_COMMAND, pdu_specific_id, data)
        print(f"Sending: Feature={feature_id}, CID={pdu_specific_id}, Data={data.hex()}")
        await cli.write_gatt_char(RX_UUID, packet, response=False)

    def _parse_data_to_small_buffer(self, data: bytearray):
        if len(data) < 4:
            return None, None, None, None
        type_val = data[0]
        length = data[1]
        msg_index = int.from_bytes(data[2:3], "little", signed=False)
        value = data[4:length+4]
        return type_val, length, msg_index, value

    def _data_handler_ecg(self, data: bytearray, length: int):
        """Handle EEG data with proper ADS1299 parsing and voltage conversion"""
        self.packet_count += 1

        # Try multiple parsing methods on first few packets
        debug_mode = self.packet_count <= 3 and not self.parsing_method_locked

        parse_results = parse_ads1299_data(data, debug=debug_mode)

        if not parse_results:
            print("No valid parsing methods found")
            return

        # Auto-select best parsing method based on reasonable EEG values
        best_method = None
        best_score = float('inf')

        for method_name, (ch1_data, ch2_data) in parse_results.items():
            if len(ch1_data) > 0 and len(ch2_data) > 0:
                # Score based on reasonable EEG amplitude (typically < 500µV)
                ch1_max = max(abs(x) for x in ch1_data)
                ch2_max = max(abs(x) for x in ch2_data)
                max_amp = max(ch1_max, ch2_max)

                # Prefer amplitudes in the 10-500µV range
                if 10 <= max_amp <= 500:
                    score = 0  # Perfect score
                elif max_amp < 10:
                    score = 10 - max_amp  # Too small
                else:
                    score = max_amp - 500  # Too large

                if debug_mode:
                    print(f"{method_name}: max_amp={max_amp:.1f}µV, score={score:.1f}")

                if score < best_score:
                    best_score = score
                    best_method = method_name

        if best_method and not self.parsing_method_locked:
            self.plot.parsing_method = best_method
            if self.packet_count >= 3:
                self.parsing_method_locked = True
                print(f"Auto-selected parsing method: {best_method}")

        # Use the selected parsing method
        if best_method and best_method in parse_results:
            ch1_data, ch2_data = parse_results[best_method]
            self.plot.packet_count = self.packet_count
            self.plot.push(ch1_data, ch2_data)
        elif self.plot.parsing_method in parse_results:
            ch1_data, ch2_data = parse_results[self.plot.parsing_method]
            self.plot.packet_count = self.packet_count
            self.plot.push(ch1_data, ch2_data)

    def _on_notify(self, _h, data: bytearray):
        if len(data) < 2:
            return

        command_id = int.from_bytes(data[:2], "big")
        feature_id = command_id >> 9
        pdu_type = (command_id >> 7) & 0x0003
        pdu_specific_id = command_id & 0x007F

        if (feature_id, pdu_type, pdu_specific_id) != (NEOCORE_SENSOR_STREAM_FEATURE_ID, PDU_TYPE_NOTIFICATION, NEOCORE_NOTIFY_ID_EEG_DATA):
            return

        payload = data[2:]
        type_val, length, msg_index, value = self._parse_data_to_small_buffer(payload)

        if type_val == 2:  # EEG data
            self._data_handler_ecg(value, length)

    async def run(self):
        addr = await self._scan()
        print(f"Connecting to {addr}...")

        async with BleakClient(addr) as cli:
            print("Connected!")

            if hasattr(cli, "request_mtu"):
                try:
                    mtu = await cli.request_mtu(247)
                    print(f"MTU set to {mtu}")
                except Exception as e:
                    print(f"Failed to set MTU: {e}")

            await cli.start_notify(TX_UUID, self._on_notify)
            print("Notifications enabled")

            if self.use_test:
                print("Enabling test signal...")
                await self._send(cli, NEOCORE_SENSOR_CFG_FEATURE_ID, NEOCORE_CMD_ID_EEG_TEST_SIGNAL_CTRL, b"\x01")
                await asyncio.sleep(0.1)

            print("Starting EEG stream...")
            await self._send(cli, NEOCORE_SENSOR_CFG_FEATURE_ID, NEOCORE_CMD_ID_DATA_STREAM_CTRL, b"\x01")
            await asyncio.sleep(0.1)

            print("EEG Plotter with ADS1299 calibration active!")
            print("Configuration:")
            print(f"- ADS1299 VREF: {ADS1299_VREF}V, Gain: {ADS1299_GAIN}x")
            print(f"- LSB value: {LSB_UV:.3f} µV")
            print(f"- Sample rate: {SAMPLE_RATE} Hz")
            print(f"- Filtering: {'ENABLED' if self.enable_filtering else 'DISABLED'}")
            print("- Auto-detecting optimal data parsing method...")

            plot_update_counter = 0
            while plt.fignum_exists(self.plot.fig.number):
                plot_update_counter += 1
                if plot_update_counter >= 3:
                    self.plot.update_plot()
                    plot_update_counter = 0

                plt.pause(0.01)
                await asyncio.sleep(0.01)

            print("Stopping stream...")
            await self._send(cli, NEOCORE_SENSOR_CFG_FEATURE_ID, NEOCORE_CMD_ID_DATA_STREAM_CTRL, b"\x00")

            if self.use_test:
                await self._send(cli, NEOCORE_SENSOR_CFG_FEATURE_ID, NEOCORE_CMD_ID_EEG_TEST_SIGNAL_CTRL, b"\x00")

            await cli.stop_notify(TX_UUID)
            print("Disconnected")

def main():
    test = "--test" in sys.argv
    enable_filtering = "--filter" in sys.argv
    mac = next((a for a in sys.argv[1:] if not a.startswith("--")), None)

    print("Neocore EEG Plotter with ADS1299 Voltage Conversion")
    print(f"ADS1299 Configuration: VREF={ADS1299_VREF}V, Gain={ADS1299_GAIN}x, LSB={LSB_UV:.3f}µV")
    print(f"Test mode: {'ON' if test else 'OFF'}")
    print(f"Filtering: {'ENABLED' if enable_filtering else 'DISABLED'}")

    try:
        asyncio.run(Headset(mac, test, enable_filtering).run())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()