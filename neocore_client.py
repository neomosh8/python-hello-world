"""
Real-time Neocore EEG plotter with pwelch frequency analysis.

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

# ─────────────────────────── Signal Processing ──────────────────────────
# Sampling rate - adjust based on your device
SAMPLE_RATE = 250  # Hz - adjust this to match your device's actual rate
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
        high_cutoff = min(80.0, nyquist - 1) / nyquist  # Ensure below Nyquist

        if high_cutoff > low_cutoff:
            # Bandpass filter
            b_bp, a_bp = signal.butter(N=4, Wn=[low_cutoff, high_cutoff], btype='band', analog=False)
            filtered_data = signal.filtfilt(b_bp, a_bp, data_array)
        else:
            # If we can't do bandpass (sampling rate too low), just do high-pass
            b_hp, a_hp = signal.butter(N=4, Wn=low_cutoff, btype='high', analog=False)
            filtered_data = signal.filtfilt(b_hp, a_hp, data_array)

        # 60 Hz notch filter (strong notch with Q=30)
        if sps > 120:  # Only apply if sampling rate allows
            notch_freq = 60.0
            Q = 30.0  # Quality factor for strong notch
            b_notch, a_notch = signal.iirnotch(notch_freq, Q, sps)
            filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data)

        return filtered_data.tolist()

    except Exception as e:
        print(f"Error in filtering: {e}")
        return data  # Return original data if filtering fails

# Initialize lists to store historical values
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
        # Calculate power spectral density
        f_, pxx = signal.welch(data, fs=sps,
                              window=signal.windows.tukey(len(data), sym=False, alpha=.17),
                              nperseg=min(len(data), WINDOW_SIZE),
                              noverlap=int(min(len(data), WINDOW_SIZE) * 0.75),
                              nfft=min(len(data), WINDOW_SIZE * 2),  # Increased FFT size for better resolution
                              return_onesided=True,
                              scaling='spectrum', axis=-1, average='mean')

        power = pxx * 0.84

        # Calculate frequency band powers
        freq_res = f_[1] - f_[0]

        # Define frequency bands (more precise with better resolution)
        theta_start = int(4 / freq_res)
        theta_end = int(8 / freq_res)
        alpha_start = int(8 / freq_res)
        alpha_end = int(13 / freq_res)
        beta_start = int(13 / freq_res)
        beta_end = int(30 / freq_res)
        gamma_start = int(30 / freq_res)
        gamma_end = int(100 / freq_res)

        # Ensure indices are within bounds
        theta_end = min(theta_end, len(power))
        alpha_end = min(alpha_end, len(power))
        beta_end = min(beta_end, len(power))
        gamma_end = min(gamma_end, len(power))

        theta = np.sum(power[theta_start:theta_end]) if theta_end > theta_start else 0
        alpha = np.sum(power[alpha_start:alpha_end]) if alpha_end > alpha_start else 0
        beta = np.sum(power[beta_start:beta_end]) if beta_end > beta_start else 0
        gamma = np.sum(power[gamma_start:gamma_end]) if gamma_end > gamma_start else 0

        # Update history
        history_theta.append(theta)
        history_alpha.append(alpha)
        history_beta.append(beta)

        # Maintain history size
        if len(history_theta) > moving_average_samples:
            history_theta.pop(0)
        if len(history_alpha) > moving_average_samples:
            history_alpha.pop(0)
        if len(history_beta) > moving_average_samples:
            history_beta.pop(0)

        # Calculate attention index (theta/beta ratio)
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
    FREQ_WIDTH = 100  # Number of frequency analysis windows to keep

    def __init__(self, enable_filtering=False):
        self.enable_filtering = enable_filtering

        # Time domain data
        self.ch1 = deque(maxlen=self.TIME_WIDTH)
        self.ch2 = deque(maxlen=self.TIME_WIDTH)
        self.filtered_ch1 = deque(maxlen=self.TIME_WIDTH)
        self.filtered_ch2 = deque(maxlen=self.TIME_WIDTH)

        # Frequency domain data
        self.freq_data = None
        self.power_ch1_history = deque(maxlen=self.FREQ_WIDTH)
        self.power_ch2_history = deque(maxlen=self.FREQ_WIDTH)

        # Band powers
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

        # Create the plot with subplots
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))

        # Determine layout based on filtering option
        if self.enable_filtering:
            # Time domain plots (raw and filtered)
            self.ax_time1 = plt.subplot(3, 2, 1)
            self.ax_time2 = plt.subplot(3, 2, 2)
            self.ax_filt1 = plt.subplot(3, 2, 3)
            self.ax_filt2 = plt.subplot(3, 2, 4)

            # Frequency domain plots
            self.ax_freq1 = plt.subplot(3, 2, 5)
            self.ax_freq2 = plt.subplot(3, 2, 6)

            # Initialize plot lines
            self.l_time1, = self.ax_time1.plot([], [], 'g-', linewidth=1, label='Ch1 Raw')
            self.l_time2, = self.ax_time2.plot([], [], 'r-', linewidth=1, label='Ch2 Raw')
            self.l_filt1, = self.ax_filt1.plot([], [], 'b-', linewidth=1, label='Ch1 Filtered')
            self.l_filt2, = self.ax_filt2.plot([], [], 'm-', linewidth=1, label='Ch2 Filtered')

            # Configure axes
            self.ax_time1.set_title('Channel 1 - Raw EEG')
            self.ax_time2.set_title('Channel 2 - Raw EEG')
            self.ax_filt1.set_title('Channel 1 - Filtered EEG (0.5-80Hz + 60Hz notch)')
            self.ax_filt2.set_title('Channel 2 - Filtered EEG (0.5-80Hz + 60Hz notch)')

            for ax in [self.ax_time1, self.ax_time2, self.ax_filt1, self.ax_filt2]:
                ax.set_xlim(0, self.TIME_WIDTH)
                ax.set_ylim(-1000000, 1000000)
                ax.grid(True, alpha=0.3)
                ax.legend()

        else:
            # Only time domain and frequency domain (no filtered plots)
            self.ax_time1 = plt.subplot(2, 2, 1)
            self.ax_time2 = plt.subplot(2, 2, 2)
            self.ax_freq1 = plt.subplot(2, 2, 3)
            self.ax_freq2 = plt.subplot(2, 2, 4)

            # Initialize plot lines
            self.l_time1, = self.ax_time1.plot([], [], 'g-', linewidth=1, label='Ch1 Raw')
            self.l_time2, = self.ax_time2.plot([], [], 'r-', linewidth=1, label='Ch2 Raw')

            # Configure axes
            self.ax_time1.set_title('Channel 1 - Raw EEG')
            self.ax_time2.set_title('Channel 2 - Raw EEG')

            for ax in [self.ax_time1, self.ax_time2]:
                ax.set_xlim(0, self.TIME_WIDTH)
                ax.set_ylim(-1000000, 1000000)
                ax.grid(True, alpha=0.3)
                ax.legend()

        # Frequency domain plots (common configuration)
        self.l_freq1, = self.ax_freq1.plot([], [], 'g-', linewidth=1, label='Ch1 PSD')
        self.l_freq2, = self.ax_freq2.plot([], [], 'r-', linewidth=1, label='Ch2 PSD')

        self.ax_freq1.set_title('Channel 1 - Power Spectral Density (0-100 Hz)')
        self.ax_freq2.set_title('Channel 2 - Power Spectral Density (0-100 Hz)')

        for ax in [self.ax_freq1, self.ax_freq2]:
            ax.set_xlim(0, 100)  # Extended to 100 Hz
            ax.set_ylim(0, 1000)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add frequency band markers
            ax.axvline(x=4, color='purple', linestyle='--', alpha=0.5, label='θ')
            ax.axvline(x=8, color='blue', linestyle='--', alpha=0.5, label='α')
            ax.axvline(x=13, color='green', linestyle='--', alpha=0.5, label='β')
            ax.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='γ')
            ax.axvline(x=60, color='orange', linestyle=':', alpha=0.7, label='60Hz')

        # Add text for band powers
        self.text1 = self.ax_freq1.text(0.02, 0.98, '', transform=self.ax_freq1.transAxes,
                                        verticalalignment='top', fontsize=8,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.text2 = self.ax_freq2.text(0.02, 0.98, '', transform=self.ax_freq2.transAxes,
                                        verticalalignment='top', fontsize=8,
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show(block=False)

    def push(self, a: List[int], b: List[int]):
        """Add new data to the plot buffers"""
        self.ch1.extend(a)
        self.ch2.extend(b)

        # Apply filtering if enabled and we have enough data
        if self.enable_filtering and len(self.ch1) >= WINDOW_SIZE:
            # Get recent data for filtering
            recent_ch1 = list(self.ch1)[-WINDOW_SIZE:]
            recent_ch2 = list(self.ch2)[-WINDOW_SIZE:]

            # Filter the data
            filtered_ch1 = filter_eeg(recent_ch1, SAMPLE_RATE, self.enable_filtering)
            filtered_ch2 = filter_eeg(recent_ch2, SAMPLE_RATE, self.enable_filtering)

            # Add filtered data (only the new samples)
            new_samples = len(a)
            if len(filtered_ch1) >= new_samples:
                self.filtered_ch1.extend(filtered_ch1[-new_samples:])
            if len(filtered_ch2) >= new_samples:
                self.filtered_ch2.extend(filtered_ch2[-new_samples:])

            # Perform frequency analysis on filtered data
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
            # Perform frequency analysis on raw data if filtering is disabled
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
        """Update all plots"""
        if not self.data_updated:
            return

        try:
            # Update time domain plots
            if len(self.ch1) > 0:
                x_data = list(range(len(self.ch1)))
                self.l_time1.set_data(x_data, list(self.ch1))
                self.l_time2.set_data(x_data, list(self.ch2))

                # Auto-scale time domain
                y1_data = list(self.ch1)
                y2_data = list(self.ch2)
                if y1_data:
                    y1_min, y1_max = min(y1_data), max(y1_data)
                    if y1_max > y1_min:
                        margin = (y1_max - y1_min) * 0.1
                        self.ax_time1.set_ylim(y1_min - margin, y1_max + margin)
                        self.ax_time2.set_ylim(y1_min - margin, y1_max + margin)

                # Update x-axis
                self.ax_time1.set_xlim(max(0, len(x_data) - self.TIME_WIDTH), len(x_data))
                self.ax_time2.set_xlim(max(0, len(x_data) - self.TIME_WIDTH), len(x_data))

            # Update filtered data plots (only if filtering is enabled)
            if self.enable_filtering and len(self.filtered_ch1) > 0:
                x_filt = list(range(len(self.filtered_ch1)))
                self.l_filt1.set_data(x_filt, list(self.filtered_ch1))
                self.l_filt2.set_data(x_filt, list(self.filtered_ch2))

                # Auto-scale filtered plots
                y1_filt = list(self.filtered_ch1)
                if y1_filt:
                    y1_min, y1_max = min(y1_filt), max(y1_filt)
                    if y1_max > y1_min:
                        margin = (y1_max - y1_min) * 0.1
                        self.ax_filt1.set_ylim(y1_min - margin, y1_max + margin)
                        self.ax_filt2.set_ylim(y1_min - margin, y1_max + margin)

                self.ax_filt1.set_xlim(max(0, len(x_filt) - self.TIME_WIDTH), len(x_filt))
                self.ax_filt2.set_xlim(max(0, len(x_filt) - self.TIME_WIDTH), len(x_filt))

            # Update frequency domain plots
            if self.freq_data is not None and len(self.power_ch1_history) > 0:
                # Use the most recent power spectrum
                latest_pxx1 = self.power_ch1_history[-1]
                latest_pxx2 = self.power_ch2_history[-1]

                # Limit to 0-100 Hz range
                freq_mask = self.freq_data <= 100
                f_plot = self.freq_data[freq_mask]
                pxx1_plot = latest_pxx1[freq_mask]
                pxx2_plot = latest_pxx2[freq_mask]

                self.l_freq1.set_data(f_plot, pxx1_plot)
                self.l_freq2.set_data(f_plot, pxx2_plot)

                # Auto-scale frequency plots
                if len(pxx1_plot) > 0:
                    max_power = max(np.max(pxx1_plot), np.max(pxx2_plot))
                    self.ax_freq1.set_ylim(0, max_power * 1.1)
                    self.ax_freq2.set_ylim(0, max_power * 1.1)

                # Update band power text
                if len(self.theta_ch1) > 0:
                    text1 = f"θ: {self.theta_ch1[-1]:.2f}\nα: {self.alpha_ch1[-1]:.2f}\nβ: {self.beta_ch1[-1]:.2f}\nγ: {self.gamma_ch1[-1]:.2f}\nAtt: {self.attention_ch1[-1]:.3f}"
                    text2 = f"θ: {self.theta_ch2[-1]:.2f}\nα: {self.alpha_ch2[-1]:.2f}\nβ: {self.beta_ch2[-1]:.2f}\nγ: {self.gamma_ch2[-1]:.2f}\nAtt: {self.attention_ch2[-1]:.3f}"
                    self.text1.set_text(text1)
                    self.text2.set_text(text2)

            # Redraw
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
        if len(data) % 4 != 0:
            return

        num_samples = len(data) // 4
        try:
            ecg_data = struct.unpack(f"<{num_samples}i", data)

            ch1_data = []
            ch2_data = []
            for i in range(0, num_samples, 2):
                ch1_data.append(ecg_data[i])
                if i + 1 < num_samples:
                    ch2_data.append(ecg_data[i + 1])

            self.plot.push(ch1_data, ch2_data)

        except struct.error as e:
            print(f"Error unpacking EEG data: {e}")

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

            print("All plots active. Close window to stop streaming.")
            print("Configuration:")
            print(f"- Sample rate: {SAMPLE_RATE} Hz")
            print(f"- Filtering: {'ENABLED (0.5-80Hz + 60Hz notch)' if self.enable_filtering else 'DISABLED'}")
            print("- PSD range: 0-100 Hz")
            print("- Brain waves: θ(4-8Hz), α(8-13Hz), β(13-30Hz), γ(30-100Hz)")
            print("- Attention index: θ/β ratio")

            plot_update_counter = 0
            while plt.fignum_exists(self.plot.fig.number):
                plot_update_counter += 1
                if plot_update_counter >= 3:  # Update every ~30ms
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

    print("Neocore EEG Plotter with Real-time Frequency Analysis")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Window size: {WINDOW_SIZE} samples ({WINDOW_DURATION}s)")
    print(f"Test mode: {'ON' if test else 'OFF'}")
    print(f"Filtering: {'ENABLED (0.5-80Hz bandpass + 60Hz notch)' if enable_filtering else 'DISABLED'}")
    print(f"PSD range: 0-100 Hz")
    if mac:
        print(f"Target MAC: {mac}")
    else:
        print("Will scan for devices")

    print("\nUsage:")
    print("  python neo_plot.py --test --filter  # Test mode with filtering")
    print("  python neo_plot.py --filter         # Normal mode with filtering")
    print("  python neo_plot.py                  # Normal mode without filtering")

    try:
        asyncio.run(Headset(mac, test, enable_filtering).run())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()