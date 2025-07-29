#!/usr/bin/env python3
"""
Neocore EEG Live Plotter - Enhanced with Dynamic Range and SNR Analysis
Stable real-time visualization with signal quality metrics.
"""

import asyncio
import sys
import struct
import time
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
from scipy import signal
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from bleak import BleakScanner, BleakClient

# ═══════════════════════════════════════════════════════════════════════════════
# BLE Configuration
# ═══════════════════════════════════════════════════════════════════════════════

RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
TARGET_NAMES = {"QCC5181", "QCC5181-LE", "NEOCORE"}

FEATURE_SENSOR_CFG = 0x01
CMD_STREAM_CTRL = 0x00
PDU_TYPE_COMMAND = 0x00

# ═══════════════════════════════════════════════════════════════════════════════
# Signal Parameters
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 250
SAMPLES_PER_CHUNK = 27
NUM_CHANNELS = 2
DISPLAY_WINDOW_SEC = 4
DISPLAY_SAMPLES = SAMPLE_RATE * DISPLAY_WINDOW_SEC

# Signal quality analysis parameters
SIGNAL_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 45)
}
NOISE_BAND = (45, 100)  # High frequency noise


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Signal Processing with Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class SignalQualityAnalyzer:
    """Analyzes signal quality including dynamic range and SNR."""

    def __init__(self, sample_rate: int):
        self.fs = sample_rate

    def calculate_dynamic_range(self, signal_data: np.ndarray) -> dict:
        """Calculate dynamic range in different forms."""
        if len(signal_data) == 0:
            return {'linear': 0, 'db': 0, 'peak_to_peak': 0, 'rms': 0}

        # Remove DC component
        signal_ac = signal_data - np.mean(signal_data)

        # Peak-to-peak dynamic range
        peak_to_peak = np.max(signal_ac) - np.min(signal_ac)

        # RMS value
        rms = np.sqrt(np.mean(signal_ac ** 2))

        # Dynamic range in dB (20*log10(max/min))
        max_val = np.max(np.abs(signal_ac))
        min_val = np.min(np.abs(signal_ac[signal_ac != 0])) if np.any(signal_ac != 0) else 1e-10
        dr_db = 20 * np.log10(max_val / min_val) if min_val > 0 else 0

        return {
            'linear': max_val / min_val if min_val > 0 else 0,
            'db': dr_db,
            'peak_to_peak': peak_to_peak,
            'rms': rms,
            'max': max_val,
            'min': min_val
        }

    def calculate_snr(self, signal_data: np.ndarray) -> dict:
        """Calculate Signal-to-Noise Ratio using spectral analysis."""
        if len(signal_data) < self.fs:
            return {'total_snr_db': 0, 'band_snr': {}, 'signal_power': 0, 'noise_power': 0}

        # Calculate power spectral density
        freqs, psd = signal.welch(signal_data, self.fs, nperseg=min(len(signal_data) // 4, self.fs),
                                  noverlap=None)

        # Calculate power in signal bands (delta to gamma)
        signal_power = 0
        band_snr = {}

        for band_name, (low, high) in SIGNAL_BANDS.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            signal_power += band_power
            band_snr[band_name] = band_power

        # Calculate noise power (high frequency)
        noise_mask = (freqs >= NOISE_BAND[0]) & (freqs <= NOISE_BAND[1])
        noise_power = np.trapz(psd[noise_mask], freqs[noise_mask])

        # Total SNR
        total_snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0

        # Band-specific SNR
        for band_name in band_snr:
            band_snr[band_name] = 10 * np.log10(band_snr[band_name] / noise_power) if noise_power > 0 else 0

        return {
            'total_snr_db': total_snr_db,
            'band_snr': band_snr,
            'signal_power': signal_power,
            'noise_power': noise_power
        }


class OnlineFilter:
    """Online filtering to eliminate edge artifacts."""

    def __init__(self, sample_rate: int):
        self.fs = sample_rate

        # Design filters
        nyq = sample_rate / 2

        # Bandpass: 0.5-40 Hz (4th order Butterworth)
        self.bp_sos = signal.butter(4, [0.5 / nyq, 40 / nyq], btype='band', output='sos')

        # Notch: 60 Hz (2nd order)
        notch_b, notch_a = signal.iirnotch(60, 30, sample_rate)
        self.notch_sos = signal.tf2sos(notch_b, notch_a)

        # Initialize filter states for both channels
        self.bp_zi_ch1 = signal.sosfilt_zi(self.bp_sos)
        self.bp_zi_ch2 = signal.sosfilt_zi(self.bp_sos)
        self.notch_zi_ch1 = signal.sosfilt_zi(self.notch_sos)
        self.notch_zi_ch2 = signal.sosfilt_zi(self.notch_sos)

        self.initialized = False

    def filter_chunk(self, ch1_data: np.ndarray, ch2_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply filtering to new data chunk while preserving filter state."""

        if not self.initialized:
            # Initialize filter states with first sample
            self.bp_zi_ch1 *= ch1_data[0]
            self.bp_zi_ch2 *= ch2_data[0]
            self.notch_zi_ch1 *= ch1_data[0]
            self.notch_zi_ch2 *= ch2_data[0]
            self.initialized = True

        # Apply bandpass filter
        ch1_bp, self.bp_zi_ch1 = signal.sosfilt(self.bp_sos, ch1_data, zi=self.bp_zi_ch1)
        ch2_bp, self.bp_zi_ch2 = signal.sosfilt(self.bp_sos, ch2_data, zi=self.bp_zi_ch2)

        # Apply notch filter
        ch1_filt, self.notch_zi_ch1 = signal.sosfilt(self.notch_sos, ch1_bp, zi=self.notch_zi_ch1)
        ch2_filt, self.notch_zi_ch2 = signal.sosfilt(self.notch_sos, ch2_bp, zi=self.notch_zi_ch2)

        return ch1_filt, ch2_filt


# ═══════════════════════════════════════════════════════════════════════════════
# BLE Protocol Functions (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def build_command(feature_id: int, pdu_id: int, payload: bytes = b"") -> bytes:
    command_id = (feature_id << 9) | (PDU_TYPE_COMMAND << 7) | pdu_id
    return command_id.to_bytes(2, 'big') + payload


def build_stream_command(start: bool) -> bytes:
    payload = b"\x01" if start else b"\x00"
    return build_command(FEATURE_SENSOR_CFG, CMD_STREAM_CTRL, payload)


async def find_device(target_mac: Optional[str] = None) -> str:
    if target_mac:
        print(f"Connecting to specified device: {target_mac}")
        return target_mac

    print("Scanning for Neocore EEG device...")
    devices = await BleakScanner.discover(timeout=10.0)

    for device in devices:
        if device.name and any(name in device.name.upper() for name in TARGET_NAMES):
            print(f"Found {device.name} at {device.address}")
            return device.address

    raise RuntimeError("No Neocore device found.")


def parse_eeg_packet(packet_data: bytes) -> Tuple[List[float], List[float]]:
    if len(packet_data) < 4:
        raise ValueError(f"Packet too short: {len(packet_data)} bytes")

    cmd = packet_data[0]
    data_len = packet_data[1]
    msg_index = struct.unpack('<H', packet_data[2:4])[0]

    if cmd != 0x02:
        raise ValueError(f"Unexpected command: 0x{cmd:02x}")

    sample_data = packet_data[4:4 + data_len]
    expected_len = SAMPLES_PER_CHUNK * NUM_CHANNELS * 4

    if len(sample_data) != expected_len:
        raise ValueError(f"Expected {expected_len} sample bytes, got {len(sample_data)}")

    ch1_samples = []
    ch2_samples = []

    for i in range(0, len(sample_data), 8):
        ch1_value = struct.unpack('<i', sample_data[i:i + 4])[0]
        ch2_value = struct.unpack('<i', sample_data[i + 4:i + 8])[0]
        ch1_samples.append(float(ch1_value))
        ch2_samples.append(float(ch2_value))

    return ch1_samples, ch2_samples


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced EEG Plotter with Quality Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedEEGPlotter:
    """Enhanced real-time EEG plotter with dynamic range and SNR analysis."""

    def __init__(self):
        # Data storage
        buffer_size = SAMPLE_RATE * 20  # 20 seconds
        self.ch1_raw = deque(maxlen=buffer_size)
        self.ch2_raw = deque(maxlen=buffer_size)
        self.ch1_filtered = deque(maxlen=buffer_size)
        self.ch2_filtered = deque(maxlen=buffer_size)

        # Signal processing
        self.filter = OnlineFilter(SAMPLE_RATE)
        self.quality_analyzer = SignalQualityAnalyzer(SAMPLE_RATE)

        # Update control
        self.update_counter = 0
        self.update_every_n_packets = 3
        self.quality_update_counter = 0
        self.quality_update_every = 25  # Update quality metrics every 25 packets

        # Statistics
        self.packet_count = 0
        self.start_time = time.time()

        # Time axis
        self.time_axis = np.arange(-DISPLAY_SAMPLES, 0) / SAMPLE_RATE

        # Set up enhanced matplotlib layout
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))

        # Create subplots with more detailed layout
        gs = self.fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[2, 2, 1])

        # Time domain plots
        self.ax_raw = self.fig.add_subplot(gs[0, :2])
        self.ax_filtered = self.fig.add_subplot(gs[1, :2])

        # PSD plot
        self.ax_psd = self.fig.add_subplot(gs[:2, 2])

        # Quality metrics display
        self.ax_metrics = self.fig.add_subplot(gs[2, :])
        self.ax_metrics.axis('off')

        # Setup time domain plots
        self._setup_time_plots()

        # Setup PSD plot
        self._setup_psd_plot()

        # Setup metrics display
        self._setup_metrics_display()

        plt.tight_layout()
        plt.show(block=False)

        print("Enhanced EEG plotter initialized with quality metrics...")

    def _setup_time_plots(self):
        """Setup time domain plot areas."""
        # Raw signal plot
        self.line_raw_ch1, = self.ax_raw.plot([], [], 'b-', label='Raw Ch1', linewidth=0.8, alpha=0.7)
        self.line_raw_ch2, = self.ax_raw.plot([], [], 'r-', label='Raw Ch2', linewidth=0.8, alpha=0.7)

        self.ax_raw.set_title('Raw EEG Signal', fontweight='bold')
        self.ax_raw.set_ylabel('Amplitude (µV)')
        self.ax_raw.legend(loc='upper right')
        self.ax_raw.grid(True, alpha=0.3)
        self.ax_raw.set_xlim(self.time_axis[0], self.time_axis[-1])

        # Filtered signal plot
        self.line_filt_ch1, = self.ax_filtered.plot([], [], 'b-', label='Filtered Ch1', linewidth=0.8)
        self.line_filt_ch2, = self.ax_filtered.plot([], [], 'r-', label='Filtered Ch2', linewidth=0.8)

        self.ax_filtered.set_title('Filtered EEG Signal (0.5-40 Hz + 60Hz Notch)', fontweight='bold')
        self.ax_filtered.set_xlabel('Time (seconds)')
        self.ax_filtered.set_ylabel('Amplitude (µV)')
        self.ax_filtered.legend(loc='upper right')
        self.ax_filtered.grid(True, alpha=0.3)
        self.ax_filtered.set_xlim(self.time_axis[0], self.time_axis[-1])

    def _setup_psd_plot(self):
        """Setup power spectral density plot."""
        self.psd_raw_ch1, = self.ax_psd.semilogy([], [], 'b--', label='Raw Ch1', linewidth=1, alpha=0.7)
        self.psd_raw_ch2, = self.ax_psd.semilogy([], [], 'r--', label='Raw Ch2', linewidth=1, alpha=0.7)
        self.psd_filt_ch1, = self.ax_psd.semilogy([], [], 'b-', label='Filtered Ch1', linewidth=1.2)
        self.psd_filt_ch2, = self.ax_psd.semilogy([], [], 'r-', label='Filtered Ch2', linewidth=1.2)

        self.ax_psd.set_title('Power Spectral Density', fontweight='bold')
        self.ax_psd.set_xlabel('Frequency (Hz)')
        self.ax_psd.set_ylabel('Power (µV²/Hz)')
        self.ax_psd.set_xlim(0, 80)
        self.ax_psd.legend(loc='upper right', fontsize=8)
        self.ax_psd.grid(True, alpha=0.3)

        # Add frequency band markers
        band_colors = {'delta': 'purple', 'theta': 'blue', 'alpha': 'green', 'beta': 'orange', 'gamma': 'red'}
        for band_name, (low, high) in SIGNAL_BANDS.items():
            self.ax_psd.axvspan(low, high, alpha=0.1, color=band_colors[band_name], label=f'{band_name}')

    def _setup_metrics_display(self):
        """Setup quality metrics text display."""
        self.metrics_text = self.ax_metrics.text(0.02, 0.5, '', fontsize=10, fontfamily='monospace',
                                                 verticalalignment='center', transform=self.ax_metrics.transAxes)

    def add_data(self, ch1_data: List[float], ch2_data: List[float]):
        """Add new data and apply online filtering."""
        # Convert to numpy arrays
        ch1_array = np.array(ch1_data)
        ch2_array = np.array(ch2_data)

        # Apply online filtering
        ch1_filt, ch2_filt = self.filter.filter_chunk(ch1_array, ch2_array)

        # Store data
        self.ch1_raw.extend(ch1_data)
        self.ch2_raw.extend(ch2_data)
        self.ch1_filtered.extend(ch1_filt)
        self.ch2_filtered.extend(ch2_filt)

        self.packet_count += 1
        self.update_counter += 1
        self.quality_update_counter += 1

        # Update display
        if self.update_counter >= self.update_every_n_packets:
            self.update_counter = 0
            self._update_plots()

        # Update quality metrics less frequently
        if self.quality_update_counter >= self.quality_update_every:
            self.quality_update_counter = 0
            self._update_quality_metrics()

    def _update_plots(self):
        """Update plot displays."""
        if len(self.ch1_filtered) < DISPLAY_SAMPLES:
            return

        try:
            # Get display data
            display_raw_ch1 = np.array(list(self.ch1_raw)[-DISPLAY_SAMPLES:])
            display_raw_ch2 = np.array(list(self.ch2_raw)[-DISPLAY_SAMPLES:])
            display_filt_ch1 = np.array(list(self.ch1_filtered)[-DISPLAY_SAMPLES:])
            display_filt_ch2 = np.array(list(self.ch2_filtered)[-DISPLAY_SAMPLES:])

            # Update time domain plots
            self.line_raw_ch1.set_data(self.time_axis, display_raw_ch1)
            self.line_raw_ch2.set_data(self.time_axis, display_raw_ch2)
            self.line_filt_ch1.set_data(self.time_axis, display_filt_ch1)
            self.line_filt_ch2.set_data(self.time_axis, display_filt_ch2)

            # Auto-scale Y axes
            self._autoscale_axes(display_raw_ch1, display_raw_ch2, display_filt_ch1, display_filt_ch2)

            # Update PSD plot
            if self.packet_count % 15 == 0:
                self._update_psd_plot()

            # Efficient drawing
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        except Exception as e:
            print(f"Plot update error: {e}")

    def _autoscale_axes(self, raw_ch1, raw_ch2, filt_ch1, filt_ch2):
        """Auto-scale Y axes for both raw and filtered signals."""
        # Raw signal scaling
        raw_max = max(
            np.percentile(np.abs(raw_ch1), 95),
            np.percentile(np.abs(raw_ch2), 95),
            100  # Minimum scale
        )
        current_raw_ylim = self.ax_raw.get_ylim()
        if abs(current_raw_ylim[1] - raw_max * 1.1) > raw_max * 0.3:
            self.ax_raw.set_ylim(-raw_max * 1.1, raw_max * 1.1)

        # Filtered signal scaling
        filt_max = max(
            np.percentile(np.abs(filt_ch1), 95),
            np.percentile(np.abs(filt_ch2), 95),
            50  # Minimum scale
        )
        current_filt_ylim = self.ax_filtered.get_ylim()
        if abs(current_filt_ylim[1] - filt_max * 1.1) > filt_max * 0.3:
            self.ax_filtered.set_ylim(-filt_max * 1.1, filt_max * 1.1)

    def _update_psd_plot(self):
        """Update power spectral density plot."""
        psd_samples = min(len(self.ch1_filtered), SAMPLE_RATE * 8)
        if psd_samples < SAMPLE_RATE * 2:
            return

        # Get data for PSD analysis
        psd_raw_ch1 = np.array(list(self.ch1_raw)[-psd_samples:])
        psd_raw_ch2 = np.array(list(self.ch2_raw)[-psd_samples:])
        psd_filt_ch1 = np.array(list(self.ch1_filtered)[-psd_samples:])
        psd_filt_ch2 = np.array(list(self.ch2_filtered)[-psd_samples:])

        # Calculate PSDs
        freqs, psd_raw_1 = signal.welch(psd_raw_ch1, SAMPLE_RATE, nperseg=SAMPLE_RATE, noverlap=SAMPLE_RATE // 2)
        _, psd_raw_2 = signal.welch(psd_raw_ch2, SAMPLE_RATE, nperseg=SAMPLE_RATE, noverlap=SAMPLE_RATE // 2)
        _, psd_filt_1 = signal.welch(psd_filt_ch1, SAMPLE_RATE, nperseg=SAMPLE_RATE, noverlap=SAMPLE_RATE // 2)
        _, psd_filt_2 = signal.welch(psd_filt_ch2, SAMPLE_RATE, nperseg=SAMPLE_RATE, noverlap=SAMPLE_RATE // 2)

        # Update PSD lines
        self.psd_raw_ch1.set_data(freqs, psd_raw_1)
        self.psd_raw_ch2.set_data(freqs, psd_raw_2)
        self.psd_filt_ch1.set_data(freqs, psd_filt_1)
        self.psd_filt_ch2.set_data(freqs, psd_filt_2)

        # Auto-scale PSD
        all_psds = [psd_raw_1, psd_raw_2, psd_filt_1, psd_filt_2]
        valid_psds = [p[p > 0] for p in all_psds if len(p[p > 0]) > 0]
        if valid_psds:
            psd_min = min(np.percentile(p, 5) for p in valid_psds)
            psd_max = max(np.percentile(p, 95) for p in valid_psds)
            self.ax_psd.set_ylim(psd_min * 0.1, psd_max * 10)

    def _update_quality_metrics(self):
        """Update signal quality metrics display."""
        if len(self.ch1_raw) < SAMPLE_RATE * 2:
            return

        try:
            # Get recent data for analysis (last 5 seconds)
            analysis_samples = min(len(self.ch1_raw), SAMPLE_RATE * 5)

            raw_ch1 = np.array(list(self.ch1_raw)[-analysis_samples:])
            raw_ch2 = np.array(list(self.ch2_raw)[-analysis_samples:])
            filt_ch1 = np.array(list(self.ch1_filtered)[-analysis_samples:])
            filt_ch2 = np.array(list(self.ch2_filtered)[-analysis_samples:])

            # Calculate quality metrics
            dr_raw_ch1 = self.quality_analyzer.calculate_dynamic_range(raw_ch1)
            dr_raw_ch2 = self.quality_analyzer.calculate_dynamic_range(raw_ch2)
            dr_filt_ch1 = self.quality_analyzer.calculate_dynamic_range(filt_ch1)
            dr_filt_ch2 = self.quality_analyzer.calculate_dynamic_range(filt_ch2)

            snr_raw_ch1 = self.quality_analyzer.calculate_snr(raw_ch1)
            snr_raw_ch2 = self.quality_analyzer.calculate_snr(raw_ch2)
            snr_filt_ch1 = self.quality_analyzer.calculate_snr(filt_ch1)
            snr_filt_ch2 = self.quality_analyzer.calculate_snr(filt_ch2)

            # Format metrics text
            metrics_text = self._format_metrics_text(
                dr_raw_ch1, dr_raw_ch2, dr_filt_ch1, dr_filt_ch2,
                snr_raw_ch1, snr_raw_ch2, snr_filt_ch1, snr_filt_ch2
            )

            self.metrics_text.set_text(metrics_text)

        except Exception as e:
            print(f"Quality metrics update error: {e}")

    def _format_metrics_text(self, dr_raw_ch1, dr_raw_ch2, dr_filt_ch1, dr_filt_ch2,
                             snr_raw_ch1, snr_raw_ch2, snr_filt_ch1, snr_filt_ch2):
        """Format quality metrics for display."""

        text = "SIGNAL QUALITY METRICS (Last 5 seconds)\n"
        text += "=" * 120 + "\n"

        # Dynamic Range
        text += f"DYNAMIC RANGE:  "
        text += f"CH1 Raw: {dr_raw_ch1['peak_to_peak']:.1f}µV ({dr_raw_ch1['db']:.1f}dB)  |  "
        text += f"CH1 Filtered: {dr_filt_ch1['peak_to_peak']:.1f}µV ({dr_filt_ch1['db']:.1f}dB)  |  "
        text += f"CH2 Raw: {dr_raw_ch2['peak_to_peak']:.1f}µV ({dr_raw_ch2['db']:.1f}dB)  |  "
        text += f"CH2 Filtered: {dr_filt_ch2['peak_to_peak']:.1f}µV ({dr_filt_ch2['db']:.1f}dB)\n"

        # SNR
        text += f"SIGNAL-TO-NOISE RATIO:  "
        text += f"CH1 Raw: {snr_raw_ch1['total_snr_db']:.1f}dB  |  "
        text += f"CH1 Filtered: {snr_filt_ch1['total_snr_db']:.1f}dB  |  "
        text += f"CH2 Raw: {snr_raw_ch2['total_snr_db']:.1f}dB  |  "
        text += f"CH2 Filtered: {snr_filt_ch2['total_snr_db']:.1f}dB\n"

        # RMS Values
        text += f"RMS VALUES:  "
        text += f"CH1 Raw: {dr_raw_ch1['rms']:.1f}µV  |  "
        text += f"CH1 Filtered: {dr_filt_ch1['rms']:.1f}µV  |  "
        text += f"CH2 Raw: {dr_raw_ch2['rms']:.1f}µV  |  "
        text += f"CH2 Filtered: {dr_filt_ch2['rms']:.1f}µV\n"

        # Band-specific SNR for filtered signals
        text += f"BAND SNR (Filtered):  "
        for band in ['alpha', 'beta', 'gamma']:
            if band in snr_filt_ch1['band_snr'] and band in snr_filt_ch2['band_snr']:
                text += f"{band.upper()}: CH1={snr_filt_ch1['band_snr'][band]:.1f}dB CH2={snr_filt_ch2['band_snr'][band]:.1f}dB  |  "

        return text

    def is_active(self) -> bool:
        return plt.fignum_exists(self.fig.number)


# ═══════════════════════════════════════════════════════════════════════════════
# BLE Communication Handler
# ═══════════════════════════════════════════════════════════════════════════════

class EEGStreamer:
    def __init__(self):
        self.plotter = EnhancedEEGPlotter()

    def notification_handler(self, sender: int, data: bytearray):
        try:
            if len(data) < 6:
                return
            ch1_samples, ch2_samples = parse_eeg_packet(data[2:])
            self.plotter.add_data(ch1_samples, ch2_samples)
        except Exception as e:
            print(f"Data parsing error: {e}")

    async def stream_data(self, device_address: str):
        print(f"Connecting to {device_address}...")

        async with BleakClient(device_address, timeout=20.0) as client:
            if not client.is_connected:
                raise RuntimeError("Failed to connect to device")

            print("Connected! Setting up data stream...")

            try:
                await client.request_mtu(247)
            except:
                pass

            await client.start_notify(TX_UUID, self.notification_handler)

            start_cmd = build_stream_command(True)
            await client.write_gatt_char(RX_UUID, start_cmd, response=False)
            print("EEG streaming started with enhanced quality metrics!")

            try:
                while self.plotter.is_active() and client.is_connected:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping stream...")
            finally:
                if client.is_connected:
                    stop_cmd = build_stream_command(False)
                    await client.write_gatt_char(RX_UUID, stop_cmd, response=False)
                    await client.stop_notify(TX_UUID)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("Neocore EEG Live Plotter - Enhanced with Quality Metrics")
    print("Features: Dynamic Range Analysis, SNR Calculation, Real-time Quality Monitoring")
    print("=" * 80)

    target_mac = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if ':' in arg:
                target_mac = arg.upper()
                break

    try:
        device_address = await find_device(target_mac)
        streamer = EEGStreamer()
        await streamer.stream_data(device_address)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBye!")
        sys.exit(0)