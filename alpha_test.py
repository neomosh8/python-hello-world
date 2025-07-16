#!/usr/bin/env python3
"""
Neocore EEG Alpha Band Comparison - Eye Open vs Eye Close
Simple data collection and alpha band analysis.
"""

import asyncio
import sys
import struct
import time
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient

# ═══════════════════════════════════════════════════════════════════════════════
# BLE Configuration (from original code)
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
RECORDING_TIME_MIN = 0.5  # 1 minute for each condition
ALPHA_BAND = (8, 12)  # Alpha frequency range


# ═══════════════════════════════════════════════════════════════════════════════
# BLE Protocol Functions (from original code)
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
# Simple Data Collector
# ═══════════════════════════════════════════════════════════════════════════════

class EEGDataCollector:
    """Simple data collector for alpha band analysis."""

    def __init__(self):
        self.ch1_data = []
        self.ch2_data = []
        self.recording = False
        self.start_time = None
        self.current_condition = ""

    def start_recording(self, condition: str):
        """Start recording for a specific condition."""
        self.ch1_data = []
        self.ch2_data = []
        self.recording = True
        self.start_time = time.time()
        self.current_condition = condition

    def stop_recording(self):
        """Stop recording."""
        self.recording = False

    def add_data(self, ch1_samples: List[float], ch2_samples: List[float]):
        """Add new data samples."""
        if self.recording:
            self.ch1_data.extend(ch1_samples)
            self.ch2_data.extend(ch2_samples)

    def get_recording_time(self) -> float:
        """Get current recording time in seconds."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get recorded data as numpy arrays."""
        return np.array(self.ch1_data), np.array(self.ch2_data)


# ═══════════════════════════════════════════════════════════════════════════════
# Signal Processing
# ═══════════════════════════════════════════════════════════════════════════════

def filter_signal(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply bandpass filter to remove artifacts."""
    nyq = sample_rate / 2
    # Bandpass filter: 1-40 Hz
    sos = signal.butter(4, [1 / nyq, 40 / nyq], btype='band', output='sos')
    return signal.sosfilt(sos, data)


def calculate_alpha_power(data: np.ndarray, sample_rate: int) -> float:
    """Calculate alpha band (8-12 Hz) power using Welch's method."""
    # Use Welch's method for PSD estimation
    freqs, psd = signal.welch(data, sample_rate, nperseg=sample_rate * 2, noverlap=sample_rate)

    # Find alpha band indices
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])

    # Calculate mean power in alpha band
    alpha_power = np.mean(psd[alpha_mask])

    return alpha_power


def process_condition_data(ch1_data: np.ndarray, ch2_data: np.ndarray) -> Tuple[float, float]:
    """Process data from one condition and return alpha power for both channels."""
    # Filter the data
    ch1_filtered = filter_signal(ch1_data, SAMPLE_RATE)
    ch2_filtered = filter_signal(ch2_data, SAMPLE_RATE)

    # Calculate alpha power
    ch1_alpha = calculate_alpha_power(ch1_filtered, SAMPLE_RATE)
    ch2_alpha = calculate_alpha_power(ch2_filtered, SAMPLE_RATE)

    return ch1_alpha, ch2_alpha


# ═══════════════════════════════════════════════════════════════════════════════
# Main Recording Application
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaExperiment:
    def __init__(self):
        self.collector = EEGDataCollector()
        self.eye_open_data = None
        self.eye_close_data = None

    def notification_handler(self, sender: int, data: bytearray):
        """Handle incoming EEG data."""
        try:
            if len(data) < 6:
                return
            ch1_samples, ch2_samples = parse_eeg_packet(data[2:])
            self.collector.add_data(ch1_samples, ch2_samples)
        except Exception as e:
            print(f"Data parsing error: {e}")

    async def run_experiment(self, device_address: str):
        """Run the complete eye open/close experiment."""
        print(f"Connecting to {device_address}...")

        async with BleakClient(device_address, timeout=20.0) as client:
            if not client.is_connected:
                raise RuntimeError("Failed to connect to device")

            print("Connected! Setting up data stream...")

            # Start notifications
            await client.start_notify(TX_UUID, self.notification_handler)

            # Start streaming
            start_cmd = build_stream_command(True)
            await client.write_gatt_char(RX_UUID, start_cmd, response=False)

            try:
                # Record Eye Open condition
                await self.record_condition("EYE OPEN", client)

                # Short break
                print("\n" + "=" * 50)
                print("Taking a 10-second break...")
                print("=" * 50)
                await asyncio.sleep(10)

                # Record Eye Close condition
                await self.record_condition("EYE CLOSE", client)

                # Analyze results
                self.analyze_results()

            finally:
                # Stop streaming
                stop_cmd = build_stream_command(False)
                await client.write_gatt_char(RX_UUID, stop_cmd, response=False)
                await client.stop_notify(TX_UUID)

    async def record_condition(self, condition: str, client):
        """Record data for one condition."""
        print(f"\n{'=' * 50}")
        print(f"PREPARE FOR: {condition}")
        print(f"{'=' * 50}")
        print("Recording will start in 5 seconds...")

        for i in range(5, 0, -1):
            print(f"{i}...")
            await asyncio.sleep(1)

        print(f"\n🔴 RECORDING {condition} - Keep your eyes {'OPEN' if 'OPEN' in condition else 'CLOSED'}!")
        print("=" * 50)

        # Start recording
        self.collector.start_recording(condition)

        # Record for specified time
        recording_time = RECORDING_TIME_MIN * 60  # Convert to seconds
        start_time = time.time()

        while time.time() - start_time < recording_time:
            remaining = recording_time - (time.time() - start_time)
            print(f"\rRecording {condition}: {remaining:.1f}s remaining", end="", flush=True)
            await asyncio.sleep(0.1)

        # Stop recording
        self.collector.stop_recording()
        ch1_data, ch2_data = self.collector.get_data()

        # Store data
        if condition == "EYE OPEN":
            self.eye_open_data = (ch1_data, ch2_data)
        else:
            self.eye_close_data = (ch1_data, ch2_data)

        print(f"\n✅ {condition} recording complete! Collected {len(ch1_data)} samples")

    def analyze_results(self):
        """Analyze and plot the alpha band comparison."""
        if self.eye_open_data is None or self.eye_close_data is None:
            print("Error: Missing data for analysis")
            return

        print("\n" + "=" * 50)
        print("ANALYZING ALPHA BAND POWER...")
        print("=" * 50)

        # Process eye open data
        ch1_open_alpha, ch2_open_alpha = process_condition_data(*self.eye_open_data)

        # Process eye close data
        ch1_close_alpha, ch2_close_alpha = process_condition_data(*self.eye_close_data)

        # Print results
        print(f"\nRESULTS:")
        print(f"Channel 1 - Eye Open: {ch1_open_alpha:.2f} µV²")
        print(f"Channel 1 - Eye Close: {ch1_close_alpha:.2f} µV²")
        print(f"Channel 1 - Ratio (Close/Open): {ch1_close_alpha / ch1_open_alpha:.2f}")
        print()
        print(f"Channel 2 - Eye Open: {ch2_open_alpha:.2f} µV²")
        print(f"Channel 2 - Eye Close: {ch2_close_alpha:.2f} µV²")
        print(f"Channel 2 - Ratio (Close/Open): {ch2_close_alpha / ch2_open_alpha:.2f}")

        # Create comparison plot
        self.plot_comparison(ch1_open_alpha, ch1_close_alpha, ch2_open_alpha, ch2_close_alpha)

    def plot_comparison(self, ch1_open, ch1_close, ch2_open, ch2_close):
        """Create a comparison plot of alpha band power."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Channel 1 comparison
        conditions = ['Eye Open', 'Eye Close']
        ch1_values = [ch1_open, ch1_close]
        colors = ['skyblue', 'lightcoral']

        bars1 = ax1.bar(conditions, ch1_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Channel 1 - Alpha Band Power (8-12 Hz)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Power (µV²)', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, ch1_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        # Channel 2 comparison
        ch2_values = [ch2_open, ch2_close]
        bars2 = ax2.bar(conditions, ch2_values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Channel 2 - Alpha Band Power (8-12 Hz)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Power (µV²)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars2, ch2_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.suptitle('Alpha Band Power Comparison: Eye Open vs Eye Close',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("Neocore EEG Alpha Band Experiment")
    print("Eye Open vs Eye Close Analysis")
    print("=" * 50)
    print(f"Recording time per condition: {RECORDING_TIME_MIN} minute(s)")
    print(f"Alpha band frequency: {ALPHA_BAND[0]}-{ALPHA_BAND[1]} Hz")
    print("=" * 50)

    target_mac = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if ':' in arg:
                target_mac = arg.upper()
                break

    try:
        device_address = await find_device(target_mac)
        experiment = AlphaExperiment()
        await experiment.run_experiment(device_address)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nExperiment interrupted!")
        sys.exit(0)