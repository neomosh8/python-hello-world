#!/usr/bin/env python3
"""
Neocore EEG Live Plotter - Stable real-time visualization
Fixed version that eliminates edge jiggling artifacts.
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


# ═══════════════════════════════════════════════════════════════════════════════
# Fixed Signal Processing (eliminates jiggling)
# ═══════════════════════════════════════════════════════════════════════════════

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
# Stable Real-time Plotter (fixed jiggling)
# ═══════════════════════════════════════════════════════════════════════════════

class StableEEGPlotter:
    """Stable real-time EEG plotter without edge artifacts."""

    def __init__(self):
        # Data storage
        buffer_size = SAMPLE_RATE * 20  # 20 seconds
        self.ch1_raw = deque(maxlen=buffer_size)
        self.ch2_raw = deque(maxlen=buffer_size)
        self.ch1_filtered = deque(maxlen=buffer_size)
        self.ch2_filtered = deque(maxlen=buffer_size)

        # Online filter
        self.filter = OnlineFilter(SAMPLE_RATE)

        # Update control - reduce update frequency
        self.update_counter = 0
        self.update_every_n_packets = 3  # Update display every 3 packets (~3Hz)

        # Statistics
        self.packet_count = 0
        self.start_time = time.time()

        # Fixed time axis (relative to newest sample)
        self.time_axis = np.arange(-DISPLAY_SAMPLES, 0) / SAMPLE_RATE

        # Set up matplotlib with better performance
        plt.ion()
        self.fig, (self.ax_time, self.ax_psd) = plt.subplots(2, 1, figsize=(12, 8))

        # Time domain plot
        self.line_ch1, = self.ax_time.plot([], [], 'b-', label='Channel 1', linewidth=0.8)
        self.line_ch2, = self.ax_time.plot([], [], 'r-', label='Channel 2', linewidth=0.8)

        self.ax_time.set_title('Filtered EEG Signal (0.5-40 Hz + 60Hz Notch)')
        self.ax_time.set_xlabel('Time (seconds)')
        self.ax_time.set_ylabel('Amplitude (µV)')
        self.ax_time.legend()
        self.ax_time.grid(True, alpha=0.3)

        # Set fixed time axis
        self.ax_time.set_xlim(self.time_axis[0], self.time_axis[-1])

        # PSD plot
        self.psd_ch1, = self.ax_psd.semilogy([], [], 'b-', label='Channel 1', linewidth=0.8)
        self.psd_ch2, = self.ax_psd.semilogy([], [], 'r-', label='Channel 2', linewidth=0.8)

        self.ax_psd.set_title('Power Spectral Density')
        self.ax_psd.set_xlabel('Frequency (Hz)')
        self.ax_psd.set_ylabel('Power (µV²/Hz)')
        self.ax_psd.set_xlim(0, 50)
        self.ax_psd.legend()
        self.ax_psd.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)

        print("Stable plotter initialized...")

    def add_data(self, ch1_data: List[float], ch2_data: List[float]):
        """Add new data and apply online filtering."""
        # Convert to numpy arrays
        ch1_array = np.array(ch1_data)
        ch2_array = np.array(ch2_data)

        # Apply online filtering (maintains filter state between chunks)
        ch1_filt, ch2_filt = self.filter.filter_chunk(ch1_array, ch2_array)

        # Store data
        self.ch1_raw.extend(ch1_data)
        self.ch2_raw.extend(ch2_data)
        self.ch1_filtered.extend(ch1_filt)
        self.ch2_filtered.extend(ch2_filt)

        self.packet_count += 1
        self.update_counter += 1

        # Update display less frequently to reduce jitter
        if self.update_counter >= self.update_every_n_packets:
            self.update_counter = 0
            self._update_plots()

    def _update_plots(self):
        """Update plot displays."""
        if len(self.ch1_filtered) < DISPLAY_SAMPLES:
            return

        try:
            # Get display data (most recent samples)
            display_ch1 = np.array(list(self.ch1_filtered)[-DISPLAY_SAMPLES:])
            display_ch2 = np.array(list(self.ch2_filtered)[-DISPLAY_SAMPLES:])

            # Update time domain plot (time axis is fixed)
            self.line_ch1.set_data(self.time_axis, display_ch1)
            self.line_ch2.set_data(self.time_axis, display_ch2)

            # Auto-scale Y axis with some stability
            y_max = max(
                np.percentile(np.abs(display_ch1), 95),  # Use 95th percentile instead of max
                np.percentile(np.abs(display_ch2), 95),
                50  # Minimum scale
            )
            current_ylim = self.ax_time.get_ylim()

            # Only update Y limits if change is significant (reduces jitter)
            if abs(current_ylim[1] - y_max * 1.1) > y_max * 0.2:
                self.ax_time.set_ylim(-y_max * 1.1, y_max * 1.1)

            # Update PSD plot (use more data, update less frequently)
            if self.packet_count % 15 == 0:  # Update PSD every 15 packets (~1.5 seconds)
                psd_samples = min(len(self.ch1_filtered), SAMPLE_RATE * 8)
                if psd_samples >= SAMPLE_RATE * 2:  # Need at least 2 seconds
                    psd_data_ch1 = np.array(list(self.ch1_filtered)[-psd_samples:])
                    psd_data_ch2 = np.array(list(self.ch2_filtered)[-psd_samples:])

                    # Welch PSD with overlap
                    freqs, psd_ch1 = signal.welch(
                        psd_data_ch1,
                        SAMPLE_RATE,
                        nperseg=SAMPLE_RATE,
                        noverlap=SAMPLE_RATE // 2
                    )
                    _, psd_ch2 = signal.welch(
                        psd_data_ch2,
                        SAMPLE_RATE,
                        nperseg=SAMPLE_RATE,
                        noverlap=SAMPLE_RATE // 2
                    )

                    self.psd_ch1.set_data(freqs, psd_ch1)
                    self.psd_ch2.set_data(freqs, psd_ch2)

                    # Stable PSD scaling
                    valid_psd1 = psd_ch1[psd_ch1 > 0]
                    valid_psd2 = psd_ch2[psd_ch2 > 0]
                    if len(valid_psd1) > 0 and len(valid_psd2) > 0:
                        psd_min = min(np.percentile(valid_psd1, 5), np.percentile(valid_psd2, 5))
                        psd_max = max(np.percentile(psd_ch1, 95), np.percentile(psd_ch2, 95))
                        self.ax_psd.set_ylim(psd_min * 0.5, psd_max * 2)

            # Efficient drawing
            self.ax_time.draw_artist(self.line_ch1)
            self.ax_time.draw_artist(self.line_ch2)
            self.fig.canvas.blit(self.ax_time.bbox)
            self.fig.canvas.flush_events()

            # Status update
            if self.packet_count % 50 == 0:
                elapsed = time.time() - self.start_time
                rate = self.packet_count / elapsed
                print(f"Packets: {self.packet_count}, Rate: {rate:.1f} Hz, "
                      f"Buffer: {len(self.ch1_filtered)} samples")

        except Exception as e:
            print(f"Plot update error: {e}")

    def is_active(self) -> bool:
        return plt.fignum_exists(self.fig.number)


# ═══════════════════════════════════════════════════════════════════════════════
# BLE Communication Handler
# ═══════════════════════════════════════════════════════════════════════════════

class EEGStreamer:
    def __init__(self):
        self.plotter = StableEEGPlotter()

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
            print("EEG streaming started - signal should be stable now!")

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
    print("Neocore EEG Live Plotter - Stable Version")
    print("=" * 50)

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