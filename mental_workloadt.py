#!/usr/bin/env python3
"""
Neocore EEG Mental Workload Assessment - Arithmetic Task
Rest vs Mental Arithmetic Beta Band Analysis
"""

import asyncio
import sys
import struct
import time
import random
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
# Task Parameters
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 250
SAMPLES_PER_CHUNK = 27
NUM_CHANNELS = 2
NUM_EQUATIONS = 10
BETA_BAND = (13, 30)  # Beta frequency range for mental workload
REST_TIME_SEC = 60  # 1 minute rest periods


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
# Arithmetic Task Generator
# ═══════════════════════════════════════════════════════════════════════════════

class ArithmeticTask:
    """Generate random arithmetic equations of varying difficulty."""

    def __init__(self):
        self.equations = []
        self.answers = []
        self.user_responses = []
        self.response_times = []

    def generate_equations(self):
        """Generate 10 random arithmetic equations."""
        self.equations = []
        self.answers = []

        for i in range(NUM_EQUATIONS):
            # Vary difficulty: easier first, harder later
            if i < 3:  # Easy
                a, b = random.randint(10, 50), random.randint(10, 50)
                op = random.choice(['+', '-'])
            elif i < 7:  # Medium
                a, b = random.randint(15, 99), random.randint(15, 99)
                op = random.choice(['+', '-', '*'])
            else:  # Hard
                a, b = random.randint(20, 99), random.randint(2, 12)
                op = random.choice(['+', '-', '*'])

            if op == '+':
                answer = a + b
            elif op == '-':
                # Ensure positive result
                if a < b:
                    a, b = b, a
                answer = a - b
            else:  # multiplication
                answer = a * b

            equation = f"{a} {op} {b}"
            self.equations.append(equation)
            self.answers.append(answer)

    def get_equation(self, index: int) -> str:
        """Get equation by index."""
        return self.equations[index]

    def get_answer(self, index: int) -> int:
        """Get correct answer by index."""
        return self.answers[index]

    def record_response(self, response: str, response_time: float):
        """Record user response and time."""
        self.user_responses.append(response)
        self.response_times.append(response_time)

    def get_stats(self) -> dict:
        """Get performance statistics."""
        correct = 0
        total_time = sum(self.response_times)

        for i, response in enumerate(self.user_responses):
            try:
                if int(response) == self.answers[i]:
                    correct += 1
            except:
                pass

        return {
            'correct': correct,
            'total': len(self.user_responses),
            'accuracy': correct / len(self.user_responses) * 100 if self.user_responses else 0,
            'avg_time': total_time / len(self.response_times) if self.response_times else 0
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Data Collector
# ═══════════════════════════════════════════════════════════════════════════════

class MentalWorkloadCollector:
    """Data collector for mental workload assessment."""

    def __init__(self):
        self.ch1_data = []
        self.ch2_data = []
        self.recording = False
        self.start_time = None
        self.current_phase = ""

    def start_recording(self, phase: str):
        """Start recording for a specific phase."""
        self.ch1_data = []
        self.ch2_data = []
        self.recording = True
        self.start_time = time.time()
        self.current_phase = phase

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
# Signal Processing for Mental Workload
# ═══════════════════════════════════════════════════════════════════════════════

def filter_signal(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply bandpass filter to remove artifacts."""
    nyq = sample_rate / 2
    # Bandpass filter: 1-40 Hz
    sos = signal.butter(4, [1 / nyq, 40 / nyq], btype='band', output='sos')
    return signal.sosfilt(sos, data)


def calculate_beta_power(data: np.ndarray, sample_rate: int) -> float:
    """Calculate beta band (13-30 Hz) power for mental workload assessment."""
    # Use Welch's method for PSD estimation
    freqs, psd = signal.welch(data, sample_rate, nperseg=sample_rate * 2, noverlap=sample_rate)

    # Find beta band indices
    beta_mask = (freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1])

    # Calculate mean power in beta band
    beta_power = np.mean(psd[beta_mask])

    return beta_power


def process_mental_workload(ch1_data: np.ndarray, ch2_data: np.ndarray) -> Tuple[float, float]:
    """Process data and return beta power for both channels."""
    # Filter the data
    ch1_filtered = filter_signal(ch1_data, SAMPLE_RATE)
    ch2_filtered = filter_signal(ch2_data, SAMPLE_RATE)

    # Calculate beta power
    ch1_beta = calculate_beta_power(ch1_filtered, SAMPLE_RATE)
    ch2_beta = calculate_beta_power(ch2_filtered, SAMPLE_RATE)

    return ch1_beta, ch2_beta


# ═══════════════════════════════════════════════════════════════════════════════
# Mental Workload Experiment
# ═══════════════════════════════════════════════════════════════════════════════

class MentalWorkloadExperiment:
    def __init__(self):
        self.collector = MentalWorkloadCollector()
        self.task = ArithmeticTask()
        self.rest_data = None
        self.arithmetic_data = None

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
        """Run the complete mental workload experiment."""
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
                # Generate arithmetic tasks
                self.task.generate_equations()

                # Phase 1: Baseline rest
                await self.baseline_rest()

                # Phase 2: Arithmetic task
                await self.arithmetic_phase()

                # Phase 3: Recovery rest
                await self.recovery_rest()

                # Analyze results
                self.analyze_mental_workload()

            finally:
                # Stop streaming
                stop_cmd = build_stream_command(False)
                await client.write_gatt_char(RX_UUID, stop_cmd, response=False)
                await client.stop_notify(TX_UUID)

    async def baseline_rest(self):
        """Record baseline rest period."""
        print(f"\n{'=' * 60}")
        print("PHASE 1: BASELINE REST")
        print(f"{'=' * 60}")
        print("Please sit comfortably and relax.")
        print("Try to keep your mind calm and avoid mental calculations.")
        print(f"Recording will start in 5 seconds for {REST_TIME_SEC} seconds...")

        for i in range(5, 0, -1):
            print(f"{i}...")
            await asyncio.sleep(1)

        print(f"\n🔴 RECORDING BASELINE - Please relax and breathe normally")
        print("=" * 60)

        self.collector.start_recording("BASELINE")

        start_time = time.time()
        while time.time() - start_time < REST_TIME_SEC:
            remaining = REST_TIME_SEC - (time.time() - start_time)
            print(f"\rBaseline recording: {remaining:.1f}s remaining", end="", flush=True)
            await asyncio.sleep(0.1)

        self.collector.stop_recording()
        baseline_data = self.collector.get_data()
        self.rest_data = baseline_data

        print(f"\n✅ Baseline recording complete! Collected {len(baseline_data[0])} samples")

    async def arithmetic_phase(self):
        """Run arithmetic task phase."""
        print(f"\n{'=' * 60}")
        print("PHASE 2: ARITHMETIC TASK")
        print(f"{'=' * 60}")
        print(f"You will solve {NUM_EQUATIONS} arithmetic problems.")
        print("Calculate mentally and type your answer, then press Enter.")
        print("Try to be as accurate and quick as possible.")
        print("\nStarting in 5 seconds...")

        for i in range(5, 0, -1):
            print(f"{i}...")
            await asyncio.sleep(1)

        print(f"\n🔴 STARTING ARITHMETIC TASK")
        print("=" * 60)

        self.collector.start_recording("ARITHMETIC")

        # Present equations one by one
        for i in range(NUM_EQUATIONS):
            equation = self.task.get_equation(i)
            print(f"\nProblem {i + 1}/{NUM_EQUATIONS}: {equation} = ?")

            start_time = time.time()
            # Simulate user input (in real use, you'd use input())
            # For demo, we'll auto-advance after 5 seconds
            await asyncio.sleep(5)
            response_time = time.time() - start_time

            # Record dummy response (in real use: response = input("Your answer: "))
            response = str(self.task.get_answer(i))  # Auto-correct for demo
            self.task.record_response(response, response_time)

            print(f"Your answer: {response} (Time: {response_time:.1f}s)")

            # Short pause between problems
            await asyncio.sleep(1)

        self.collector.stop_recording()
        arithmetic_data = self.collector.get_data()
        self.arithmetic_data = arithmetic_data

        print(f"\n✅ Arithmetic task complete! Collected {len(arithmetic_data[0])} samples")

        # Show performance stats
        stats = self.task.get_stats()
        print(f"\nPERFORMANCE STATISTICS:")
        print(f"Accuracy: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
        print(f"Average response time: {stats['avg_time']:.1f} seconds")

    async def recovery_rest(self):
        """Record recovery rest period."""
        print(f"\n{'=' * 60}")
        print("PHASE 3: RECOVERY REST")
        print(f"{'=' * 60}")
        print("Task complete! Please relax again.")
        print(f"Final recording for {REST_TIME_SEC} seconds...")

        for i in range(5, 0, -1):
            print(f"{i}...")
            await asyncio.sleep(1)

        print(f"\n🔴 RECORDING RECOVERY - Please relax")
        print("=" * 60)

        self.collector.start_recording("RECOVERY")

        start_time = time.time()
        while time.time() - start_time < REST_TIME_SEC:
            remaining = REST_TIME_SEC - (time.time() - start_time)
            print(f"\rRecovery recording: {remaining:.1f}s remaining", end="", flush=True)
            await asyncio.sleep(0.1)

        self.collector.stop_recording()
        print(f"\n✅ Recovery recording complete!")

    def analyze_mental_workload(self):
        """Analyze mental workload using beta band power."""
        if self.rest_data is None or self.arithmetic_data is None:
            print("Error: Missing data for analysis")
            return

        print(f"\n{'=' * 60}")
        print("ANALYZING MENTAL WORKLOAD (BETA BAND POWER)")
        print(f"{'=' * 60}")

        # Process rest data
        ch1_rest_beta, ch2_rest_beta = process_mental_workload(*self.rest_data)

        # Process arithmetic data
        ch1_arith_beta, ch2_arith_beta = process_mental_workload(*self.arithmetic_data)

        # Calculate workload indices (arithmetic / rest)
        ch1_workload = ch1_arith_beta / ch1_rest_beta
        ch2_workload = ch2_arith_beta / ch2_rest_beta

        # Print results
        print(f"\nMENTAL WORKLOAD RESULTS:")
        print(f"Channel 1 - Rest: {ch1_rest_beta:.2f} µV²")
        print(f"Channel 1 - Arithmetic: {ch1_arith_beta:.2f} µV²")
        print(f"Channel 1 - Workload Index: {ch1_workload:.2f}")
        print()
        print(f"Channel 2 - Rest: {ch2_rest_beta:.2f} µV²")
        print(f"Channel 2 - Arithmetic: {ch2_arith_beta:.2f} µV²")
        print(f"Channel 2 - Workload Index: {ch2_workload:.2f}")

        # Create workload comparison plot
        self.plot_mental_workload(ch1_rest_beta, ch1_arith_beta, ch2_rest_beta, ch2_arith_beta)

    def plot_mental_workload(self, ch1_rest, ch1_arith, ch2_rest, ch2_arith):
        """Create mental workload comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Channel 1 comparison
        conditions = ['Rest\n(Baseline)', 'Arithmetic\n(Mental Load)']
        ch1_values = [ch1_rest, ch1_arith]
        colors = ['lightgreen', 'orange']

        bars1 = ax1.bar(conditions, ch1_values, color=colors, alpha=0.7, edgecolor='black', width=0.6)
        ax1.set_title('Channel 1 - Mental Workload\nBeta Band Power (13-30 Hz)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Beta Power (µV²)', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels and workload index
        for bar, value in zip(bars1, ch1_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Add workload index
        workload_ch1 = ch1_arith / ch1_rest
        ax1.text(0.5, max(ch1_values) * 0.8, f'Workload Index: {workload_ch1:.2f}x',
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # Channel 2 comparison
        ch2_values = [ch2_rest, ch2_arith]
        bars2 = ax2.bar(conditions, ch2_values, color=colors, alpha=0.7, edgecolor='black', width=0.6)
        ax2.set_title('Channel 2 - Mental Workload\nBeta Band Power (13-30 Hz)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Beta Power (µV²)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels and workload index
        for bar, value in zip(bars2, ch2_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Add workload index
        workload_ch2 = ch2_arith / ch2_rest
        ax2.text(0.5, max(ch2_values) * 0.8, f'Workload Index: {workload_ch2:.2f}x',
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        plt.tight_layout()
        plt.suptitle('Mental Workload Assessment: Rest vs Arithmetic Task',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("Neocore EEG Mental Workload Assessment")
    print("Arithmetic Task vs Rest Comparison")
    print("=" * 60)
    print(f"Number of arithmetic problems: {NUM_EQUATIONS}")
    print(f"Rest periods: {REST_TIME_SEC} seconds each")
    print(f"Beta band frequency: {BETA_BAND[0]}-{BETA_BAND[1]} Hz")
    print("=" * 60)

    target_mac = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if ':' in arg:
                target_mac = arg.upper()
                break

    try:
        device_address = await find_device(target_mac)
        experiment = MentalWorkloadExperiment()
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