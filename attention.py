#!/usr/bin/env python3
"""
Neocore EEG Attention Assessment - Visual Oddball Paradigm
Measures sustained attention and selective attention using P300 ERP and frequency analysis
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
# Attention Task Parameters
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 250
SAMPLES_PER_CHUNK = 27
NUM_CHANNELS = 2
NUM_TRIALS = 100  # Total number of stimuli
TARGET_PROBABILITY = 0.2  # 20% targets, 80% standards
STIMULUS_DURATION = 1.5  # seconds per stimulus
ISI_RANGE = (0.8, 1.2)  # Inter-stimulus interval range (seconds)
REST_TIME_SEC = 60  # 1 minute rest periods

# Frequency bands for attention analysis
THETA_BAND = (4, 8)  # Attention and working memory
ALPHA_BAND = (8, 12)  # Alertness (decreases with attention)
BETA_BAND = (13, 30)  # Focused attention


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
# Attention Task Generator
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionTask:
    """Visual Oddball paradigm for attention assessment."""

    def __init__(self):
        self.trial_sequence = []
        self.target_trials = []
        self.responses = []
        self.response_times = []
        self.hits = 0
        self.false_alarms = 0
        self.misses = 0
        self.correct_rejections = 0

    def generate_trial_sequence(self):
        """Generate randomized sequence of standard and target stimuli."""
        # Calculate number of targets and standards
        num_targets = int(NUM_TRIALS * TARGET_PROBABILITY)
        num_standards = NUM_TRIALS - num_targets

        # Create sequence: True = target, False = standard
        sequence = [True] * num_targets + [False] * num_standards
        random.shuffle(sequence)

        # Ensure no more than 3 consecutive targets
        self._balance_sequence(sequence)

        self.trial_sequence = sequence
        self.target_trials = [i for i, is_target in enumerate(sequence) if is_target]

        print(f"Generated {NUM_TRIALS} trials: {num_targets} targets, {num_standards} standards")

    def _balance_sequence(self, sequence):
        """Ensure no more than 3 consecutive targets."""
        for i in range(len(sequence) - 3):
            if all(sequence[i:i + 4]):  # 4 consecutive targets
                # Find next standard and swap
                for j in range(i + 4, len(sequence)):
                    if not sequence[j]:
                        sequence[i + 3], sequence[j] = sequence[j], sequence[i + 3]
                        break

    def get_stimulus_text(self, trial_num: int) -> str:
        """Get stimulus text for display."""
        if self.trial_sequence[trial_num]:
            return "🎯 O 🎯"  # Target stimulus
        else:
            return "✕ X ✕"  # Standard stimulus

    def is_target(self, trial_num: int) -> bool:
        """Check if trial is a target."""
        return self.trial_sequence[trial_num]

    def record_response(self, trial_num: int, responded: bool, response_time: float):
        """Record response for a trial."""
        is_target = self.trial_sequence[trial_num]

        if is_target and responded:
            self.hits += 1
        elif is_target and not responded:
            self.misses += 1
        elif not is_target and responded:
            self.false_alarms += 1
        else:
            self.correct_rejections += 1

        self.responses.append(responded)
        self.response_times.append(response_time if responded else None)

    def get_performance_stats(self) -> dict:
        """Calculate attention performance metrics."""
        # Calculate hit rate and false alarm rate
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        fa_rate = self.false_alarms / (self.false_alarms + self.correct_rejections) if (
                                                                                                   self.false_alarms + self.correct_rejections) > 0 else 0

        # Calculate d-prime (sensitivity) and criterion
        # Avoid extreme values
        hit_rate = max(0.01, min(0.99, hit_rate))
        fa_rate = max(0.01, min(0.99, fa_rate))

        from scipy.stats import norm
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))

        # Calculate reaction time for hits
        hit_rts = [rt for i, rt in enumerate(self.response_times)
                   if rt is not None and self.trial_sequence[i]]
        avg_rt = np.mean(hit_rts) if hit_rts else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'false_alarms': self.false_alarms,
            'correct_rejections': self.correct_rejections,
            'hit_rate': hit_rate,
            'false_alarm_rate': fa_rate,
            'd_prime': d_prime,
            'criterion': criterion,
            'avg_reaction_time': avg_rt,
            'accuracy': (self.hits + self.correct_rejections) / NUM_TRIALS
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Data Collector for Attention
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionDataCollector:
    """Data collector for attention assessment."""

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
# Signal Processing for Attention
# ═══════════════════════════════════════════════════════════════════════════════

def filter_signal(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply bandpass filter to remove artifacts."""
    nyq = sample_rate / 2
    # Bandpass filter: 1-40 Hz
    sos = signal.butter(4, [1 / nyq, 40 / nyq], btype='band', output='sos')
    return signal.sosfilt(sos, data)


def calculate_band_power(data: np.ndarray, sample_rate: int, freq_band: Tuple[float, float]) -> float:
    """Calculate power in specific frequency band."""
    freqs, psd = signal.welch(data, sample_rate, nperseg=sample_rate * 2, noverlap=sample_rate)

    # Find frequency band indices
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])

    # Calculate mean power in band
    band_power = np.mean(psd[band_mask])

    return band_power


def process_attention_data(ch1_data: np.ndarray, ch2_data: np.ndarray) -> dict:
    """Process data and return attention-related frequency bands."""
    # Filter the data
    ch1_filtered = filter_signal(ch1_data, SAMPLE_RATE)
    ch2_filtered = filter_signal(ch2_data, SAMPLE_RATE)

    results = {}

    # Calculate power in attention-related frequency bands
    for band_name, freq_band in [('theta', THETA_BAND), ('alpha', ALPHA_BAND), ('beta', BETA_BAND)]:
        ch1_power = calculate_band_power(ch1_filtered, SAMPLE_RATE, freq_band)
        ch2_power = calculate_band_power(ch2_filtered, SAMPLE_RATE, freq_band)
        results[f'ch1_{band_name}'] = ch1_power
        results[f'ch2_{band_name}'] = ch2_power

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Attention Experiment
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionExperiment:
    def __init__(self):
        self.collector = AttentionDataCollector()
        self.task = AttentionTask()
        self.rest_data = None
        self.attention_data = None

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
        """Run the complete attention experiment."""
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
                # Generate task sequence
                self.task.generate_trial_sequence()

                # Phase 1: Baseline rest
                await self.baseline_rest()

                # Phase 2: Attention task
                await self.attention_task()

                # Phase 3: Recovery rest
                await self.recovery_rest()

                # Analyze results
                self.analyze_attention()

            finally:
                # Stop streaming
                stop_cmd = build_stream_command(False)
                await client.write_gatt_char(RX_UUID, stop_cmd, response=False)
                await client.stop_notify(TX_UUID)

    async def baseline_rest(self):
        """Record baseline rest period."""
        print(f"\n{'=' * 70}")
        print("PHASE 1: BASELINE REST")
        print(f"{'=' * 70}")
        print("Please sit comfortably and relax with eyes open.")
        print("Look at a fixed point and try to minimize eye movements.")
        print(f"Recording will start in 5 seconds for {REST_TIME_SEC} seconds...")

        for i in range(5, 0, -1):
            print(f"{i}...")
            await asyncio.sleep(1)

        print(f"\n🔴 RECORDING BASELINE - Please stay relaxed and alert")
        print("=" * 70)

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

    async def attention_task(self):
        """Run visual oddball attention task."""
        print(f"\n{'=' * 70}")
        print("PHASE 2: VISUAL ODDBALL ATTENTION TASK")
        print(f"{'=' * 70}")
        print("You will see X's and O's on the screen.")
        print("INSTRUCTIONS:")
        print("• Press SPACE when you see the TARGET: 🎯 O 🎯")
        print("• DO NOT press anything for STANDARD: ✕ X ✕")
        print("• Respond as quickly and accurately as possible")
        print("• Keep your eyes focused on the center")
        print(f"\nTotal trials: {NUM_TRIALS}")
        print(f"Target probability: {TARGET_PROBABILITY * 100:.0f}%")
        print("\nStarting in 5 seconds...")

        for i in range(5, 0, -1):
            print(f"{i}...")
            await asyncio.sleep(1)

        print(f"\n🔴 STARTING ATTENTION TASK")
        print("=" * 70)

        self.collector.start_recording("ATTENTION")

        # Run oddball trials
        for trial in range(NUM_TRIALS):
            stimulus = self.task.get_stimulus_text(trial)
            is_target = self.task.is_target(trial)

            # Display stimulus
            print(f"\nTrial {trial + 1}/{NUM_TRIALS}")
            print(f"\n      {stimulus}")
            print(f"      {'TARGET!' if is_target else 'Standard'}")

            # Record response (in real implementation, use keyboard input)
            start_time = time.time()
            await asyncio.sleep(STIMULUS_DURATION)

            # Simulate response based on target detection (for demo)
            if is_target:
                # Simulate 85% hit rate with some reaction time
                responded = random.random() < 0.85
                rt = random.uniform(0.3, 0.8) if responded else 0
            else:
                # Simulate 10% false alarm rate
                responded = random.random() < 0.10
                rt = random.uniform(0.4, 1.0) if responded else 0

            self.task.record_response(trial, responded, rt)

            if responded:
                print(f"      Response: SPACE (RT: {rt:.2f}s)")
            else:
                print(f"      Response: None")

            # Inter-stimulus interval
            isi = random.uniform(*ISI_RANGE)
            await asyncio.sleep(isi)

            # Progress update
            if (trial + 1) % 20 == 0:
                progress = (trial + 1) / NUM_TRIALS * 100
                print(f"\n--- Progress: {progress:.0f}% complete ---")

        self.collector.stop_recording()
        attention_data = self.collector.get_data()
        self.attention_data = attention_data

        print(f"\n✅ Attention task complete! Collected {len(attention_data[0])} samples")

        # Show performance stats
        stats = self.task.get_performance_stats()
        print(f"\nPERFORMANCE STATISTICS:")
        print(f"Accuracy: {stats['accuracy']:.1%}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"False Alarm Rate: {stats['false_alarm_rate']:.1%}")
        print(f"d' (Sensitivity): {stats['d_prime']:.2f}")
        print(f"Average Reaction Time: {stats['avg_reaction_time']:.2f}s")

    async def recovery_rest(self):
        """Record recovery rest period."""
        print(f"\n{'=' * 70}")
        print("PHASE 3: RECOVERY REST")
        print(f"{'=' * 70}")
        print("Task complete! Please relax again.")
        print(f"Final recording for {REST_TIME_SEC} seconds...")

        for i in range(5, 0, -1):
            print(f"{i}...")
            await asyncio.sleep(1)

        print(f"\n🔴 RECORDING RECOVERY - Please relax")
        print("=" * 70)

        self.collector.start_recording("RECOVERY")

        start_time = time.time()
        while time.time() - start_time < REST_TIME_SEC:
            remaining = REST_TIME_SEC - (time.time() - start_time)
            print(f"\rRecovery recording: {remaining:.1f}s remaining", end="", flush=True)
            await asyncio.sleep(0.1)

        self.collector.stop_recording()
        print(f"\n✅ Recovery recording complete!")

    def analyze_attention(self):
        """Analyze attention using frequency band analysis."""
        if self.rest_data is None or self.attention_data is None:
            print("Error: Missing data for analysis")
            return

        print(f"\n{'=' * 70}")
        print("ANALYZING ATTENTION MARKERS")
        print(f"{'=' * 70}")

        # Process data
        rest_bands = process_attention_data(*self.rest_data)
        attention_bands = process_attention_data(*self.attention_data)

        # Calculate attention indices
        results = {}
        for band in ['theta', 'alpha', 'beta']:
            for ch in ['ch1', 'ch2']:
                key = f'{ch}_{band}'
                rest_power = rest_bands[key]
                attention_power = attention_bands[key]
                ratio = attention_power / rest_power
                results[f'{key}_ratio'] = ratio

        # Print results
        print(f"\nATTENTION ANALYSIS RESULTS:")
        print(f"\nChannel 1:")
        print(
            f"  Theta (4-8 Hz) - Rest: {rest_bands['ch1_theta']:.2f}, Task: {attention_bands['ch1_theta']:.2f}, Ratio: {results['ch1_theta_ratio']:.2f}")
        print(
            f"  Alpha (8-12 Hz) - Rest: {rest_bands['ch1_alpha']:.2f}, Task: {attention_bands['ch1_alpha']:.2f}, Ratio: {results['ch1_alpha_ratio']:.2f}")
        print(
            f"  Beta (13-30 Hz) - Rest: {rest_bands['ch1_beta']:.2f}, Task: {attention_bands['ch1_beta']:.2f}, Ratio: {results['ch1_beta_ratio']:.2f}")

        print(f"\nChannel 2:")
        print(
            f"  Theta (4-8 Hz) - Rest: {rest_bands['ch2_theta']:.2f}, Task: {attention_bands['ch2_theta']:.2f}, Ratio: {results['ch2_theta_ratio']:.2f}")
        print(
            f"  Alpha (8-12 Hz) - Rest: {rest_bands['ch2_alpha']:.2f}, Task: {attention_bands['ch2_alpha']:.2f}, Ratio: {results['ch2_alpha_ratio']:.2f}")
        print(
            f"  Beta (13-30 Hz) - Rest: {rest_bands['ch2_beta']:.2f}, Task: {attention_bands['ch2_beta']:.2f}, Ratio: {results['ch2_beta_ratio']:.2f}")

        # Create attention analysis plot
        self.plot_attention_analysis(rest_bands, attention_bands, results)

    def plot_attention_analysis(self, rest_bands, attention_bands, ratios):
        """Create comprehensive attention analysis plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Channel 1 frequency bands
        bands = ['Theta\n(4-8 Hz)', 'Alpha\n(8-12 Hz)', 'Beta\n(13-30 Hz)']
        ch1_rest = [rest_bands['ch1_theta'], rest_bands['ch1_alpha'], rest_bands['ch1_beta']]
        ch1_task = [attention_bands['ch1_theta'], attention_bands['ch1_alpha'], attention_bands['ch1_beta']]

        x = np.arange(len(bands))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, ch1_rest, width, label='Rest', color='lightblue', alpha=0.7)
        bars2 = ax1.bar(x + width / 2, ch1_task, width, label='Attention Task', color='orange', alpha=0.7)

        ax1.set_title('Channel 1 - Attention-Related Frequency Bands', fontweight='bold')
        ax1.set_ylabel('Power (µV²)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(bands)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # Channel 2 frequency bands
        ch2_rest = [rest_bands['ch2_theta'], rest_bands['ch2_alpha'], rest_bands['ch2_beta']]
        ch2_task = [attention_bands['ch2_theta'], attention_bands['ch2_alpha'], attention_bands['ch2_beta']]

        bars3 = ax2.bar(x - width / 2, ch2_rest, width, label='Rest', color='lightblue', alpha=0.7)
        bars4 = ax2.bar(x + width / 2, ch2_task, width, label='Attention Task', color='orange', alpha=0.7)

        ax2.set_title('Channel 2 - Attention-Related Frequency Bands', fontweight='bold')
        ax2.set_ylabel('Power (µV²)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(bands)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        # Attention ratios Channel 1
        ch1_ratios = [ratios['ch1_theta_ratio'], ratios['ch1_alpha_ratio'], ratios['ch1_beta_ratio']]
        colors = ['green' if r > 1 else 'red' for r in ch1_ratios]

        bars5 = ax3.bar(bands, ch1_ratios, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Channel 1 - Attention Indices (Task/Rest)', fontweight='bold')
        ax3.set_ylabel('Ratio')
        ax3.grid(True, alpha=0.3)

        # Add value labels and interpretation
        for bar, ratio in zip(bars5, ch1_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')

        # Attention ratios Channel 2
        ch2_ratios = [ratios['ch2_theta_ratio'], ratios['ch2_alpha_ratio'], ratios['ch2_beta_ratio']]
        colors = ['green' if r > 1 else 'red' for r in ch2_ratios]

        bars6 = ax4.bar(bands, ch2_ratios, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Channel 2 - Attention Indices (Task/Rest)', fontweight='bold')
        ax4.set_ylabel('Ratio')
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for bar, ratio in zip(bars6, ch2_ratios):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.suptitle('Attention Assessment: Visual Oddball Paradigm Results',
                     fontsize=16, fontweight='bold', y=1.02)

        # Add interpretation text
        fig.text(0.5, 0.01,
                 'Expected: ↑Theta (attention/memory), ↓Alpha (reduced relaxation), ↑Beta (focused attention)',
                 ha='center', fontsize=10, style='italic')

        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("Neocore EEG Attention Assessment")
    print("Visual Oddball Paradigm")
    print("=" * 70)
    print(f"Number of trials: {NUM_TRIALS}")
    print(f"Target probability: {TARGET_PROBABILITY * 100:.0f}%")
    print(f"Rest periods: {REST_TIME_SEC} seconds each")
    print("Frequency bands: Theta (4-8), Alpha (8-12), Beta (13-30) Hz")
    print("=" * 70)

    target_mac = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if ':' in arg:
                target_mac = arg.upper()
                break

    try:
        device_address = await find_device(target_mac)
        experiment = AttentionExperiment()
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