#!/usr/bin/env python3
"""
Neocore EEG Attention Assessment - Enhanced Visual Oddball Paradigm
Complete implementation with artifact rejection and improved analysis
"""

import asyncio
import sys
import struct
import time
import random
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
from scipy import signal, stats
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
# Attention Task Parameters
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RATE = 250
SAMPLES_PER_CHUNK = 27
NUM_CHANNELS = 2
NUM_TRIALS = 20  # Total number of stimuli
TARGET_PROBABILITY = 0.2  # 20% targets, 80% standards
STIMULUS_DURATION = 1.5  # seconds per stimulus
ISI_RANGE = (0.8, 1.2)  # Inter-stimulus interval range (seconds)
REST_TIME_SEC = 10  # 1 minute rest periods

# Frequency bands for attention analysis
THETA_BAND = (4, 8)  # Attention and working memory
ALPHA_BAND = (8, 12)  # Alertness (decreases with attention)
BETA_BAND = (13, 30)  # Focused attention


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Signal Processing with Artifact Rejection
# ═══════════════════════════════════════════════════════════════════════════════

class ImprovedEEGProcessor:
    """Enhanced EEG processing with artifact rejection and validation."""

    def __init__(self, sample_rate=250):
        self.fs = sample_rate
        self.setup_filters()

    def setup_filters(self):
        """Setup filtering parameters."""
        nyq = self.fs / 2

        # More aggressive artifact removal
        # Bandpass: 0.5-45 Hz (removes DC drift and high-freq noise)
        self.bp_sos = signal.butter(6, [0.5 / nyq, 45 / nyq], btype='band', output='sos')

        # Notch filters for line noise - FIXED: convert to SOS format
        notch_60_b, notch_60_a = signal.iirnotch(60, 30, self.fs)
        self.notch_60_sos = signal.tf2sos(notch_60_b, notch_60_a)

        notch_50_b, notch_50_a = signal.iirnotch(50, 25, self.fs)
        self.notch_50_sos = signal.tf2sos(notch_50_b, notch_50_a)

    def preprocess_signal(self, data: np.ndarray) -> np.ndarray:
        """Apply comprehensive preprocessing."""
        if len(data) < self.fs:  # Need at least 1 second of data
            return data

        try:
            # Remove extreme outliers (likely artifacts)
            data = self.remove_extreme_outliers(data)

            # Apply bandpass filter
            filtered = signal.sosfilt(self.bp_sos, data)

            # Apply notch filters
            filtered = signal.sosfilt(self.notch_60_sos, filtered)
            filtered = signal.sosfilt(self.notch_50_sos, filtered)

            # Remove remaining artifacts
            filtered = self.artifact_rejection(filtered)

            return filtered

        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Return basic filtered data if advanced processing fails
            try:
                return signal.sosfilt(self.bp_sos, data)
            except:
                return data

    def remove_extreme_outliers(self, data: np.ndarray, threshold_std=5) -> np.ndarray:
        """Remove extreme outliers that are likely artifacts."""
        mean_val = np.mean(data)
        std_val = np.std(data)

        # Clip values beyond threshold_std standard deviations
        lower_bound = mean_val - threshold_std * std_val
        upper_bound = mean_val + threshold_std * std_val

        return np.clip(data, lower_bound, upper_bound)

    def artifact_rejection(self, data: np.ndarray, window_size=1.0) -> np.ndarray:
        """Reject segments with high amplitude or gradient (artifacts)."""
        if len(data) < self.fs:
            return data

        window_samples = int(window_size * self.fs)
        cleaned_data = data.copy()

        # Sliding window artifact detection
        for i in range(0, len(data) - window_samples, window_samples // 2):
            segment = data[i:i + window_samples]

            # Check for high amplitude artifacts
            if np.max(np.abs(segment)) > 200:  # Adjust threshold as needed
                # Replace with interpolated values
                if i > 0 and i + window_samples < len(data):
                    start_val = data[i - 1]
                    end_val = data[i + window_samples]
                    cleaned_data[i:i + window_samples] = np.linspace(start_val, end_val, window_samples)

            # Check for high gradient (muscle artifacts)
            gradient = np.abs(np.diff(segment))
            if np.mean(gradient) > 50:  # Adjust threshold as needed
                if i > 0 and i + window_samples < len(data):
                    start_val = data[i - 1]
                    end_val = data[i + window_samples]
                    cleaned_data[i:i + window_samples] = np.linspace(start_val, end_val, window_samples)

        return cleaned_data

    def calculate_band_power_robust(self, data: np.ndarray, freq_band: Tuple[float, float]) -> float:
        """Calculate band power with robust spectral estimation."""
        if len(data) < 2 * self.fs:  # Need at least 2 seconds
            return 0.0

        try:
            # Use Welch's method with more conservative parameters
            nperseg = min(self.fs * 2, len(data) // 4)  # 2-second windows or 1/4 of data
            noverlap = nperseg // 2

            freqs, psd = signal.welch(
                data,
                self.fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann',
                detrend='constant'
            )

            # Find frequency band
            band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])

            if not np.any(band_mask):
                return 0.0

            # Calculate power (integrate PSD)
            freq_res = freqs[1] - freqs[0]
            band_power = np.sum(psd[band_mask]) * freq_res

            return band_power

        except Exception as e:
            print(f"Error calculating band power: {e}")
            return 0.0

    def validate_data_quality(self, data: np.ndarray) -> dict:
        """Assess data quality metrics."""
        if len(data) == 0:
            return {"quality": "poor", "reasons": ["No data"]}

        reasons = []

        # Check signal amplitude
        amplitude = np.max(data) - np.min(data)
        if amplitude < 10:
            reasons.append("Very low amplitude")
        elif amplitude > 1000:
            reasons.append("Very high amplitude - likely artifacts")

        # Check for flat segments
        diff_data = np.diff(data)
        flat_ratio = np.sum(np.abs(diff_data) < 1e-6) / len(diff_data)
        if flat_ratio > 0.1:
            reasons.append("Too many flat segments")

        # Check for excessive high-frequency noise
        high_freq_power = np.sum(np.abs(diff_data) > 50)
        if high_freq_power > len(data) * 0.05:
            reasons.append("Excessive high-frequency noise")

        # Overall quality assessment
        if len(reasons) == 0:
            quality = "good"
        elif len(reasons) <= 2:
            quality = "fair"
        else:
            quality = "poor"

        return {"quality": quality, "reasons": reasons, "amplitude": amplitude}

# ═══════════════════════════════════════════════════════════════════════════════
# BLE Protocol Functions
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
# Enhanced Signal Processing Functions
# ═══════════════════════════════════════════════════════════════════════════════

def process_attention_data_improved(ch1_data: np.ndarray, ch2_data: np.ndarray) -> dict:
    """Enhanced attention data processing with artifact rejection."""
    processor = ImprovedEEGProcessor(250)

    # Validate input data
    if len(ch1_data) < 500 or len(ch2_data) < 500:  # Need at least 2 seconds
        print("Warning: Insufficient data for reliable analysis")
        return {}

    print(f"Processing {len(ch1_data)} samples ({len(ch1_data) / 250:.1f} seconds)")

    # Preprocess signals
    print("Applying artifact rejection and filtering...")
    ch1_clean = processor.preprocess_signal(ch1_data)
    ch2_clean = processor.preprocess_signal(ch2_data)

    # Validate data quality
    ch1_quality = processor.validate_data_quality(ch1_clean)
    ch2_quality = processor.validate_data_quality(ch2_clean)

    print(f"Channel 1 quality: {ch1_quality['quality']} - {ch1_quality.get('reasons', [])}")
    print(f"Channel 2 quality: {ch2_quality['quality']} - {ch2_quality.get('reasons', [])}")

    # Calculate frequency band powers
    freq_bands = {
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (13, 30)
    }

    results = {
        'ch1_quality': ch1_quality,
        'ch2_quality': ch2_quality,
        'data_length': len(ch1_data) / 250  # in seconds
    }

    for band_name, freq_range in freq_bands.items():
        ch1_power = processor.calculate_band_power_robust(ch1_clean, freq_range)
        ch2_power = processor.calculate_band_power_robust(ch2_clean, freq_range)

        results[f'ch1_{band_name}'] = ch1_power
        results[f'ch2_{band_name}'] = ch2_power

        print(
            f"{band_name.capitalize()} ({freq_range[0]}-{freq_range[1]} Hz): CH1={ch1_power:.3f}, CH2={ch2_power:.3f}")

    return results


def calculate_attention_indices(rest_data: dict, task_data: dict) -> dict:
    """Calculate attention indices with validation."""
    indices = {}
    warnings = []

    bands = ['theta', 'alpha', 'beta']
    channels = ['ch1', 'ch2']

    for band in bands:
        for ch in channels:
            rest_key = f'{ch}_{band}'

            if rest_key not in rest_data or rest_key not in task_data:
                continue

            rest_power = rest_data[rest_key]
            task_power = task_data[rest_key]

            # Validate powers
            if rest_power <= 0 or task_power <= 0:
                warnings.append(f"{ch} {band}: Invalid power values")
                continue

            ratio = task_power / rest_power

            # Check for unrealistic ratios
            if ratio > 10:
                warnings.append(f"{ch} {band}: Very high ratio ({ratio:.1f}) - possible artifacts")
            elif ratio < 0.1:
                warnings.append(f"{ch} {band}: Very low ratio ({ratio:.1f}) - possible artifacts")

            indices[f'{ch}_{band}_ratio'] = ratio

    if warnings:
        print("Data quality warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return indices


def plot_attention_analysis_improved(rest_results, task_results, indices):
    """Create improved attention analysis plots with data quality indicators."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    bands = ['Theta\n(4-8 Hz)', 'Alpha\n(8-12 Hz)', 'Beta\n(13-30 Hz)']
    band_keys = ['theta', 'alpha', 'beta']

    # Function to get values safely
    def get_values(channel, results):
        return [results.get(f'{channel}_{band}', 0) for band in band_keys]

    # Channel 1 power comparison
    ch1_rest = get_values('ch1', rest_results)
    ch1_task = get_values('ch1', task_results)

    x = np.arange(len(bands))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, ch1_rest, width, label='Rest', color='lightblue', alpha=0.7)
    bars2 = ax1.bar(x + width / 2, ch1_task, width, label='Attention Task', color='orange', alpha=0.7)

    ax1.set_title('Channel 1 - Frequency Band Power', fontweight='bold')
    ax1.set_ylabel('Power (µV²)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bands)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization

    # Add quality indicator
    quality1 = rest_results.get('ch1_quality', {}).get('quality', 'unknown')
    ax1.text(0.02, 0.98, f'Data Quality: {quality1}', transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Channel 2 power comparison
    ch2_rest = get_values('ch2', rest_results)
    ch2_task = get_values('ch2', task_results)

    bars3 = ax2.bar(x - width / 2, ch2_rest, width, label='Rest', color='lightblue', alpha=0.7)
    bars4 = ax2.bar(x + width / 2, ch2_task, width, label='Attention Task', color='orange', alpha=0.7)

    ax2.set_title('Channel 2 - Frequency Band Power', fontweight='bold')
    ax2.set_ylabel('Power (µV²)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bands)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization

    # Add quality indicator
    quality2 = rest_results.get('ch2_quality', {}).get('quality', 'unknown')
    ax2.text(0.02, 0.98, f'Data Quality: {quality2}', transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Channel 1 ratios
    ch1_ratios = [indices.get(f'ch1_{band}_ratio', 1) for band in band_keys]
    colors1 = []
    for i, ratio in enumerate(ch1_ratios):
        if band_keys[i] == 'alpha':
            colors1.append('green' if ratio < 1 else 'red')  # Alpha should decrease
        else:
            colors1.append('green' if ratio > 1 else 'red')  # Theta/Beta should increase

    bars5 = ax3.bar(bands, ch1_ratios, color=colors1, alpha=0.7, edgecolor='black')
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax3.set_title('Channel 1 - Attention Indices (Task/Rest)', fontweight='bold')
    ax3.set_ylabel('Ratio')
    ax3.set_ylim(0, min(5, max(ch1_ratios) * 1.1))  # Cap at reasonable values
    ax3.grid(True, alpha=0.3)

    for bar, ratio in zip(bars5, ch1_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')

    # Channel 2 ratios
    ch2_ratios = [indices.get(f'ch2_{band}_ratio', 1) for band in band_keys]
    colors2 = []
    for i, ratio in enumerate(ch2_ratios):
        if band_keys[i] == 'alpha':
            colors2.append('green' if ratio < 1 else 'red')  # Alpha should decrease
        else:
            colors2.append('green' if ratio > 1 else 'red')  # Theta/Beta should increase

    bars6 = ax4.bar(bands, ch2_ratios, color=colors2, alpha=0.7, edgecolor='black')
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax4.set_title('Channel 2 - Attention Indices (Task/Rest)', fontweight='bold')
    ax4.set_ylabel('Ratio')
    ax4.set_ylim(0, min(5, max(ch2_ratios) * 1.1))  # Cap at reasonable values
    ax4.grid(True, alpha=0.3)

    for bar, ratio in zip(bars6, ch2_ratios):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.suptitle('Enhanced Attention Assessment with Artifact Rejection',
                 fontsize=16, fontweight='bold', y=1.02)

    # Add interpretation guide
    fig.text(0.5, 0.01,
             'Expected: ↑Theta (attention), ↓Alpha (alertness), ↑Beta (focus). Green=Good, Red=Poor/Artifacts',
             ha='center', fontsize=10, style='italic')

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Attention Experiment
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
        """Enhanced attention analysis with better data handling."""
        if self.rest_data is None or self.attention_data is None:
            print("Error: Missing data for analysis")
            return

        print(f"\n{'=' * 70}")
        print("ENHANCED ATTENTION ANALYSIS")
        print(f"{'=' * 70}")

        # Process data with improved methods
        print("\nProcessing REST data:")
        rest_results = process_attention_data_improved(*self.rest_data)

        print("\nProcessing ATTENTION TASK data:")
        task_results = process_attention_data_improved(*self.attention_data)

        if not rest_results or not task_results:
            print("Error: Could not process data")
            return

        # Calculate indices with validation
        print("\nCalculating attention indices...")
        indices = calculate_attention_indices(rest_results, task_results)

        # Display results
        print(f"\n{'=' * 50}")
        print("IMPROVED ATTENTION RESULTS:")
        print(f"{'=' * 50}")

        bands = ['theta', 'alpha', 'beta']
        band_ranges = ['(4-8 Hz)', '(8-12 Hz)', '(13-30 Hz)']

        for i, (band, range_str) in enumerate(zip(bands, band_ranges)):
            print(f"\n{band.upper()} {range_str}:")

            for ch in ['ch1', 'ch2']:
                rest_key = f'{ch}_{band}'
                ratio_key = f'{ch}_{band}_ratio'

                if rest_key in rest_results and rest_key in task_results:
                    rest_val = rest_results[rest_key]
                    task_val = task_results[rest_key]
                    ratio_val = indices.get(ratio_key, 0)

                    print(f"  {ch.upper()}: Rest={rest_val:.3f}, Task={task_val:.3f}, Ratio={ratio_val:.2f}")

                    # Interpretation
                    if band == 'theta' and ratio_val > 1.2:
                        print(f"    ✓ Good attention/working memory engagement")
                    elif band == 'alpha' and ratio_val < 0.8:
                        print(f"    ✓ Good alertness (alpha suppression)")
                    elif band == 'beta' and ratio_val > 1.1:
                        print(f"    ✓ Good focused attention")
                    else:
                        print(f"    ? Ratio within normal range or possible artifacts")

        # Create improved visualization
        if indices:
            plot_attention_analysis_improved(rest_results, task_results, indices)
        else:
            print("Could not create plots due to data quality issues")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("Neocore EEG Enhanced Attention Assessment")
    print("Visual Oddball Paradigm with Artifact Rejection")
    print("=" * 70)
    print(f"Number of trials: {NUM_TRIALS}")
    print(f"Target probability: {TARGET_PROBABILITY * 100:.0f}%")
    print(f"Rest periods: {REST_TIME_SEC} seconds each")
    print("Frequency bands: Theta (4-8), Alpha (8-12), Beta (13-30) Hz")
    print("Enhanced with artifact rejection and data quality validation")
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