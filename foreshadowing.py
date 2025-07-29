#!/usr/bin/env python3
"""
YouTube Short Video Foreshadowing Player with EEG Engagement Monitoring
Uses Neocore EEG to detect engagement drops and trigger foreshadowing effects
Fixed video display with proper VLC integration
"""
try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available - install with: pip install Pillow")
import asyncio
import sys
import struct
import time
import json
import os
import threading
import queue
import subprocess
import webbrowser
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import random
import tempfile
import platform

import numpy as np
from scipy import signal
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import requests
from datetime import datetime

# Video processing
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available - some features will be limited")

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available - some features will be limited")

# YouTube processing
try:
    import yt_dlp

    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    print("yt-dlp not available - install with: pip install yt-dlp")

# VLC (optional)
VLC_AVAILABLE = False
try:
    import vlc

    VLC_AVAILABLE = True
    print("VLC is available for video playback")
except ImportError:
    print("VLC not available - using alternative video backends")

# EEG device integration
from bleak import BleakScanner, BleakClient

# ═══════════════════════════════════════════════════════════════════════════════
# EEG Configuration
# ═══════════════════════════════════════════════════════════════════════════════

RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
TARGET_NAMES = {"QCC5181", "QCC5181-LE", "NEOCORE"}

FEATURE_SENSOR_CFG = 0x01
CMD_STREAM_CTRL = 0x00
PDU_TYPE_COMMAND = 0x00

# Signal Parameters
SAMPLE_RATE = 250
SAMPLES_PER_CHUNK = 27
NUM_CHANNELS = 2
ENGAGEMENT_WINDOW = 2  # seconds for engagement calculation

# Engagement thresholds
BASELINE_DURATION = 5  # seconds to establish baseline
DROPOUT_THRESHOLD = 0.9  # 30% below baseline triggers foreshadowing
FORESHADOW_COOLDOWN = 15  # seconds between foreshadowing events


@dataclass
class EngagementState:
    """Current engagement state"""
    current_score: float
    baseline_score: float
    is_engaged: bool
    last_foreshadow_time: float
    foreshadow_count: int


@dataclass
class VideoInfo:
    """Video information"""
    title: str
    duration: int
    url: str
    thumbnail: str
    description: str
    local_path: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# EEG Processing (same as before)
# ═══════════════════════════════════════════════════════════════════════════════

def build_command(feature_id: int, pdu_id: int, payload: bytes = b"") -> bytes:
    command_id = (feature_id << 9) | (PDU_TYPE_COMMAND << 7) | pdu_id
    return command_id.to_bytes(2, 'big') + payload


def build_stream_command(start: bool) -> bytes:
    payload = b"\x01" if start else b"\x00"
    return build_command(FEATURE_SENSOR_CFG, CMD_STREAM_CTRL, payload)


async def find_device(target_mac: Optional[str] = None) -> str:
    if target_mac:
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


class EngagementMonitor:
    """Real-time engagement monitoring from EEG"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.fs = sample_rate
        self.window_size = sample_rate * ENGAGEMENT_WINDOW

        # Data buffers
        self.ch1_buffer = deque(maxlen=self.window_size)
        self.ch2_buffer = deque(maxlen=self.window_size)

        # Timing and phases
        self.data_start_time = None
        self.discard_duration = 5.0  # Discard first 5 seconds
        self.baseline_duration = BASELINE_DURATION  # Then collect baseline for 30 seconds

        # Engagement tracking phases
        self.phase = "discarding"  # "discarding" -> "collecting_baseline" -> "monitoring"
        self.baseline_scores = []
        self.baseline_established = False
        self.baseline_mean = 0.0

        # State
        self.current_state = EngagementState(0.0, 0.0, True, 0.0, 0)

    def add_data(self, ch1_data: np.ndarray, ch2_data: np.ndarray):
        """Add new EEG data"""
        # Mark start time on first data
        if self.data_start_time is None:
            self.data_start_time = time.time()
            print("EEG data started - discarding first 5 seconds for sensor settling")

        self.ch1_buffer.extend(ch1_data)
        self.ch2_buffer.extend(ch2_data)

    def calculate_engagement(self) -> float:
        """Calculate current engagement score"""
        if len(self.ch1_buffer) < self.window_size:
            return 0.0

        # Get current window
        ch1_data = np.array(list(self.ch1_buffer))
        ch2_data = np.array(list(self.ch2_buffer))

        # Average channels
        avg_data = (ch1_data + ch2_data) / 2

        # Calculate power spectral density
        freqs, psd = signal.welch(avg_data, self.fs, nperseg=self.fs)

        # Define frequency bands
        def get_band_power(low, high):
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            return np.mean(psd[idx]) if len(idx) > 0 else 0.0

        alpha_power = get_band_power(8, 12)
        beta_power = get_band_power(13, 30)
        theta_power = get_band_power(4, 8)

        # Engagement metric: beta / (alpha + theta)
        engagement_score = beta_power / (alpha_power + theta_power + 1e-10)

        return engagement_score

    def update_engagement_state(self) -> EngagementState:
        """Update and return current engagement state"""
        if self.data_start_time is None:
            return self.current_state

        # Calculate time since data started
        elapsed_time = time.time() - self.data_start_time
        current_score = self.calculate_engagement()
        self.current_state.current_score = current_score

        # Phase 1: Discard initial noisy data
        if self.phase == "discarding":
            if elapsed_time < self.discard_duration:
                # Still discarding
                remaining = self.discard_duration - elapsed_time
                # Don't print too frequently
                if int(remaining) != int(remaining + 0.1):  # Print roughly every 0.1 seconds
                    print(f"Discarding noisy EEG data: {remaining:.1f}s remaining")
                return self.current_state
            else:
                # Move to baseline collection phase
                self.phase = "collecting_baseline"
                self.baseline_start_time = time.time()
                print("✅ EEG signal stabilized - starting baseline collection")

        # Phase 2: Collect baseline data
        elif self.phase == "collecting_baseline":
            baseline_elapsed = time.time() - self.baseline_start_time

            if baseline_elapsed < self.baseline_duration:
                # Still collecting baseline
                self.baseline_scores.append(current_score)
                remaining = self.baseline_duration - baseline_elapsed

                # Update progress every few seconds
                if len(self.baseline_scores) % (self.fs // SAMPLES_PER_CHUNK) == 0:  # Every ~1 second
                    progress = (baseline_elapsed / self.baseline_duration) * 100
                    print(f"Collecting baseline: {progress:.0f}% complete ({remaining:.0f}s remaining)")

                return self.current_state
            else:
                # Baseline collection complete
                if self.baseline_scores:
                    self.baseline_mean = np.mean(self.baseline_scores)
                    self.current_state.baseline_score = self.baseline_mean
                    self.baseline_established = True
                    self.phase = "monitoring"
                    print(f"🎯 Baseline established: {self.baseline_mean:.3f}")
                    print("🔍 Now monitoring engagement for foreshadowing triggers")
                else:
                    print("⚠️ No baseline data collected - continuing without baseline")
                    self.phase = "monitoring"

        # Phase 3: Active monitoring
        elif self.phase == "monitoring":
            # Check engagement level against baseline
            if self.baseline_established:
                threshold = self.baseline_mean * DROPOUT_THRESHOLD
                self.current_state.is_engaged = current_score >= threshold

                # Debug output occasionally
                if hasattr(self, '_last_debug_time'):
                    if time.time() - self._last_debug_time > 5:  # Every 5 seconds
                        status = "✅ Engaged" if self.current_state.is_engaged else "⚠️ Low engagement"
                        print(f"Engagement: {current_score:.3f} (threshold: {threshold:.3f}) - {status}")
                        self._last_debug_time = time.time()
                else:
                    self._last_debug_time = time.time()

        return self.current_state

    def get_phase_info(self) -> dict:
        """Get information about current monitoring phase"""
        if self.data_start_time is None:
            return {"phase": "waiting", "message": "Waiting for EEG data"}

        elapsed = time.time() - self.data_start_time

        if self.phase == "discarding":
            remaining = self.discard_duration - elapsed
            return {
                "phase": "discarding",
                "message": f"Stabilizing EEG signal ({remaining:.1f}s remaining)",
                "progress": (elapsed / self.discard_duration) * 100
            }
        elif self.phase == "collecting_baseline":
            baseline_elapsed = time.time() - self.baseline_start_time
            remaining = self.baseline_duration - baseline_elapsed
            return {
                "phase": "collecting_baseline",
                "message": f"Collecting baseline ({remaining:.0f}s remaining)",
                "progress": (baseline_elapsed / self.baseline_duration) * 100
            }
        else:
            return {
                "phase": "monitoring",
                "message": "Monitoring engagement",
                "baseline": self.baseline_mean if self.baseline_established else None
            }

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Backend Video Handler
# ═══════════════════════════════════════════════════════════════════════════════

class MultiBackendVideoProcessor:
    """Handle YouTube video with multiple backend support"""

    def __init__(self):
        self.video_info = None
        self.temp_dir = tempfile.mkdtemp()

    def get_video_info(self, url: str) -> VideoInfo:
        """Get video information"""
        if not YTDLP_AVAILABLE:
            # Fallback: open in browser
            return VideoInfo(
                title="Browser Playback",
                duration=60,  # Assume 60 seconds
                url=url,
                thumbnail="",
                description="Playing in browser - EEG monitoring only"
            )

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            self.video_info = VideoInfo(
                title=info.get('title', 'Unknown'),
                duration=info.get('duration', 0),
                url=url,
                thumbnail=info.get('thumbnail', ''),
                description=info.get('description', '')
            )

        return self.video_info

    def download_video(self, url: str) -> str:
        """Download video for local processing"""
        if not YTDLP_AVAILABLE:
            raise RuntimeError("yt-dlp not available for video download")

        ydl_opts = {
            'format': 'best[height<=720]',  # Good quality for display
            'outtmpl': os.path.join(self.temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)

        self.video_info.local_path = video_path
        return video_path

    def extract_timestamps(self, video_path: str) -> List[float]:
        """Extract interesting timestamps for foreshadowing"""
        try:
            if OPENCV_AVAILABLE and os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                cap.release()
            else:
                duration = self.video_info.duration if self.video_info else 60

            # Simple approach: divide into segments
            num_segments = 4
            timestamps = []
            for i in range(1, num_segments + 1):
                timestamp = (i * duration) / (num_segments + 1)
                timestamps.append(timestamp)

            return timestamps

        except Exception as e:
            print(f"Video analysis error: {e}")
            # Fallback timestamps
            return [15, 30, 45, 60]


# ═══════════════════════════════════════════════════════════════════════════════
# Proper VLC Video Player Integration
# ═══════════════════════════════════════════════════════════════════════════════

class OpenCVVideoPlayer:
    """OpenCV-based video player with video foreshadowing and synchronized audio"""

    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.cap = None
        self.is_playing = False
        self.is_paused = False
        self.video_path = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.frame_delay = 33  # milliseconds

        # Timing synchronization
        self.video_start_time = 0.0
        self.last_frame_time = 0.0
        self.target_frame_time = 0.0

        # Audio player
        self.audio_player = AudioPlayer()

        # Foreshadowing state
        self.is_foreshadowing = False
        self.saved_position = 0
        self.foreshadow_start_time = 0
        self.foreshadow_duration = 2.0  # seconds
        self.foreshadow_frames_played = 0

        self.setup_video_display()

    def setup_video_display(self):
        """Setup video display canvas"""
        self.video_frame = tk.Frame(self.parent_frame, bg='black')
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for video display
        self.canvas = tk.Canvas(self.video_frame, bg='black', width=800, height=450)
        self.canvas.pack(expand=True)

        # Progress bar
        progress_frame = tk.Frame(self.video_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=2)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

        # Time display
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        time_label = ttk.Label(progress_frame, textvariable=self.time_var)
        time_label.pack()

    def load_video(self, path: str) -> bool:
        """Load video file"""
        if not os.path.exists(path):
            print(f"Video file not found: {path}")
            return False

        try:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                print("Failed to open video")
                return False

            self.video_path = path
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_delay = int(1000 / self.fps)
            self.current_frame = 0

            print(f"Video loaded: {path}")
            print(f"FPS: {self.fps}, Total frames: {self.total_frames}")

            # Extract and load audio
            audio_path = self.audio_player.extract_audio(path)
            if audio_path:
                self.audio_player.load_audio(audio_path)

            # Show first frame
            self.show_current_frame()
            return True

        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def play(self):
        """Start video playback with synchronized audio"""
        if self.cap:
            self.is_playing = True
            self.is_paused = False

            # Set timing references
            self.video_start_time = time.time()
            self.last_frame_time = self.video_start_time

            # Start audio from current position
            current_time_seconds = self.current_frame / self.fps
            self.audio_player.start_playback(current_time_seconds)

            # Start video playback
            self.play_next_frame()
            print("Video playback started with audio sync")

    def pause(self):
        """Pause video playback"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.audio_player.pause()
            print("Video paused")
        else:
            # Resume with current timing
            self.video_start_time = time.time() - (self.current_frame / self.fps)
            self.audio_player.resume()
            print("Video resumed")
            self.play_next_frame()

    def stop(self):
        """Stop video playback"""
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.audio_player.stop()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.show_current_frame()
        print("Video stopped")

    def start_video_foreshadowing(self, duration_ms: int = 2000):
        """Jump ahead and show preview clip, then return"""
        if not self.cap or self.is_foreshadowing:
            return False

        try:
            # Save current position
            self.saved_position = self.current_frame
            current_time_seconds = self.current_frame / self.fps
            total_duration = self.total_frames / self.fps

            # Don't do foreshadowing if already in the last quarter
            if current_time_seconds >= total_duration * 0.75:
                print("Already in last quarter - no foreshadowing needed")
                return False

            # Calculate dynamic preview zone based on video length
            if total_duration <= 120:  # Videos 2 minutes or less
                preview_fraction = 0.25  # Last 1/4 (like 15s of 60s)
            elif total_duration <= 300:  # Videos 2-5 minutes
                preview_fraction = 0.33  # Last 1/3
            elif total_duration <= 600:  # Videos 5-10 minutes
                preview_fraction = 0.4  # Last 2/5
            else:  # Videos longer than 10 minutes
                preview_fraction = 0.5  # Last 1/2

            # Calculate preview zone
            preview_start_time = total_duration * (1 - preview_fraction)
            preview_zone_duration = total_duration - preview_start_time

            # Check if we have enough content for preview
            if preview_zone_duration < 10:  # Need at least 10 seconds of content
                print("Not enough content in preview zone for foreshadowing")
                return False

            # Get or create list of used positions for this video
            if not hasattr(self, 'used_foreshadow_positions'):
                self.used_foreshadow_positions = set()

            # Find unused position in preview zone
            max_attempts = 10
            attempts = 0

            while attempts < max_attempts:
                # Random position in preview zone (but not too close to the end)
                safe_zone_end = total_duration - 5  # Leave 5 seconds at the end
                random_time = preview_start_time + random.uniform(0, safe_zone_end - preview_start_time)

                # Convert to frame number
                foreshadow_position = int(random_time * self.fps)

                # Check if this position was used recently (within 30 seconds)
                position_used = False
                for used_pos in self.used_foreshadow_positions:
                    if abs(foreshadow_position - used_pos) < (30 * self.fps):  # 30 second buffer
                        position_used = True
                        break

                if not position_used:
                    break

                attempts += 1

            if attempts >= max_attempts:
                print("Could not find unused foreshadowing position")
                return False

            # Add to used positions and clean old ones
            self.used_foreshadow_positions.add(foreshadow_position)

            # Clean positions that are too old (remove positions more than 2 minutes behind current)
            cleanup_threshold = self.current_frame - (120 * self.fps)
            self.used_foreshadow_positions = {
                pos for pos in self.used_foreshadow_positions
                if pos > cleanup_threshold
            }

            # Set video to foreshadowing position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, foreshadow_position)
            self.current_frame = foreshadow_position

            # Start foreshadowing timing
            self.is_foreshadowing = True
            self.foreshadow_start_time = time.time()
            self.foreshadow_duration = duration_ms / 1000.0
            self.foreshadow_frames_played = 0

            # Start preview audio with pre-extracted segment
            foreshadow_time_seconds = foreshadow_position / self.fps
            self.audio_player.seek_to(
                foreshadow_time_seconds,
                is_preview=True,
                preview_duration=self.foreshadow_duration
            )

            print(f"Foreshadowing: jumped from {current_time_seconds:.1f}s to {foreshadow_time_seconds:.1f}s")
            print(f"Preview zone: last {preview_fraction * 100:.0f}% of video ({preview_zone_duration:.1f}s)")
            print(f"Will play preview for {self.foreshadow_duration} seconds")
            return True

        except Exception as e:
            print(f"Foreshadowing error: {e}")
            return False

    def end_foreshadowing(self):
        """Return to original video position"""
        if not self.is_foreshadowing:
            return

        try:
            # Return to saved position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.saved_position)
            self.current_frame = self.saved_position
            self.is_foreshadowing = False

            # Resume regular audio from saved position
            saved_time_seconds = self.saved_position / self.fps
            self.audio_player.seek_to(saved_time_seconds, is_preview=False)

            # Reset video timing
            self.video_start_time = time.time() - saved_time_seconds

            print(f"Foreshadowing ended: returned to frame {self.saved_position}")

        except Exception as e:
            print(f"Error ending foreshadowing: {e}")

    def check_foreshadowing_end(self):
        """Check if foreshadowing should end and return to original position"""
        if not self.is_foreshadowing:
            return

        current_time = time.time()
        elapsed_time = current_time - self.foreshadow_start_time

        # Check if we've played enough time
        if elapsed_time >= self.foreshadow_duration:
            print(f"Foreshadowing ending: played for {elapsed_time:.2f} seconds")
            self.end_foreshadowing()


    def play_next_frame(self):
        """Play next video frame with proper timing and audio sync"""
        if not self.is_playing or self.is_paused or not self.cap:
            return

        current_time = time.time()

        # Calculate target time for this frame
        if not self.is_foreshadowing:
            target_frame_time = self.video_start_time + (self.current_frame / self.fps)
        else:
            target_frame_time = current_time  # Play foreshadowing at natural speed

        # Check if we're ahead of schedule
        time_diff = target_frame_time - current_time

        if time_diff > 0.001:  # More than 1ms ahead
            # Schedule next frame at the right time
            delay_ms = max(1, int(time_diff * 1000))
            self.parent_frame.after(delay_ms, self.play_next_frame)
            return

        # Read and display frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1

            # Track foreshadowing frames
            if self.is_foreshadowing:
                self.foreshadow_frames_played += 1

            self.display_frame_with_overlay(frame)
            self.update_progress()

            # Sync audio every few frames
            if self.current_frame % 10 == 0:  # Every 10 frames
                video_time = self.current_frame / self.fps
                self.audio_player.sync_to_video_time(video_time, self.video_start_time)

            # Check if foreshadowing should end
            if self.is_foreshadowing:
                self.check_foreshadowing_end()

            # Schedule next frame
            self.parent_frame.after(1, self.play_next_frame)
        else:
            # End of video or foreshadowing segment
            if self.is_foreshadowing:
                print("Reached end of foreshadowing segment")
                self.end_foreshadowing()
                self.play_next_frame()  # Continue from saved position
            else:
                self.is_playing = False
                self.audio_player.stop()
                print("Video finished")

    def show_current_frame(self):
        """Show current frame without advancing"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.display_frame_with_overlay(frame)
                # Go back one frame since we just read it
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def add_foreshadowing_overlay(self, frame):
        """Add foreshadowing indicator overlay"""
        if self.is_foreshadowing:
            overlay = frame.copy()

            # Add border
            border_thickness = max(4, min(frame.shape[0], frame.shape[1]) // 100)  # Dynamic border size
            height, width = frame.shape[:2]

            # Gold borders
            cv2.rectangle(overlay, (0, 0), (width, border_thickness), (255, 215, 0), -1)
            cv2.rectangle(overlay, (0, height - border_thickness), (width, height), (255, 215, 0), -1)
            cv2.rectangle(overlay, (0, 0), (border_thickness, height), (255, 215, 0), -1)
            cv2.rectangle(overlay, (width - border_thickness, 0), (width, height), (255, 215, 0), -1)

            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

            # Add text with countdown
            elapsed_time = time.time() - self.foreshadow_start_time
            remaining_time = max(0, self.foreshadow_duration - elapsed_time)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"🔮 PREVIEW - COMING UP! ({remaining_time:.1f}s)"

            # Dynamic font size based on frame size
            font_scale = min(width, height) / 800.0  # Scale font to frame size
            font_scale = max(0.5, min(font_scale, 2.0))  # Clamp between 0.5 and 2.0
            thickness = max(1, int(font_scale * 2))

            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            x = (width - text_size[0]) // 2
            y = max(30, int(height * 0.08))  # Position relative to frame height

            # Black background for text
            padding = 10
            cv2.rectangle(frame, (x - padding, y - text_size[1] - padding),
                          (x + text_size[0] + padding, y + padding), (0, 0, 0), -1)
            cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness)

        return frame

    def display_frame_with_overlay(self, frame):
        """Display frame with proper aspect ratio and foreshadowing overlays"""
        try:
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Use default size if canvas not ready
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 800, 450

            # Get original frame dimensions
            original_height, original_width = frame.shape[:2]
            original_aspect = original_width / original_height
            canvas_aspect = canvas_width / canvas_height

            # Calculate dimensions maintaining aspect ratio
            if original_aspect > canvas_aspect:
                # Video is wider - fit to width, add black bars top/bottom
                new_width = canvas_width
                new_height = int(canvas_width / original_aspect)
                x_offset = 0
                y_offset = (canvas_height - new_height) // 2
            else:
                # Video is taller - fit to height, add black bars left/right
                new_height = canvas_height
                new_width = int(canvas_height * original_aspect)
                x_offset = (canvas_width - new_width) // 2
                y_offset = 0

            # Resize frame maintaining aspect ratio
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # Add foreshadowing overlay if active (on resized frame)
            frame_with_overlay = self.add_foreshadowing_overlay(frame_resized)

            # Create black canvas
            canvas_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

            # Place resized frame on black canvas (letterboxing/pillarboxing)
            canvas_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame_with_overlay

            # Convert to RGB and display
            frame_rgb = cv2.cvtColor(canvas_frame, cv2.COLOR_BGR2RGB)

            from PIL import Image, ImageTk
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
            self.canvas.image = photo

        except Exception as e:
            print(f"Error displaying frame: {e}")

    def update_progress(self):
        """Update progress bar and time display"""
        if self.total_frames > 0:
            # Use saved position for progress when foreshadowing
            display_frame = self.saved_position if self.is_foreshadowing else self.current_frame
            progress = (display_frame / self.total_frames) * 100
            self.progress_var.set(progress)

            current_seconds = display_frame / self.fps
            total_seconds = self.total_frames / self.fps

            current_time = f"{int(current_seconds // 60):02d}:{int(current_seconds % 60):02d}"
            total_time = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"

            if self.is_foreshadowing:
                foreshadow_seconds = self.current_frame / self.fps
                foreshadow_time = f"{int(foreshadow_seconds // 60):02d}:{int(foreshadow_seconds % 60):02d}"
                self.time_var.set(f"{current_time} / {total_time} (Preview: {foreshadow_time})")
            else:
                self.time_var.set(f"{current_time} / {total_time}")

    def seek_to_position(self, position_seconds: float):
        """Seek to specific position in video"""
        if self.cap and self.total_frames > 0:
            target_frame = int(position_seconds * self.fps)
            target_frame = max(0, min(target_frame, self.total_frames - 1))

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.current_frame = target_frame

            # Reset timing
            if self.is_playing:
                self.video_start_time = time.time() - position_seconds
                self.audio_player.seek_to(position_seconds)

            self.show_current_frame()
            print(f"Seeked to {position_seconds:.2f}s (frame {target_frame})")

    def get_current_time(self) -> float:
        """Get current playback time in seconds"""
        if self.fps > 0:
            return self.current_frame / self.fps
        return 0.0

    def get_duration(self) -> float:
        """Get total video duration in seconds"""
        if self.fps > 0 and self.total_frames > 0:
            return self.total_frames / self.fps
        return 0.0

# Foreshadowing Engine
# ═══════════════════════════════════════════════════════════════════════════════

class ForeshadowingEngine:
    """Generate foreshadowing effects"""

    def __init__(self, video_processor: MultiBackendVideoProcessor):
        self.video_processor = video_processor
        self.timestamps = []
        self.used_timestamps = set()

    def prepare_foreshadowing(self, video_path: Optional[str] = None):
        """Prepare foreshadowing content"""
        if video_path:
            self.timestamps = self.video_processor.extract_timestamps(video_path)
        else:
            # Generate timestamps for browser playback
            duration = self.video_processor.video_info.duration if self.video_processor.video_info else 60
            self.timestamps = [i * duration / 5 for i in range(1, 5)]

        print(f"Prepared {len(self.timestamps)} timestamps for foreshadowing")

    def generate_text_foreshadow(self) -> str:
        """Generate text-based foreshadowing"""
        phrases = [
            "🔥 Something amazing is coming up!",
            "⚡ Wait until you see what happens next!",
            "🎯 The best part is just ahead...",
            "🚀 You won't want to miss this...",
            "💫 Get ready for the surprise!",
            "🎪 The climax is approaching...",
            "🎭 Stay tuned for the big reveal!",
            "🌟 Epic moment incoming...",
            "💥 Brace yourself for this...",
            "🎨 The magic happens soon!",
        ]
        return random.choice(phrases)

    def get_next_timestamp(self) -> Optional[float]:
        """Get next unused timestamp"""
        available = [t for i, t in enumerate(self.timestamps) if i not in self.used_timestamps]

        if available:
            timestamp = random.choice(available)
            self.used_timestamps.add(self.timestamps.index(timestamp))
            return timestamp

        return None

    def generate_time_hint(self, timestamp: float) -> str:
        """Generate time-based hint"""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        return f"⏰ Something exciting at {minutes}:{seconds:02d}!"


import pygame
import threading
import subprocess
import tempfile

import subprocess
import threading
import time

import subprocess
import threading
import time
import os

import subprocess
import threading
import time
import os
import tempfile


class AudioPlayer:
    """Handle audio playback with better handling for short clips"""

    def __init__(self):
        self.audio_available = self.check_ffplay()
        self.current_audio_file = None
        self.audio_process = None
        self.is_playing = False
        self.current_position = 0.0
        self.audio_thread = None
        self.should_stop = False

        # For preview clips - pre-extract segments
        self.preview_segments = {}
        self.temp_dir = tempfile.mkdtemp()

    def check_ffplay(self) -> bool:
        """Check if ffplay is available"""
        try:
            result = subprocess.run(['ffplay', '-version'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print("Audio system initialized with ffplay")
                return True
        except:
            pass

        print("ffplay not available - audio disabled")
        return False

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file"""
        if not self.audio_available:
            return None

        try:
            audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'

            # Skip if already exists
            if os.path.exists(audio_path):
                print(f"Audio file already exists: {audio_path}")
                return audio_path

            # Use ffmpeg to extract audio with better settings for streaming
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '2',
                '-avoid_negative_ts', 'make_zero',
                '-fflags', '+genpts',  # Generate presentation timestamps
                '-y', audio_path
            ]

            print("Extracting audio...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Audio extracted to: {audio_path}")
                return audio_path
            else:
                print(f"Audio extraction failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"Audio extraction error: {e}")
            return None

    def load_audio(self, audio_path: str):
        """Load audio file"""
        if not self.audio_available or not audio_path:
            return

        self.current_audio_file = audio_path
        print(f"Audio loaded: {audio_path}")

    def prepare_preview_segment(self, start_time: float, duration: float) -> str:
        """Pre-extract a preview audio segment for smooth playback"""
        if not self.audio_available or not self.current_audio_file:
            return None

        try:
            # Create unique filename for this segment
            segment_key = f"{start_time:.1f}_{duration:.1f}"
            segment_path = os.path.join(self.temp_dir, f"preview_{segment_key}.wav")

            # Check if already exists
            if segment_key in self.preview_segments:
                return self.preview_segments[segment_key]

            # Extract segment with padding for smoother playback
            padding = 0.5  # 0.5 second padding on each side
            extract_start = max(0, start_time - padding)
            extract_duration = duration + (2 * padding)

            cmd = [
                'ffmpeg',
                '-ss', f"{extract_start:.3f}",
                '-i', self.current_audio_file,
                '-t', f"{extract_duration:.3f}",
                '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '2',
                '-af', f'adelay={int(padding * 1000)}|{int(padding * 1000)}',  # Add delay to account for padding
                '-y', segment_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.preview_segments[segment_key] = segment_path
                print(f"Preview segment prepared: {start_time:.1f}s for {duration:.1f}s")
                return segment_path
            else:
                print(f"Preview segment extraction failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"Preview segment error: {e}")
            return None

    def start_playback(self, start_position: float = 0.0, is_preview: bool = False, preview_duration: float = 0.0):
        """Start audio playback with special handling for previews"""
        if not self.audio_available or not self.current_audio_file:
            return

        try:
            # Stop any existing audio
            self.stop()

            self.current_position = start_position
            self.should_stop = False

            if is_preview and preview_duration > 0:
                # Use pre-extracted segment for preview
                segment_path = self.prepare_preview_segment(start_position, preview_duration)
                if segment_path:
                    self._start_preview_audio(segment_path, preview_duration)
                else:
                    # Fallback to regular method
                    self._start_regular_audio(start_position)
            else:
                # Regular playback
                self._start_regular_audio(start_position)

            self.is_playing = True
            print(f"Audio started from {start_position:.2f}s {'(preview)' if is_preview else ''}")

        except Exception as e:
            print(f"Audio start error: {e}")

    def _start_preview_audio(self, segment_path: str, duration: float):
        """Start preview audio with pre-extracted segment"""

        def play_preview():
            try:
                self._is_preview_playing = True
                self._playback_start_time = time.time()

                cmd = [
                    'ffplay',
                    '-autoexit',
                    '-nodisp',
                    '-v', 'quiet',
                    '-af', f'volume=1.0',  # Ensure consistent volume
                    segment_path
                ]

                self.audio_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Wait for duration or until stopped
                start_time = time.time()
                while (time.time() - start_time < duration and
                       self.audio_process and
                       self.audio_process.poll() is None and
                       not self.should_stop):
                    time.sleep(0.01)

                # Clean stop
                if self.audio_process:
                    try:
                        self.audio_process.terminate()
                        self.audio_process.wait(timeout=0.5)
                    except:
                        try:
                            self.audio_process.kill()
                        except:
                            pass

            except Exception as e:
                print(f"Preview audio error: {e}")
            finally:
                self.audio_process = None
                self._is_preview_playing = False
                if hasattr(self, '_playback_start_time'):
                    delattr(self, '_playback_start_time')

        # Start in thread
        self.audio_thread = threading.Thread(target=play_preview, daemon=True)
        self.audio_thread.start()

    def _start_regular_audio(self, start_position: float):
        """Start regular audio playback"""

        def play_regular():
            try:
                self._playback_start_time = time.time()
                self._is_preview_playing = False

                cmd = [
                    'ffplay',
                    '-ss', f"{start_position:.3f}",
                    '-nodisp',
                    '-autoexit',
                    '-v', 'quiet',
                    '-af', 'aresample=async=1:first_pts=0',  # Better sync
                    self.current_audio_file
                ]

                self.audio_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Wait for process to finish or be stopped
                while (self.audio_process and
                       self.audio_process.poll() is None and
                       not self.should_stop):
                    time.sleep(0.1)

            except Exception as e:
                print(f"Regular audio error: {e}")
            finally:
                self.audio_process = None
                if hasattr(self, '_playback_start_time'):
                    delattr(self, '_playback_start_time')

        # Start in thread
        self.audio_thread = threading.Thread(target=play_regular, daemon=True)
        self.audio_thread.start()

    def get_current_position(self) -> float:
        """Get current audio position"""
        return self.current_position

    def pause(self):
        """Pause audio playback"""
        if self.is_playing:
            self.stop()
            print(f"Audio paused at {self.current_position:.2f}s")

    def resume(self):
        """Resume audio playback"""
        if not self.is_playing:
            self.start_playback(self.current_position)
            print(f"Audio resumed from {self.current_position:.2f}s")

    def seek_to(self, position: float, is_preview: bool = False, preview_duration: float = 0.0):
        """Seek to specific position with preview support"""
        was_playing = self.is_playing
        self.current_position = position

        if was_playing:
            self.start_playback(position, is_preview, preview_duration)

        print(f"Audio seeked to {position:.2f}s {'(preview)' if is_preview else ''}")

    def stop(self):
        """Stop audio completely"""
        self.should_stop = True

        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=1)
            except:
                try:
                    self.audio_process.kill()
                except:
                    pass
            self.audio_process = None

        self.is_playing = False

    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass

    def sync_to_video_time(self, video_position: float, video_start_time: float):
            """Synchronize audio to video position"""
            if not self.is_playing or not self.audio_available:
                return

            # For preview clips, don't sync (they're pre-extracted and short)
            if hasattr(self, '_is_preview_playing') and self._is_preview_playing:
                return

            # Calculate expected audio position based on video
            current_time = time.time()
            expected_position = video_position

            # Get actual audio position estimate
            if hasattr(self, '_playback_start_time'):
                actual_position = self.current_position + (current_time - self._playback_start_time)
            else:
                actual_position = self.current_position

            # Check if we're significantly out of sync (more than 0.3 seconds)
            sync_threshold = 0.3
            position_diff = abs(actual_position - expected_position)

            if position_diff > sync_threshold:
                print(f"Audio out of sync by {position_diff:.2f}s, restarting at {expected_position:.2f}s")
                self.start_playback(expected_position)


class EEGVideoPlayer:
    """Main video player with EEG engagement monitoring"""

    def __init__(self):
        # Initialize communication queue FIRST
        self.gui_queue = queue.Queue()

        # Components
        self.engagement_monitor = EngagementMonitor()
        self.video_processor = MultiBackendVideoProcessor()
        self.foreshadowing_engine = ForeshadowingEngine(self.video_processor)

        # EEG streaming
        self.eeg_client = None
        self.streaming = False
        self.eeg_thread = None

        # Video state
        self.video_info = None
        self.video_player = None  # Will be created with GUI

        # Simulation state
        self.simulation_running = False

        # GUI (must be after gui_queue initialization)
        self.setup_gui()

    def setup_gui(self):
        """Setup GUI interface"""
        self.root = tk.Tk()
        self.root.title("EEG-Enhanced YouTube Player with Video Foreshadowing")
        self.root.geometry("1200x800")

        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Right panel for video
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # Setup left panel
        self.setup_controls(left_frame)

        # Setup video player in right panel
        self.setup_video_player(right_frame)

    def setup_controls(self, parent):
        """Setup control panel"""
        # Title
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(title_frame, text="🎬 EEG Video Player",
                  font=("Arial", 16, "bold")).pack()

        # Backend info
        backend_info = "OpenCV Video Player with Foreshadowing"
        ttk.Label(title_frame, text=backend_info,
                  font=("Arial", 10, "italic")).pack()

        # URL input
        url_frame = ttk.LabelFrame(parent, text="YouTube Video")
        url_frame.pack(fill=tk.X, padx=5, pady=5)

        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(url_frame, textvariable=self.url_var, width=40)
        url_entry.pack(fill=tk.X, padx=5, pady=2)

        # Example URL for testing
        example_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.url_var.set(example_url)

        self.load_btn = ttk.Button(url_frame, text="📥 Load Video", command=self.load_video)
        self.load_btn.pack(pady=2)

        # Video controls
        control_frame = ttk.LabelFrame(parent, text="Video Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=5)

        self.play_btn = ttk.Button(btn_frame, text="▶️ Play", command=self.play_video, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(btn_frame, text="⏸️ Pause", command=self.pause_video, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Stop", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        # EEG controls
        eeg_frame = ttk.LabelFrame(parent, text="EEG Monitoring")
        eeg_frame.pack(fill=tk.X, padx=5, pady=5)

        eeg_btn_frame = ttk.Frame(eeg_frame)
        eeg_btn_frame.pack(pady=5)

        self.eeg_btn = ttk.Button(eeg_btn_frame, text="🧠 Start EEG", command=self.start_eeg_monitoring)
        self.eeg_btn.pack(side=tk.LEFT, padx=2)

        self.simulate_btn = ttk.Button(eeg_btn_frame, text="🔄 Simulate", command=self.start_simulation)
        self.simulate_btn.pack(side=tk.LEFT, padx=2)

        # Manual foreshadowing button for testing
        manual_foreshadow_btn = ttk.Button(eeg_btn_frame, text="🔮 Test Preview", command=self.manual_foreshadowing)
        manual_foreshadow_btn.pack(side=tk.LEFT, padx=2)

        # Status display
        status_frame = ttk.LabelFrame(parent, text="EEG Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        self.engagement_var = tk.StringVar(value="Engagement: --")
        self.baseline_var = tk.StringVar(value="Baseline: Not established")

        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W, padx=5)
        ttk.Label(status_frame, textvariable=self.engagement_var).pack(anchor=tk.W, padx=5)
        ttk.Label(status_frame, textvariable=self.baseline_var).pack(anchor=tk.W, padx=5)

        # Video info
        info_frame = ttk.LabelFrame(parent, text="Video Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, font=("Arial", 9))
        scrollbar1 = ttk.Scrollbar(info_frame, orient="vertical", command=self.video_info_text.yview)
        self.video_info_text.configure(yscrollcommand=scrollbar1.set)

        self.video_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)

        # Foreshadowing display
        foreshadow_frame = ttk.LabelFrame(parent, text="🎬 Foreshadowing Events")
        foreshadow_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.foreshadow_text = tk.Text(foreshadow_frame, height=8, wrap=tk.WORD,
                                       bg="#f0f0f0", font=("Arial", 9))
        scrollbar2 = ttk.Scrollbar(foreshadow_frame, orient="vertical", command=self.foreshadow_text.yview)
        self.foreshadow_text.configure(yscrollcommand=scrollbar2.set)

        self.foreshadow_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_video_player(self, parent):
        """Setup video display area"""
        video_frame = ttk.LabelFrame(parent, text="📺 Video Display with Foreshadowing")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create OpenCV player
        self.video_player = OpenCVVideoPlayer(video_frame)

        # Start queue processing
        self.process_gui_queue()

    def process_gui_queue(self):
        """Process GUI update messages"""
        try:
            while True:
                message = self.gui_queue.get_nowait()
                self.handle_gui_message(message)
        except queue.Empty:
            pass

        self.root.after(100, self.process_gui_queue)

    def handle_gui_message(self, message):
        """Handle GUI update messages"""
        msg_type, data = message

        if msg_type == 'engagement_update':
            state = data

            # Get phase information
            phase_info = self.engagement_monitor.get_phase_info()

            if phase_info["phase"] == "discarding":
                self.engagement_var.set(f"🔄 Stabilizing: {phase_info['progress']:.0f}%")
                self.baseline_var.set("Discarding initial noisy data...")
            elif phase_info["phase"] == "collecting_baseline":
                self.engagement_var.set(f"📊 Baseline: {phase_info['progress']:.0f}%")
                self.baseline_var.set(phase_info["message"])
            else:
                # Normal monitoring phase
                self.engagement_var.set(f"Engagement: {state.current_score:.3f}")
                if state.baseline_score > 0:
                    self.baseline_var.set(
                        f"Baseline: {state.baseline_score:.3f} ({'✅ Engaged' if state.is_engaged else '⚠️ Low'})")

            # Check for foreshadowing trigger (only in monitoring phase)
            if (phase_info["phase"] == "monitoring" and
                    not state.is_engaged and
                    state.baseline_score > 0 and
                    time.time() - state.last_foreshadow_time > FORESHADOW_COOLDOWN):
                self.trigger_foreshadowing()

        elif msg_type == 'status_update':
            self.status_var.set(data)

        elif msg_type == 'foreshadow_event':
            timestamp = datetime.now().strftime("%H:%M:%S")
            event_text = f"[{timestamp}] {data}\n"
            self.foreshadow_text.insert(tk.END, event_text)
            self.foreshadow_text.see(tk.END)

    def load_video(self):
        """Load YouTube video"""
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return

        try:
            self.status_var.set("Loading video...")
            self.load_btn.config(state=tk.DISABLED)

            # Get video info
            self.video_info = self.video_processor.get_video_info(url)

            # Display info
            info_text = f"Title: {self.video_info.title}\n"
            info_text += f"Duration: {self.video_info.duration} seconds\n"
            info_text += f"URL: {self.video_info.url}\n"
            if self.video_info.description:
                info_text += f"Description: {self.video_info.description[:200]}..."

            self.video_info_text.delete(1.0, tk.END)
            self.video_info_text.insert(1.0, info_text)

            # Download video
            if YTDLP_AVAILABLE:
                self.status_var.set("Downloading video...")
                video_path = self.video_processor.download_video(url)

                # Load into video player
                if self.video_player.load_video(video_path):
                    self.status_var.set("Video loaded successfully!")
                    self.play_btn.config(state=tk.NORMAL)
                    self.pause_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.NORMAL)

                    # Prepare foreshadowing
                    self.foreshadowing_engine.prepare_foreshadowing(video_path)
                else:
                    self.status_var.set("Failed to load video")

            else:
                # Browser fallback
                self.status_var.set("yt-dlp not available")
                messagebox.showerror("Error", "yt-dlp is required for video foreshadowing")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            self.status_var.set("Error loading video")
        finally:
            self.load_btn.config(state=tk.NORMAL)

    def play_video(self):
        """Start video playback"""
        if self.video_player:
            self.video_player.play()
        self.status_var.set("Playing video...")

    def pause_video(self):
        """Pause video playback"""
        if self.video_player:
            self.video_player.pause()
        self.status_var.set("Video paused")

    def stop_video(self):
        """Stop video playback"""
        if self.video_player:
            self.video_player.stop()
        self.status_var.set("Video stopped")

    def manual_foreshadowing(self):
        """Manually trigger foreshadowing for testing"""
        self.trigger_foreshadowing()

    def trigger_foreshadowing(self):
        """Trigger video foreshadowing - jump ahead and return"""
        try:
            # Update foreshadow timing
            current_time = time.time()
            self.engagement_monitor.current_state.last_foreshadow_time = current_time
            self.engagement_monitor.current_state.foreshadow_count += 1

            # Choose foreshadowing type
            foreshadow_type = random.choice(['video_preview', 'quick_preview', 'long_preview'])

            if foreshadow_type == 'video_preview':
                duration = 2000  # 2 seconds
                if hasattr(self.video_player, 'start_video_foreshadowing'):
                    success = self.video_player.start_video_foreshadowing(duration)
                    if success:
                        self.gui_queue.put(('foreshadow_event', f"🎬 Video Preview: 2-second glimpse ahead!"))
                    else:
                        self.gui_queue.put(('foreshadow_event', f"⚠️ Foreshadowing failed - video too short"))

            elif foreshadow_type == 'quick_preview':
                duration = 1500  # 1.5 seconds
                if hasattr(self.video_player, 'start_video_foreshadowing'):
                    success = self.video_player.start_video_foreshadowing(duration)
                    if success:
                        self.gui_queue.put(('foreshadow_event', f"⚡ Quick Preview: 1.5-second sneak peek!"))

            elif foreshadow_type == 'long_preview':
                duration = 3000  # 3 seconds
                if hasattr(self.video_player, 'start_video_foreshadowing'):
                    success = self.video_player.start_video_foreshadowing(duration)
                    if success:
                        self.gui_queue.put(('foreshadow_event', f"🔮 Extended Preview: 3-second future glimpse!"))

        except Exception as e:
            print(f"Foreshadowing error: {e}")

    # ... (rest of the methods remain the same: start_eeg_monitoring, stop_eeg_monitoring, etc.)
    def start_eeg_monitoring(self):
        """Start EEG monitoring"""
        if not self.streaming:
            self.eeg_thread = threading.Thread(target=self.run_eeg_thread, daemon=True)
            self.eeg_thread.start()
            self.eeg_btn.config(text="🛑 Stop EEG", command=self.stop_eeg_monitoring)

    def stop_eeg_monitoring(self):
        """Stop EEG monitoring"""
        self.streaming = False
        self.eeg_btn.config(text="🧠 Start EEG", command=self.start_eeg_monitoring)

    def start_simulation(self):
        """Start simulated EEG data for testing"""
        if not self.simulation_running:
            self.simulation_running = True
            simulation_thread = threading.Thread(target=self.run_simulation, daemon=True)
            simulation_thread.start()
            self.simulate_btn.config(text="🛑 Stop Sim", command=self.stop_simulation)

    def stop_simulation(self):
        """Stop EEG simulation"""
        self.simulation_running = False
        self.simulate_btn.config(text="🔄 Simulate", command=self.start_simulation)

    def run_simulation(self):
        """Run simulated EEG data"""
        print("Starting EEG simulation...")
        baseline_phase = True
        baseline_start = time.time()

        while self.simulation_running:
            # Generate fake EEG data
            ch1_data = np.random.normal(0, 100, SAMPLES_PER_CHUNK)
            ch2_data = np.random.normal(0, 100, SAMPLES_PER_CHUNK)

            # Simulate engagement patterns
            if baseline_phase:
                # Higher engagement during baseline
                engagement_boost = np.random.normal(50, 10, SAMPLES_PER_CHUNK)
                if time.time() - baseline_start > BASELINE_DURATION:
                    baseline_phase = False
                    print("Baseline established - starting engagement simulation")
            else:
                # Randomly drop engagement to trigger foreshadowing
                if random.random() < 0.1:  # 10% chance of engagement drop
                    engagement_boost = np.random.normal(-30, 5, SAMPLES_PER_CHUNK)
                else:
                    engagement_boost = np.random.normal(20, 10, SAMPLES_PER_CHUNK)

            ch1_data += engagement_boost
            ch2_data += engagement_boost

            # Process through engagement monitor
            self.engagement_monitor.add_data(ch1_data, ch2_data)
            state = self.engagement_monitor.update_engagement_state()

            # Send to GUI
            self.gui_queue.put(('engagement_update', state))

            time.sleep(0.1)  # 10 Hz update rate

    def run_eeg_thread(self):
        """Run EEG streaming in background thread"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            device_address = loop.run_until_complete(find_device())
            loop.run_until_complete(self.start_eeg_streaming(device_address))

        except Exception as e:
            print(f"EEG thread error: {e}")
            self.gui_queue.put(('status_update', f"EEG Error: {e}"))

    async def start_eeg_streaming(self, device_address: str):
        """Start EEG data streaming"""
        self.eeg_client = BleakClient(device_address, timeout=20.0)
        await self.eeg_client.connect()

        if not self.eeg_client.is_connected:
            raise RuntimeError("Failed to connect to EEG device")

        self.gui_queue.put(('status_update', "EEG connected"))

        await self.eeg_client.start_notify(TX_UUID, self.eeg_notification_handler)

        start_cmd = build_stream_command(True)
        await self.eeg_client.write_gatt_char(RX_UUID, start_cmd, response=False)

        self.streaming = True

        try:
            while self.streaming and self.eeg_client.is_connected:
                await asyncio.sleep(0.1)
        finally:
            if self.eeg_client and self.eeg_client.is_connected:
                stop_cmd = build_stream_command(False)
                await self.eeg_client.write_gatt_char(RX_UUID, stop_cmd, response=False)
                await self.eeg_client.stop_notify(TX_UUID)
                await self.eeg_client.disconnect()

    def eeg_notification_handler(self, sender: int, data: bytearray):
        """Handle EEG data"""
        try:
            if len(data) < 6:
                return

            ch1_samples, ch2_samples = parse_eeg_packet(data[2:])

            # Add to engagement monitor
            self.engagement_monitor.add_data(np.array(ch1_samples), np.array(ch2_samples))

            # Update engagement state
            state = self.engagement_monitor.update_engagement_state()

            # Send to GUI
            self.gui_queue.put(('engagement_update', state))

        except Exception as e:
            print(f"EEG processing error: {e}")

    def run(self):
        """Start the application"""
        self.root.mainloop()

# ═══════════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════════

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg available for audio")
            return True
    except:
        pass

    print("❌ FFmpeg not found - audio will be disabled")
    print("   Install with: brew install ffmpeg")
    return False


def main():
    print("EEG-Enhanced YouTube Short Video Player")
    print("=" * 50)

    # Check available backends
    print("Available backends:")
    if VLC_AVAILABLE:
        print("✅ VLC")
    else:
        print("❌ VLC (install with: brew install --cask vlc)")

    if OPENCV_AVAILABLE:
        print("✅ OpenCV")
    else:
        print("❌ OpenCV (install with: pip install opencv-python)")

    if YTDLP_AVAILABLE:
        print("✅ yt-dlp")
    else:
        print("❌ yt-dlp (install with: pip install yt-dlp)")

    # Check ffmpeg
    check_ffmpeg()

    print("✅ System Player (always available)")
    print("=" * 50)

    try:
        player = EEGVideoPlayer()
        player.run()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)