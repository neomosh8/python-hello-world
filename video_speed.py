#!/usr/bin/env python3
"""
YouTube Short Video Adaptive Speed Player with EEG Engagement Monitoring - FIXED
Uses Neocore EEG to detect engagement drops and adjusts playback speed accordingly
Lower engagement = faster playback (1.5x to 2x speed)
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
ENGAGEMENT_WINDOW = 5  # seconds for engagement calculation

# Engagement thresholds
BASELINE_DURATION = 20  # seconds to establish baseline
SPEED_THRESHOLD_LOW = 0.8  # 20% below baseline triggers 1.5x speed
SPEED_THRESHOLD_HIGH = 0.6  # 40% below baseline triggers 2.0x speed
SPEED_SMOOTHING = 0.05  # How quickly speed changes (reduced for stability)
SPEED_CHANGE_MIN_INTERVAL = 2.0  # Minimum seconds between speed changes

# ── at the top of the file, keep with the other constants ─────────────────────
ENGAGE_CONFIRM_TIME     = 3.0     # seconds above baseline before we call it engaged
DISENGAGE_CONFIRM_TIME  = 2.0     # seconds below threshold before we call it disengaged

@dataclass
class EngagementState:
    """Current engagement state with speed control"""
    current_score: float
    baseline_score: float
    is_engaged: bool
    recommended_speed: float
    speed_reason: str


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
# EEG Processing - FIXED
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
    """Real-time engagement monitoring from EEG with speed recommendations - FIXED"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.fs = sample_rate
        self.window_size = sample_rate * ENGAGEMENT_WINDOW

        # Data buffers
        self.ch1_buffer = deque(maxlen=self.window_size)
        self.ch2_buffer = deque(maxlen=self.window_size)

        # Timing and phases
        self.data_start_time = None
        self.discard_duration = 15.0  # Discard first 5 seconds
        self.baseline_duration = BASELINE_DURATION

        # Engagement tracking phases
        self.phase = "discarding"  # "discarding" -> "collecting_baseline" -> "monitoring"
        self.baseline_scores = []
        self.baseline_established = False
        self.baseline_mean = 0.0

        # Speed control
        self.current_speed = 1.0
        self.target_speed = 1.0
        self.last_speed_change = 0

        self.engaged_state   = True
        self._above_start    = None
        self._below_start    = None

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

        # ─── inside class EngagementMonitor ────────────────────────────────────────────
    def calculate_recommended_speed(self, current_score: float) -> Tuple[float, str]:
            """
            Map engagement to fixed speeds:
                • ≥ 80 % of baseline → 1.0 x
                • 60 % – 79 %       → 1.5 x
                • < 60 %             → 2.0 x
            """
            if not self.baseline_established or self.baseline_mean <= 0:
                return 1.0, "No baseline"

            ratio = current_score / self.baseline_mean
            now = time.time()

            # Cooldown so we do not flip‑flop every update
            if now - self.last_speed_change < SPEED_CHANGE_MIN_INTERVAL:
                return self.current_speed, "Cooldown"

            if ratio >= SPEED_THRESHOLD_LOW:  # ≥ 0.8
                target, reason = 1.0, "Engaged"
            elif ratio >= SPEED_THRESHOLD_HIGH:  # 0.6 – 0.79
                target, reason = 1.5, "Moderate drop"
            else:  # < 0.6
                target, reason = 2.0, "Low engagement"

            if abs(target - self.current_speed) >= 0.1:
                self.current_speed = target
                self.last_speed_change = now

            return self.current_speed, reason

    def update_engagement_state(self) -> EngagementState:
        """
        Advance the engagement‑monitoring state machine
        and return the current EngagementState instance.

        Phases
        -------
        1. "discarding"           – ignore the first few seconds while sensors settle
        2. "collecting_baseline"  – build a baseline engagement value
        3. "monitoring"           – normal run time

        Extra logic
        ------------
        • Discrete speed levels: 1.0 x, 1.5 x, 2.0 x
        • Hysteresis on engaged ⇄ disengaged transitions:
            - must stay **below** baseline for DISENGAGE_CONFIRM_TIME to disengage
            - must stay **above** baseline for ENGAGE_CONFIRM_TIME to re‑engage
        """
        # No data yet
        if self.data_start_time is None:
            return EngagementState(0.0, 0.0, True, 1.0, "Waiting for data")

        elapsed = time.time() - self.data_start_time
        current_score = self.calculate_engagement()

        # ── Phase 1: discard noisy start‑up data ───────────────────────────────
        if self.phase == "discarding":
            remaining = self.discard_duration - elapsed
            if remaining > 0:
                return EngagementState(
                    current_score, 0.0, True, 1.0,
                    f"Stabilizing ({remaining:.1f}s)"
                )
            # move to baseline phase
            self.phase = "collecting_baseline"
            self.baseline_start_time = time.time()
            self.baseline_scores.clear()

        # ── Phase 2: collect baseline ─────────────────────────────────────────
        if self.phase == "collecting_baseline":
            base_elapsed = time.time() - self.baseline_start_time
            if base_elapsed < self.baseline_duration:
                self.baseline_scores.append(current_score)
                progress = (base_elapsed / self.baseline_duration) * 100
                return EngagementState(
                    current_score, 0.0, True, 1.0,
                    f"Baseline {progress:.0f}%"
                )

            # baseline ready
            scores = np.clip(
                self.baseline_scores,
                np.percentile(self.baseline_scores, 5),
                np.percentile(self.baseline_scores, 95)
            )
            self.baseline_mean = float(np.median(scores))
            self.baseline_established = True
            self.phase = "monitoring"
            print(f"Baseline (median) = {self.baseline_mean:.3f}")

        # ── Phase 3: monitoring ──────────────────────────────────────────────
        speed, reason = self.calculate_recommended_speed(current_score)

        # engaged / disengaged debouncing
        if self.baseline_established and self.baseline_mean > 0:
            ratio = current_score / self.baseline_mean
            now = time.time()

            if self.engaged_state:
                if ratio < 1.0:  # below baseline
                    self._below_start = self._below_start or now
                    if now - self._below_start >= DISENGAGE_CONFIRM_TIME:
                        self.engaged_state = False
                        self._above_start = None
                else:
                    self._below_start = None
            else:
                if ratio >= 1.0:  # at or above baseline
                    self._above_start = self._above_start or now
                    if now - self._above_start >= ENGAGE_CONFIRM_TIME:
                        self.engaged_state = True
                        self._below_start = None
                else:
                    self._above_start = None

        # optional debug print every 5 s
        if not hasattr(self, "_last_debug_time"):
            self._last_debug_time = time.time()
        if time.time() - self._last_debug_time > 5:
            if self.baseline_established:
                ratio = current_score / self.baseline_mean
                print(
                    f"Engagement {current_score:.3f} "
                    f"({ratio:.2f}× baseline) "
                    f"→ speed {speed:.1f}× ({reason})"
                )
            self._last_debug_time = time.time()

        return EngagementState(
            current_score,
            self.baseline_mean,
            self.engaged_state,
            speed,
            reason
        )

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
                "message": "Monitoring engagement for speed control",
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
            return VideoInfo(
                title="Browser Playback",
                duration=60,
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
            'format': 'best[height<=720]',
            'outtmpl': os.path.join(self.temp_dir, '%(title)s.%(ext)s'),
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)

        self.video_info.local_path = video_path
        return video_path


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Speed Video Player - FIXED
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveSpeedVideoPlayer:
    """OpenCV-based video player with adaptive playback speed based on engagement - FIXED"""

    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.cap = None
        self.is_playing = False
        self.is_paused = False
        self.video_path = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30

        # Speed control - FIXED
        self.playback_speed = 1.0
        self.last_frame_time = 0
        self.accumulated_time = 0.0  # Track accumulated virtual time

        # Audio player
        self.audio_player = AdaptiveAudioPlayer()

        # Playback control
        self.playback_start_time = 0

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

        # Time and speed display
        info_frame = tk.Frame(progress_frame)
        info_frame.pack(fill=tk.X)

        self.time_var = tk.StringVar(value="00:00 / 00:00")
        time_label = ttk.Label(info_frame, textvariable=self.time_var)
        time_label.pack(side=tk.LEFT)

        self.speed_var = tk.StringVar(value="Speed: 1.0x")
        speed_label = ttk.Label(info_frame, textvariable=self.speed_var, foreground="blue")
        speed_label.pack(side=tk.RIGHT)

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
            self.current_frame = 0
            self.accumulated_time = 0.0

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
        """Start video playback"""
        if self.cap:
            self.is_playing = True
            self.is_paused = False

            # Set timing references
            self.playback_start_time = time.time()
            self.last_frame_time = self.playback_start_time

            # Start audio
            current_time_seconds = self.current_frame / self.fps
            self.audio_player.start_playback(current_time_seconds, self.playback_speed)

            # Start video playback
            self.play_next_frame()
            print(f"Video playback started at {self.playback_speed}x speed")

    def pause(self):
        """Pause video playback"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.audio_player.pause()
            print("Video paused")
        else:
            # Resume with current timing
            self.playback_start_time = time.time() - self.accumulated_time
            self.last_frame_time = time.time()
            self.audio_player.resume()
            print("Video resumed")
            self.play_next_frame()

    def stop(self):
        """Stop video playback"""
        self.is_playing = False
        self.is_paused = False
        self.current_frame = 0
        self.accumulated_time = 0.0
        self.audio_player.stop()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.show_current_frame()
        print("Video stopped")

        # ─── inside class AdaptiveSpeedVideoPlayer ────────────────────────────────────
    def set_playback_speed(self, speed: float):
            """Apply new speed and keep accumulated time coherent."""
            speed = max(0.25, min(speed, 3.0))
            if abs(speed - self.playback_speed) < 0.05:
                return

            # 1) current video timestamp BEFORE changing anything
            video_pos_sec = self.current_frame / self.fps if self.fps else 0.0

            # 2) restart / retime audio first
            self.audio_player.change_speed_at_position(speed, video_pos_sec)

            # 3) retime video clock
            now = time.time()
            self.accumulated_time = video_pos_sec  # real seconds already shown
            self.playback_start_time = now
            self.last_frame_time = now
            self.playback_speed = speed

            # 4) UI feedback
            self.speed_var.set(f"Speed: {self.playback_speed:.1f}x")
            print(f"Video speed set to {self.playback_speed:.1f}x")

    def play_next_frame(self):
        """Play next video frame with speed-adjusted timing - FIXED"""
        if not self.is_playing or self.is_paused or not self.cap:
            return

        current_time = time.time()

        # Calculate frame timing with current speed
        if self.last_frame_time > 0:
            dt = current_time - self.last_frame_time
            self.accumulated_time += dt * self.playback_speed

        self.last_frame_time = current_time

        # Calculate target frame based on accumulated time
        target_frame = int(self.accumulated_time * self.fps)
        target_frame = min(target_frame, self.total_frames - 1)

        # Read frames up to target (skip if behind, wait if ahead)
        if target_frame > self.current_frame:
            # We need to advance to target frame
            frames_to_skip = min(target_frame - self.current_frame, 3)  # Don't skip too many

            for _ in range(frames_to_skip):
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.current_frame += 1

            if ret:
                self.display_frame_with_overlay(frame)
                self.update_progress()

        # Check if video is finished
        if self.current_frame >= self.total_frames:
            self.is_playing = False
            self.audio_player.stop()
            print("Video finished")
            return

        # Calculate delay for next frame based on speed
        base_delay = 1000.0 / self.fps  # Base delay in ms
        actual_delay = max(1, int(base_delay / self.playback_speed))

        self.parent_frame.after(actual_delay, self.play_next_frame)

    def show_current_frame(self):
        """Show current frame without advancing"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.display_frame_with_overlay(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def add_speed_overlay(self, frame):
        """Add speed indicator overlay"""
        if abs(self.playback_speed - 1.0) > 0.05:  # Only show if speed is significantly different
            overlay = frame.copy()
            height, width = frame.shape[:2]

            # Speed indicator
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Dynamic colors based on speed
            if self.playback_speed > 1.8:
                color = (0, 0, 255)  # Red for high speed
                text = f"⚡ {self.playback_speed:.1f}x FAST"
            elif self.playback_speed > 1.3:
                color = (0, 165, 255)  # Orange for medium speed
                text = f"🔸 {self.playback_speed:.1f}x"
            elif self.playback_speed > 1.05:
                color = (0, 255, 255)  # Yellow for slight speed
                text = f"▶️ {self.playback_speed:.1f}x"
            else:
                return frame  # No overlay for normal speed

            # Dynamic font size
            font_scale = min(width, height) / 1000.0
            font_scale = max(0.5, min(font_scale, 1.5))
            thickness = max(1, int(font_scale * 2))

            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            x = width - text_size[0] - 20
            y = 40

            # Background for text
            padding = 8
            cv2.rectangle(frame, (x - padding, y - text_size[1] - padding),
                          (x + text_size[0] + padding, y + padding), (0, 0, 0), -1)
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

        return frame

    def display_frame_with_overlay(self, frame):
        """Display frame with proper aspect ratio and speed overlay"""
        try:
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width, canvas_height = 800, 450

            # Get original frame dimensions
            original_height, original_width = frame.shape[:2]
            original_aspect = original_width / original_height
            canvas_aspect = canvas_width / canvas_height

            # Calculate dimensions maintaining aspect ratio
            if original_aspect > canvas_aspect:
                new_width = canvas_width
                new_height = int(canvas_width / original_aspect)
                x_offset = 0
                y_offset = (canvas_height - new_height) // 2
            else:
                new_height = canvas_height
                new_width = int(canvas_height * original_aspect)
                x_offset = (canvas_width - new_width) // 2
                y_offset = 0

            # Resize frame
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # Add speed overlay
            frame_with_overlay = self.add_speed_overlay(frame_resized)

            # Create black canvas
            canvas_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            canvas_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = frame_with_overlay

            # Convert to RGB and display
            frame_rgb = cv2.cvtColor(canvas_frame, cv2.COLOR_BGR2RGB)

            from PIL import Image, ImageTk
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)

            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
            self.canvas.image = photo

        except Exception as e:
            print(f"Error displaying frame: {e}")

    def update_progress(self):
        """Update progress bar and time display"""
        if self.total_frames > 0:
            progress = (self.current_frame / self.total_frames) * 100
            self.progress_var.set(progress)

            current_seconds = self.current_frame / self.fps
            total_seconds = self.total_frames / self.fps

            current_time = f"{int(current_seconds // 60):02d}:{int(current_seconds % 60):02d}"
            total_time = f"{int(total_seconds // 60):02d}:{int(total_seconds % 60):02d}"

            self.time_var.set(f"{current_time} / {total_time}")

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


class AdaptiveAudioPlayer:
    """Handle audio playback with speed control - FIXED"""

    def __init__(self):
        self.audio_available = self.check_ffplay()
        self.current_audio_file = None
        self.audio_process = None
        self.is_playing = False
        self.current_speed = 1.0
        self.start_position = 0.0
        self.playback_start_time = 0.0
        self.last_position_update = 0.0

    def check_ffplay(self) -> bool:
        """Check if ffplay is available"""
        try:
            result = subprocess.run(['ffplay', '-version'],
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                print("Audio system initialized with speed control")
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

            if os.path.exists(audio_path):
                print(f"Audio file already exists: {audio_path}")
                return audio_path

            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '2',
                '-avoid_negative_ts', 'make_zero',
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

    def start_playback(self, start_position: float = 0.0, speed: float = 1.0):
        """Start audio playback with speed control"""
        if not self.audio_available or not self.current_audio_file:
            return

        try:
            self.stop()
            self.start_position = start_position
            self.current_speed = speed
            self.playback_start_time = time.time()
            self.last_position_update = self.playback_start_time
            self._start_audio_process()
            self.is_playing = True

            print(f"Audio started from {start_position:.2f}s at {speed:.1f}x speed")

        except Exception as e:
            print(f"Audio start error: {e}")

    def _start_audio_process(self):
        """Start audio process with current speed"""
        try:
            if self.audio_process:
                try:
                    self.audio_process.terminate()
                    self.audio_process.wait(timeout=0.5)
                except:
                    try:
                        self.audio_process.kill()
                    except:
                        pass

            # Build ffplay command with speed control
            cmd = [
                'ffplay',
                '-ss', f"{self.start_position:.3f}",
                '-nodisp', '-autoexit', '-v', 'quiet'
            ]

            # Add tempo filter for speed control without pitch change
            if abs(self.current_speed - 1.0) > 0.01:
                tempo_filter = f"atempo={self.current_speed:.3f}"
                cmd.extend(['-af', tempo_filter])

            cmd.append(self.current_audio_file)

            self.audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )

        except Exception as e:
            print(f"Audio process error: {e}")

    def get_current_position(self) -> float:
        """Calculate current audio position"""
        if not self.is_playing:
            return self.start_position

        current_time = time.time()
        elapsed_real_time = current_time - self.playback_start_time
        position = self.start_position + (elapsed_real_time * self.current_speed)
        return position

        # ─── inside class AdaptiveAudioPlayer ─────────────────────────────────────────
    def change_speed_at_position(self, new_speed: float, current_position: float):
            """Restart ffplay at the requested speed, keeping sync."""
            if not self.audio_available or not self.current_audio_file:
                return

            old_speed = self.current_speed
            if abs(new_speed - old_speed) < 0.05:
                return  # no perceptible change

            self.start_position = current_position
            self.current_speed = new_speed
            self.playback_start_time = time.time()

            # Always restart because tempo filter must be rebuilt
            self._start_audio_process()
            print(f"Audio speed {old_speed:.1f} → {new_speed:.1f} at {current_position:.2f}s")

    def pause(self):
        """Pause audio playback"""
        if self.is_playing and self.audio_process:
            try:
                self.audio_process.terminate()
            except:
                pass
            self.is_playing = False

    def resume(self):
        """Resume audio playback"""
        if not self.is_playing:
            self._start_audio_process()
            self.is_playing = True

    def stop(self):
        """Stop audio completely"""
        if self.audio_process:
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self.audio_process.pid), 15)
                else:
                    self.audio_process.terminate()
                self.audio_process.wait(timeout=1)
            except:
                try:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(self.audio_process.pid), 9)
                    else:
                        self.audio_process.kill()
                except:
                    pass
            self.audio_process = None
        self.is_playing = False


# ═══════════════════════════════════════════════════════════════════════════════
# Main EEG Video Player Application - FIXED
# ═══════════════════════════════════════════════════════════════════════════════

class EEGVideoPlayer:
    """Main video player with EEG engagement monitoring and adaptive speed - FIXED"""

    def __init__(self):
        # Initialize communication queue FIRST
        self.gui_queue = queue.Queue()

        # Components
        self.engagement_monitor = EngagementMonitor()
        self.video_processor = MultiBackendVideoProcessor()

        # EEG streaming
        self.eeg_client = None
        self.streaming = False
        self.eeg_thread = None

        # Video state
        self.video_info = None
        self.video_player = None

        # Simulation state
        self.simulation_running = False

        # Speed control - FIXED
        self.last_speed_change = 0
        self.speed_change_cooldown = SPEED_CHANGE_MIN_INTERVAL

        # GUI
        self.setup_gui()

    def setup_gui(self):
        """Setup GUI interface"""
        self.root = tk.Tk()
        self.root.title("EEG-Enhanced Adaptive Speed Video Player - FIXED")
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
        ttk.Label(title_frame, text="🧠⚡ EEG Adaptive Speed Player",
                  font=("Arial", 16, "bold")).pack()

        ttk.Label(title_frame, text="Playback speed adapts to your engagement level",
                  font=("Arial", 10, "italic")).pack()

        # URL input
        url_frame = ttk.LabelFrame(parent, text="YouTube Video")
        url_frame.pack(fill=tk.X, padx=5, pady=5)

        self.url_var = tk.StringVar()
        url_entry = ttk.Entry(url_frame, textvariable=self.url_var, width=40)
        url_entry.pack(fill=tk.X, padx=5, pady=2)

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

        # Manual speed controls
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(pady=5)

        ttk.Label(speed_frame, text="Manual Speed:").pack(side=tk.LEFT)

        for speed in [0.5, 1.0, 1.5, 2.0]:
            btn = ttk.Button(speed_frame, text=f"{speed}x",
                             command=lambda s=speed: self.set_manual_speed(s))
            btn.pack(side=tk.LEFT, padx=1)

        # EEG controls
        eeg_frame = ttk.LabelFrame(parent, text="EEG Monitoring")
        eeg_frame.pack(fill=tk.X, padx=5, pady=5)

        eeg_btn_frame = ttk.Frame(eeg_frame)
        eeg_btn_frame.pack(pady=5)

        self.eeg_btn = ttk.Button(eeg_btn_frame, text="🧠 Start EEG", command=self.start_eeg_monitoring)
        self.eeg_btn.pack(side=tk.LEFT, padx=2)

        self.simulate_btn = ttk.Button(eeg_btn_frame, text="🔄 Simulate", command=self.start_simulation)
        self.simulate_btn.pack(side=tk.LEFT, padx=2)

        # Speed explanation
        ttk.Label(eeg_frame, text="Low engagement → Higher speed (1.5x-2.0x)",
                  font=("Arial", 9), foreground="blue").pack()

        # Status display
        status_frame = ttk.LabelFrame(parent, text="EEG Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        self.engagement_var = tk.StringVar(value="Engagement: --")
        self.baseline_var = tk.StringVar(value="Baseline: Not established")
        self.speed_status_var = tk.StringVar(value="Speed: 1.0x (Normal)")

        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W, padx=5)
        ttk.Label(status_frame, textvariable=self.engagement_var).pack(anchor=tk.W, padx=5)
        ttk.Label(status_frame, textvariable=self.baseline_var).pack(anchor=tk.W, padx=5)
        ttk.Label(status_frame, textvariable=self.speed_status_var, foreground="red").pack(anchor=tk.W, padx=5)

        # Video info
        info_frame = ttk.LabelFrame(parent, text="Video Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_info_text = tk.Text(info_frame, height=6, wrap=tk.WORD, font=("Arial", 9))
        scrollbar1 = ttk.Scrollbar(info_frame, orient="vertical", command=self.video_info_text.yview)
        self.video_info_text.configure(yscrollcommand=scrollbar1.set)

        self.video_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)

        # Speed events log
        speed_frame = ttk.LabelFrame(parent, text="⚡ Speed Changes")
        speed_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.speed_log_text = tk.Text(speed_frame, height=8, wrap=tk.WORD,
                                      bg="#f0f0f0", font=("Arial", 9))
        scrollbar2 = ttk.Scrollbar(speed_frame, orient="vertical", command=self.speed_log_text.yview)
        self.speed_log_text.configure(yscrollcommand=scrollbar2.set)

        self.speed_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_video_player(self, parent):
        """Setup video display area"""
        video_frame = ttk.LabelFrame(parent, text="📺 Adaptive Speed Video Display")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create adaptive speed player
        self.video_player = AdaptiveSpeedVideoPlayer(video_frame)

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
        """Handle GUI update messages - FIXED"""
        msg_type, data = message

        if msg_type == 'engagement_update':
            state = data

            # Get phase information
            phase_info = self.engagement_monitor.get_phase_info()

            if phase_info["phase"] == "discarding":
                self.engagement_var.set(f"🔄 Stabilizing: {phase_info['progress']:.0f}%")
                self.baseline_var.set("Discarding initial noisy data...")
                self.speed_status_var.set("Speed: 1.0x (Stabilizing)")
            elif phase_info["phase"] == "collecting_baseline":
                self.engagement_var.set(f"📊 Baseline: {phase_info['progress']:.0f}%")
                self.baseline_var.set(phase_info["message"])
                self.speed_status_var.set("Speed: 1.0x (Collecting baseline)")
            else:
                # Normal monitoring phase
                self.engagement_var.set(f"Engagement: {state.current_score:.3f}")
                if state.baseline_score > 0:
                    ratio = state.current_score / state.baseline_score
                    self.baseline_var.set(
                        f"Baseline: {state.baseline_score:.3f} (Current: {ratio:.2f}x)")

                # Update speed status
                self.speed_status_var.set(f"Speed: {state.recommended_speed}x ({state.speed_reason})")

                # Apply speed change with cooldown - FIXED
                current_time = time.time()
                if (phase_info["phase"] == "monitoring" and
                        current_time - self.last_speed_change > self.speed_change_cooldown):

                    if self.video_player and abs(state.recommended_speed - self.video_player.playback_speed) > 0.1:
                        self.video_player.set_playback_speed(state.recommended_speed)
                        self.last_speed_change = current_time

                        # Log speed change
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_entry = f"[{timestamp}] Speed: {state.recommended_speed:.1f}x ({state.speed_reason})\n"
                        self.speed_log_text.insert(tk.END, log_entry)
                        self.speed_log_text.see(tk.END)

        elif msg_type == 'status_update':
            self.status_var.set(data)

        elif msg_type == 'speed_event':
            timestamp = datetime.now().strftime("%H:%M:%S")
            event_text = f"[{timestamp}] {data}\n"
            self.speed_log_text.insert(tk.END, event_text)
            self.speed_log_text.see(tk.END)

    def set_manual_speed(self, speed: float):
        """Manually set playback speed"""
        if self.video_player:
            self.video_player.set_playback_speed(speed)
            self.gui_queue.put(('speed_event', f"Manual speed change: {speed}x"))

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
                else:
                    self.status_var.set("Failed to load video")

            else:
                self.status_var.set("yt-dlp not available")
                messagebox.showerror("Error", "yt-dlp is required for video playback")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {e}")
            self.status_var.set("Error loading video")
        finally:
            self.load_btn.config(state=tk.NORMAL)

    def play_video(self):
        """Start video playbook"""
        if self.video_player:
            self.video_player.play()
        self.status_var.set("Playing video with adaptive speed...")

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
        """Run simulated EEG data with varying engagement patterns - IMPROVED"""
        print("Starting EEG simulation for adaptive speed control...")

        # Simulation phases
        phase = "baseline"
        phase_start_time = time.time()
        baseline_duration = BASELINE_DURATION + 5  # Extra time for baseline
        low_engagement_duration = 8  # Periods of low engagement
        high_engagement_duration = 6  # Periods of high engagement

        while self.simulation_running:
            current_time = time.time()
            phase_elapsed = current_time - phase_start_time

            # Generate base EEG data
            ch1_data = np.random.normal(0, 100, SAMPLES_PER_CHUNK)
            ch2_data = np.random.normal(0, 100, SAMPLES_PER_CHUNK)

            # Add engagement patterns based on phase
            if phase == "baseline":
                # Stable moderate engagement for baseline
                engagement_boost = np.random.normal(40, 5, SAMPLES_PER_CHUNK)

                if phase_elapsed > baseline_duration:
                    phase = "varying"
                    phase_start_time = current_time
                    print("Baseline complete - starting engagement variation simulation")

            elif phase == "varying":
                # Cycle between high and low engagement
                cycle_time = phase_elapsed % (low_engagement_duration + high_engagement_duration)

                if cycle_time < low_engagement_duration:
                    # Low engagement period - should trigger speed increase
                    # Vary the depth of engagement drop
                    drop_factor = 0.3 + 0.4 * np.sin(cycle_time * 0.3)  # 30-70% drop, slower oscillation
                    engagement_boost = np.random.normal(-30 * drop_factor, 8, SAMPLES_PER_CHUNK)

                    if cycle_time < 1:  # Log phase change
                        expected_speed = 1.5 if drop_factor < 0.5 else 2.0
                        print(f"Simulating low engagement (drop: {drop_factor:.1f}) - expect {expected_speed}x speed")

                else:
                    # High engagement period - should return to normal speed
                    engagement_boost = np.random.normal(50, 10, SAMPLES_PER_CHUNK)

                    if cycle_time < low_engagement_duration + 1:  # Log phase change
                        print("Simulating high engagement - expect 1.0x speed")

            # Apply engagement pattern
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
            print("✅ FFmpeg available for adaptive audio speed")
            return True
    except:
        pass

    print("❌ FFmpeg not found - audio speed control will be disabled")
    print("   Install with: brew install ffmpeg")
    return False


def main():
    print("EEG-Enhanced Adaptive Speed Video Player - FIXED")
    print("=" * 50)

    # Check available backends
    print("Available backends:")
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

    print("\nAdaptive Speed Control - FIXED:")
    print("• High engagement → Normal speed (1.0x)")
    print("• Low engagement → Faster speed (1.5x-2.0x)")
    print("• Speed adapts smoothly with improved timing")
    print("• Audio continues from current position on speed changes")
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