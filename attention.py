#!/usr/bin/env python3
"""
Neocore EEG Attention/Engagement Measurement System
Advanced cognitive task paradigm with real-time attention analysis
"""

import asyncio
import sys
import struct
import time
import json
import os
from collections import deque
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import threading
from dataclasses import dataclass
import queue

import numpy as np
from scipy import signal
from scipy.stats import ttest_rel
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from bleak import BleakScanner, BleakClient
import openai

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration & Constants
# ═══════════════════════════════════════════════════════════════════════════════

# BLE Configuration (from original code)
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
WINDOW_SIZE = SAMPLE_RATE * 2  # 2-second analysis window

# Experiment Configuration
REST_DURATION = 120  # 2 minutes
TASK_DURATION = 180  # 3 minutes
NUM_CYCLES = 3

# Frequency Bands for Attention Analysis
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 45)
}


@dataclass
class AttentionMetrics:
    """Container for attention measurement results"""
    timestamp: float
    alpha_power: float
    beta_power: float
    theta_power: float
    attention_index: float
    engagement_score: float
    state: str  # 'rest' or 'task'


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI Integration
# ═══════════════════════════════════════════════════════════════════════════════

class AITaskGenerator:
    """Generates comprehension tasks using OpenAI API"""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.difficulty_level = 1

    def generate_passage_and_questions(self, difficulty: int = 1) -> Dict:
        """Generate reading passage with comprehension questions"""

        difficulty_prompts = {
            1: "elementary level, simple vocabulary and concepts",
            2: "middle school level, moderate complexity",
            3: "high school level, complex ideas and vocabulary",
            4: "college level, abstract concepts and analysis"
        }

        prompt = f"""Generate a short reading passage ({150 + difficulty * 50} words) at {difficulty_prompts[difficulty]} 
        followed by 3 multiple choice questions testing comprehension. 

        Format your response as JSON:
        {{
            "passage": "text here",
            "questions": [
                {{
                    "question": "question text",
                    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
                    "correct": "A"
                }}
            ]
        }}

        Topic: science, technology, or current events. Make it engaging and thought-provoking."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            content = response.choices[0].message.content
            # Extract JSON from response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]

            return json.loads(json_str)

        except Exception as e:
            print(f"AI generation error: {e}")
            return self._fallback_task()

    def _fallback_task(self) -> Dict:
        """Fallback task if AI fails"""
        return {
            "passage": """Artificial intelligence has transformed many aspects of modern life. 
            From recommendation systems that suggest what to watch or buy, to autonomous vehicles 
            that can navigate complex traffic situations, AI systems are becoming increasingly 
            sophisticated. However, these advances also raise important questions about privacy, 
            employment, and the role of human judgment in critical decisions.""",
            "questions": [
                {
                    "question": "What is the main topic of this passage?",
                    "options": ["A) Transportation", "B) AI in modern life", "C) Privacy concerns", "D) Employment"],
                    "correct": "B"
                },
                {
                    "question": "According to the passage, AI raises questions about:",
                    "options": ["A) Only privacy", "B) Only employment", "C) Privacy and employment",
                                "D) Traffic navigation"],
                    "correct": "C"
                }
            ]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Attention Analysis Engine
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionAnalyzer:
    """Real-time attention metrics calculation from EEG data"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.fs = sample_rate
        self.window_size = WINDOW_SIZE

        # Data buffers for analysis
        self.ch1_buffer = deque(maxlen=self.window_size)
        self.ch2_buffer = deque(maxlen=self.window_size)

        # Results storage
        self.attention_history = []

    def add_data(self, ch1_data: np.ndarray, ch2_data: np.ndarray):
        """Add new EEG data to analysis buffers"""
        self.ch1_buffer.extend(ch1_data)
        self.ch2_buffer.extend(ch2_data)

    def calculate_attention_metrics(self, state: str) -> Optional[AttentionMetrics]:
        """Calculate current attention metrics"""
        if len(self.ch1_buffer) < self.window_size:
            return None

        # Get current window data
        ch1_data = np.array(list(self.ch1_buffer))
        ch2_data = np.array(list(self.ch2_buffer))

        # Calculate power spectral density
        freqs1, psd1 = signal.welch(ch1_data, self.fs, nperseg=self.fs)
        freqs2, psd2 = signal.welch(ch2_data, self.fs, nperseg=self.fs)

        # Average power across channels
        psd_avg = (psd1 + psd2) / 2

        # Calculate band powers
        band_powers = {}
        for band, (low, high) in BANDS.items():
            idx = np.where((freqs1 >= low) & (freqs1 <= high))[0]
            band_powers[band] = np.mean(psd_avg[idx])

        # Calculate attention metrics
        alpha_power = band_powers['alpha']
        beta_power = band_powers['beta']
        theta_power = band_powers['theta']

        # Attention Index: Beta / (Alpha + Theta)
        attention_index = beta_power / (alpha_power + theta_power + 1e-10)

        # Engagement Score: (Beta + Gamma) / (Alpha + Theta + Delta)
        engagement_score = (band_powers['beta'] + band_powers['gamma']) / \
                           (band_powers['alpha'] + band_powers['theta'] + band_powers['delta'] + 1e-10)

        metrics = AttentionMetrics(
            timestamp=time.time(),
            alpha_power=alpha_power,
            beta_power=beta_power,
            theta_power=theta_power,
            attention_index=attention_index,
            engagement_score=engagement_score,
            state=state
        )

        self.attention_history.append(metrics)
        return metrics

    def get_state_averages(self, state: str) -> Dict:
        """Get average metrics for a specific state"""
        state_metrics = [m for m in self.attention_history if m.state == state]
        if not state_metrics:
            return {}

        return {
            'attention_index': np.mean([m.attention_index for m in state_metrics]),
            'engagement_score': np.mean([m.engagement_score for m in state_metrics]),
            'alpha_power': np.mean([m.alpha_power for m in state_metrics]),
            'beta_power': np.mean([m.beta_power for m in state_metrics]),
            'count': len(state_metrics)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Controller
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentController:
    """Manages the attention measurement experiment"""

    def __init__(self, ai_generator: AITaskGenerator, attention_analyzer: AttentionAnalyzer):
        self.ai_generator = ai_generator
        self.attention_analyzer = attention_analyzer

        # Experiment state
        self.current_state = 'idle'
        self.current_cycle = 0
        self.experiment_running = False
        self.state_start_time = 0

        # Task data
        self.current_task = None
        self.task_responses = []

        # Thread-safe communication
        self.gui_queue = queue.Queue()

    def start_experiment(self):
        """Start the attention measurement experiment"""
        self.experiment_running = True
        self.current_cycle = 0
        self.attention_analyzer.attention_history.clear()
        self._start_rest_period()

    def stop_experiment(self):
        """Stop the experiment"""
        self.experiment_running = False
        self.current_state = 'idle'

    def _start_rest_period(self):
        """Begin rest period"""
        self.current_state = 'rest'
        self.state_start_time = time.time()
        print(f"Starting REST period {self.current_cycle + 1}")

        self.gui_queue.put(('state_change', 'rest', REST_DURATION))

    def _start_task_period(self):
        """Begin task period"""
        self.current_state = 'task'
        self.state_start_time = time.time()

        # Generate new task in separate thread to avoid blocking
        def generate_task():
            task_data = self.ai_generator.generate_passage_and_questions()
            self.current_task = task_data
            self.gui_queue.put(('task_generated', task_data))

        task_thread = threading.Thread(target=generate_task, daemon=True)
        task_thread.start()

        print(f"Starting TASK period {self.current_cycle + 1}")
        self.gui_queue.put(('state_change', 'task', TASK_DURATION))

    def update(self):
        """Update experiment state (call regularly)"""
        if not self.experiment_running:
            return

        elapsed = time.time() - self.state_start_time

        if self.current_state == 'rest' and elapsed >= REST_DURATION:
            self._start_task_period()
        elif self.current_state == 'task' and elapsed >= TASK_DURATION:
            self.current_cycle += 1
            if self.current_cycle < NUM_CYCLES:
                self._start_rest_period()
            else:
                self._finish_experiment()

    def _finish_experiment(self):
        """Complete the experiment"""
        self.experiment_running = False
        self.current_state = 'complete'
        print("Experiment completed!")

        self.gui_queue.put(('state_change', 'complete', 0))

    def record_task_response(self, question_idx: int, selected_answer: str):
        """Record participant's response to task question"""
        if self.current_task and question_idx < len(self.current_task['questions']):
            correct = self.current_task['questions'][question_idx]['correct']
            is_correct = selected_answer == correct

            response = {
                'timestamp': time.time(),
                'cycle': self.current_cycle,
                'question_idx': question_idx,
                'selected': selected_answer,
                'correct': correct,
                'is_correct': is_correct
            }
            self.task_responses.append(response)
            return is_correct
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# GUI Interface (Fixed for main thread)
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionExperimentGUI:
    """Main GUI for attention measurement experiment"""

    def __init__(self, experiment_controller: ExperimentController):
        self.controller = experiment_controller

        # Create GUI
        self.root = tk.Tk()
        self.root.title("EEG Attention Measurement System")
        self.root.geometry("1000x700")

        self.setup_gui()

        # Current task tracking
        self.current_question_idx = 0

        # Start queue processing
        self.process_queue()

    def setup_gui(self):
        """Set up the GUI layout"""
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Experiment tab
        self.exp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.exp_frame, text="Experiment")

        # Real-time metrics tab
        self.metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_frame, text="Real-time Metrics")

        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Analysis Results")

        self.setup_experiment_tab()
        self.setup_metrics_tab()
        self.setup_results_tab()

    def setup_experiment_tab(self):
        """Set up experiment control interface"""
        # Control panel
        control_frame = ttk.LabelFrame(self.exp_frame, text="Experiment Control")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_btn = ttk.Button(control_frame, text="Start Experiment",
                                    command=self.start_experiment)
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop Experiment",
                                   command=self.stop_experiment, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Status display
        self.status_var = tk.StringVar(value="Ready to start")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=20)

        # Progress bar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var)
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.X, expand=True)

        # Task display area
        task_frame = ttk.LabelFrame(self.exp_frame, text="Current Task")
        task_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Passage display
        self.passage_text = scrolledtext.ScrolledText(task_frame, height=8, wrap=tk.WORD)
        self.passage_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Question area
        question_frame = ttk.Frame(task_frame)
        question_frame.pack(fill=tk.X, padx=5, pady=5)

        self.question_label = ttk.Label(question_frame, text="", wraplength=600)
        self.question_label.pack(anchor=tk.W)

        # Answer options
        self.answer_var = tk.StringVar()
        self.option_buttons = []
        for i in range(4):
            btn = ttk.Radiobutton(question_frame, text="", variable=self.answer_var,
                                  value=chr(ord('A') + i))
            btn.pack(anchor=tk.W, pady=2)
            self.option_buttons.append(btn)

        # Submit button
        self.submit_btn = ttk.Button(question_frame, text="Submit Answer",
                                     command=self.submit_answer, state=tk.DISABLED)
        self.submit_btn.pack(pady=10)

    def setup_metrics_tab(self):
        """Set up real-time metrics display"""
        # Current metrics
        current_frame = ttk.LabelFrame(self.metrics_frame, text="Current Metrics")
        current_frame.pack(fill=tk.X, padx=5, pady=5)

        self.attention_var = tk.StringVar(value="Attention Index: --")
        self.engagement_var = tk.StringVar(value="Engagement Score: --")
        self.state_var = tk.StringVar(value="State: Idle")

        ttk.Label(current_frame, textvariable=self.attention_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(current_frame, textvariable=self.engagement_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(current_frame, textvariable=self.state_var).pack(anchor=tk.W, padx=5, pady=2)

    def setup_results_tab(self):
        """Set up results analysis display"""
        # Analysis button
        analyze_btn = ttk.Button(self.results_frame, text="Generate Analysis",
                                 command=self.generate_analysis)
        analyze_btn.pack(pady=10)

        # Results text
        self.results_text = scrolledtext.ScrolledText(self.results_frame, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def process_queue(self):
        """Process messages from experiment controller"""
        try:
            while True:
                message = self.controller.gui_queue.get_nowait()
                msg_type = message[0]

                if msg_type == 'state_change':
                    self.on_state_change(message[1], message[2])
                elif msg_type == 'task_generated':
                    self.on_task_generated(message[1])
                elif msg_type == 'metrics_update':
                    self.update_metrics_display(message[1])

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_queue)

    def start_experiment(self):
        """Start the experiment"""
        self.controller.start_experiment()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_experiment(self):
        """Stop the experiment"""
        self.controller.stop_experiment()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def on_state_change(self, state: str, duration: int):
        """Handle experiment state changes"""
        if state == 'rest':
            self.status_var.set(f"REST period - Relax and breathe normally")
            self.passage_text.delete(1.0, tk.END)
            self.passage_text.insert(tk.END,
                                     "REST PERIOD\n\nPlease sit quietly and relax.\nTry to clear your mind and breathe normally.\nAvoid excessive movement.")
            self.question_label.config(text="")
            for btn in self.option_buttons:
                btn.config(text="", state=tk.DISABLED)
            self.submit_btn.config(state=tk.DISABLED)

        elif state == 'task':
            self.status_var.set(f"TASK period - Read carefully and answer questions")

        elif state == 'complete':
            self.status_var.set("Experiment completed! Check Analysis tab for results.")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def on_task_generated(self, task_data: Dict):
        """Handle new task generation"""
        # Display passage
        self.passage_text.delete(1.0, tk.END)
        self.passage_text.insert(tk.END, "READING TASK\n\n" + task_data['passage'])

        # Start with first question
        self.current_question_idx = 0
        self.show_current_question()

    def show_current_question(self):
        """Display current question"""
        if not self.controller.current_task:
            return

        questions = self.controller.current_task['questions']
        if self.current_question_idx >= len(questions):
            self.question_label.config(text="All questions completed!")
            for btn in self.option_buttons:
                btn.config(state=tk.DISABLED)
            self.submit_btn.config(state=tk.DISABLED)
            return

        question = questions[self.current_question_idx]
        self.question_label.config(text=f"Question {self.current_question_idx + 1}: {question['question']}")

        for i, option in enumerate(question['options']):
            self.option_buttons[i].config(text=option, state=tk.NORMAL)

        self.answer_var.set("")
        self.submit_btn.config(state=tk.NORMAL)

    def submit_answer(self):
        """Submit current answer"""
        selected = self.answer_var.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select an answer before submitting.")
            return

        is_correct = self.controller.record_task_response(self.current_question_idx, selected)

        # Show feedback
        if is_correct:
            messagebox.showinfo("Correct!", "That's the correct answer!")
        else:
            correct_answer = self.controller.current_task['questions'][self.current_question_idx]['correct']
            messagebox.showinfo("Incorrect", f"The correct answer was {correct_answer}")

        # Move to next question
        self.current_question_idx += 1
        self.show_current_question()

    def update_metrics_display(self, metrics: AttentionMetrics):
        """Update real-time metrics display"""
        self.attention_var.set(f"Attention Index: {metrics.attention_index:.3f}")
        self.engagement_var.set(f"Engagement Score: {metrics.engagement_score:.3f}")
        self.state_var.set(f"State: {metrics.state.upper()}")

    def generate_analysis(self):
        """Generate and display analysis results"""
        analyzer = self.controller.attention_analyzer

        if not analyzer.attention_history:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No data available for analysis. Run experiment first.")
            return

        # Calculate statistics
        rest_metrics = analyzer.get_state_averages('rest')
        task_metrics = analyzer.get_state_averages('task')

        # Statistical comparison
        rest_attention = [m.attention_index for m in analyzer.attention_history if m.state == 'rest']
        task_attention = [m.attention_index for m in analyzer.attention_history if m.state == 'task']

        # Performance analysis
        correct_responses = sum(1 for r in self.controller.task_responses if r['is_correct'])
        total_responses = len(self.controller.task_responses)
        accuracy = (correct_responses / total_responses * 100) if total_responses > 0 else 0

        # Generate report
        report = f"""
EEG ATTENTION MEASUREMENT ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXPERIMENT SUMMARY:
- Total cycles completed: {self.controller.current_cycle}
- Total data points: {len(analyzer.attention_history)}
- Task accuracy: {accuracy:.1f}% ({correct_responses}/{total_responses})

ATTENTION METRICS COMPARISON:

REST STATE:
- Average Attention Index: {rest_metrics.get('attention_index', 0):.3f}
- Average Engagement Score: {rest_metrics.get('engagement_score', 0):.3f}
- Data points: {rest_metrics.get('count', 0)}

TASK STATE:
- Average Attention Index: {task_metrics.get('attention_index', 0):.3f}
- Average Engagement Score: {task_metrics.get('engagement_score', 0):.3f}
- Data points: {task_metrics.get('count', 0)}

STATISTICAL ANALYSIS:
"""

        if len(rest_attention) > 5 and len(task_attention) > 5:
            min_len = min(len(rest_attention), len(task_attention))
            t_stat, p_value = ttest_rel(task_attention[:min_len], rest_attention[:min_len])
            report += f"- Paired t-test (Task vs Rest): t = {t_stat:.3f}, p = {p_value:.3f}\n"

            if p_value < 0.05:
                report += "- SIGNIFICANT difference in attention between states!\n"
            else:
                report += "- No significant difference detected\n"

        # Add recommendations
        if task_metrics.get('attention_index', 0) > rest_metrics.get('attention_index', 0):
            report += "\nFINDINGS:\n- Attention increased during cognitive tasks (expected)\n"
        else:
            report += "\nFINDINGS:\n- Attention did not increase as expected during tasks\n"

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, report)

        # Generate plots
        self.plot_results()

    def plot_results(self):
        """Generate analysis plots"""
        analyzer = self.controller.attention_analyzer

        if not analyzer.attention_history:
            return

        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Timeline plot
        times = [(m.timestamp - analyzer.attention_history[0].timestamp) / 60
                 for m in analyzer.attention_history]
        attention_vals = [m.attention_index for m in analyzer.attention_history]
        states = [m.state for m in analyzer.attention_history]

        rest_times = [t for t, s in zip(times, states) if s == 'rest']
        rest_vals = [v for v, s in zip(attention_vals, states) if s == 'rest']
        task_times = [t for t, s in zip(times, states) if s == 'task']
        task_vals = [v for v, s in zip(attention_vals, states) if s == 'task']

        ax1.plot(rest_times, rest_vals, 'bo-', label='Rest', alpha=0.7)
        ax1.plot(task_times, task_vals, 'ro-', label='Task', alpha=0.7)
        ax1.set_title('Attention Index Over Time')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Attention Index')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot comparison
        if rest_vals and task_vals:
            ax2.boxplot([rest_vals, task_vals], labels=['Rest', 'Task'])
            ax2.set_title('Attention Index Distribution')
            ax2.set_ylabel('Attention Index')

        # Band power comparison
        rest_metrics = analyzer.get_state_averages('rest')
        task_metrics = analyzer.get_state_averages('task')

        if rest_metrics and task_metrics:
            bands = ['Alpha', 'Beta']
            rest_powers = [rest_metrics.get('alpha_power', 0), rest_metrics.get('beta_power', 0)]
            task_powers = [task_metrics.get('alpha_power', 0), task_metrics.get('beta_power', 0)]

            x = np.arange(len(bands))
            width = 0.35

            ax3.bar(x - width / 2, rest_powers, width, label='Rest', alpha=0.7)
            ax3.bar(x + width / 2, task_powers, width, label='Task', alpha=0.7)
            ax3.set_title('EEG Band Power Comparison')
            ax3.set_ylabel('Power (µV²)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(bands)
            ax3.legend()

        # Performance correlation
        if self.controller.task_responses:
            response_times = [r['timestamp'] - analyzer.attention_history[0].timestamp
                              for r in self.controller.task_responses]
            accuracies = [1 if r['is_correct'] else 0 for r in self.controller.task_responses]

            ax4.scatter(response_times, accuracies, alpha=0.7)
            ax4.set_title('Task Performance Over Time')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Accuracy (1=Correct, 0=Incorrect)')
            ax4.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.show()

    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
# EEG Data Processing (from original code)
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
# Main Experiment System (Fixed threading)
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionMeasurementSystem:
    """Main system integrating EEG streaming with attention analysis"""

    def __init__(self, openai_api_key: str):
        # Initialize components
        self.ai_generator = AITaskGenerator(openai_api_key)
        self.attention_analyzer = AttentionAnalyzer()
        self.experiment_controller = ExperimentController(self.ai_generator, self.attention_analyzer)

        # GUI setup (will run in main thread)
        self.gui = AttentionExperimentGUI(self.experiment_controller)

        # EEG streaming
        self.eeg_client = None
        self.streaming = False
        self.eeg_thread = None

    def notification_handler(self, sender: int, data: bytearray):
        """Handle incoming EEG data"""
        try:
            if len(data) < 6:
                return

            ch1_samples, ch2_samples = parse_eeg_packet(data[2:])

            # Add to attention analyzer
            self.attention_analyzer.add_data(np.array(ch1_samples), np.array(ch2_samples))

            # Calculate current metrics if in experiment
            if self.experiment_controller.experiment_running:
                metrics = self.attention_analyzer.calculate_attention_metrics(
                    self.experiment_controller.current_state)

                if metrics:
                    # Send metrics to GUI via queue
                    self.experiment_controller.gui_queue.put(('metrics_update', metrics))

        except Exception as e:
            print(f"EEG data processing error: {e}")

    async def start_eeg_streaming(self, device_address: str):
        """Start EEG data streaming"""
        print(f"Connecting to EEG device: {device_address}")

        self.eeg_client = BleakClient(device_address, timeout=20.0)
        await self.eeg_client.connect()

        if not self.eeg_client.is_connected:
            raise RuntimeError("Failed to connect to EEG device")

        print("EEG device connected! Setting up data stream...")

        try:
            await self.eeg_client.request_mtu(247)
        except:
            pass

        await self.eeg_client.start_notify(TX_UUID, self.notification_handler)

        start_cmd = build_stream_command(True)
        await self.eeg_client.write_gatt_char(RX_UUID, start_cmd, response=False)

        self.streaming = True
        print("EEG streaming started!")

        # Update experiment while streaming
        try:
            while self.streaming and self.eeg_client.is_connected:
                self.experiment_controller.update()
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"EEG streaming error: {e}")
        finally:
            if self.eeg_client and self.eeg_client.is_connected:
                try:
                    stop_cmd = build_stream_command(False)
                    await self.eeg_client.write_gatt_char(RX_UUID, stop_cmd, response=False)
                    await self.eeg_client.stop_notify(TX_UUID)
                    await self.eeg_client.disconnect()
                except:
                    pass

    def stop_eeg_streaming(self):
        """Stop EEG streaming"""
        self.streaming = False

    def run_eeg_thread(self):
        """Run EEG streaming in separate thread"""
        try:
            # Find and connect to device
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            device_address = loop.run_until_complete(find_device())
            loop.run_until_complete(self.start_eeg_streaming(device_address))

        except Exception as e:
            print(f"EEG thread error: {e}")
        finally:
            print("EEG thread stopped")

    def run(self):
        """Run the complete system with proper threading"""
        # Start EEG streaming in background thread
        self.eeg_thread = threading.Thread(target=self.run_eeg_thread, daemon=True)
        self.eeg_thread.start()

        # Run GUI in main thread
        try:
            self.gui.run()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.stop_eeg_streaming()
            print("System shutdown complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("EEG Attention Measurement System")
    print("=" * 50)

    # Get OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        openai_api_key = input("Enter your OpenAI API key: ").strip()
        if not openai_api_key:
            print("OpenAI API key required for task generation!")
            return 1

    try:
        # Initialize and run system
        system = AttentionMeasurementSystem(openai_api_key)
        system.run()

    except Exception as e:
        print(f"System error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nSystem stopped by user.")
        sys.exit(0)