"""
Real-time Neocore EEG plotter with detailed parsing debug.

pip install bleak matplotlib
python neo_plot.py                 # scan for any QCC5181 / NEOCORE
python neo_plot.py AA:BB:CC:DD:EE  # connect by MAC
python neo_plot.py --test          # enable built-in square-wave
"""

import asyncio
import struct
import sys
from collections import deque
from typing import List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from bleak import BleakScanner, BleakClient

# ─────────────────────────── BLE constants ──────────────────────────
RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # write
TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # notify
TARGET_NAMES = ["QCC5181", "QCC5181-LE", "NEOCORE"]

# Correct feature IDs based on working code
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

# ─────────────────────────── plotting helper ─────────────────────────
class LivePlot:
    WIDTH = 1000

    def __init__(self):
        self.ch1 = deque(maxlen=self.WIDTH)
        self.ch2 = deque(maxlen=self.WIDTH)
        self.data_updated = False

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        self.l1, = self.ax1.plot([], [], 'g-', linewidth=1, label='Channel 1')
        self.l2, = self.ax2.plot([], [], 'r-', linewidth=1, label='Channel 2')

        for ax, title in zip([self.ax1, self.ax2], ['Channel 1', 'Channel 2']):
            ax.set_xlim(0, self.WIDTH)
            ax.set_ylim(-2000000, 2000000)
            ax.grid(True, alpha=0.3)
            ax.set_title(title)
            ax.legend()

        self.ax2.set_xlabel('Samples')
        self.ax1.set_ylabel('Amplitude')
        self.ax2.set_ylabel('Amplitude')

        plt.tight_layout()
        plt.show(block=False)

    def push(self, a: List[int], b: List[int]):
        self.ch1.extend(a)
        self.ch2.extend(b)
        self.data_updated = True

    def update_plot(self):
        if not self.data_updated or len(self.ch1) == 0:
            return

        try:
            x_data = list(range(len(self.ch1)))
            y1_data = list(self.ch1)
            y2_data = list(self.ch2)

            self.l1.set_data(x_data, y1_data)
            self.l2.set_data(x_data, y2_data)

            if y1_data:
                y1_min, y1_max = min(y1_data), max(y1_data)
                y1_range = y1_max - y1_min
                if y1_range > 0:
                    margin = y1_range * 0.1
                    self.ax1.set_ylim(y1_min - margin, y1_max + margin)

            if y2_data:
                y2_min, y2_max = min(y2_data), max(y2_data)
                y2_range = y2_max - y2_min
                if y2_range > 0:
                    margin = y2_range * 0.1
                    self.ax2.set_ylim(y2_min - margin, y2_max + margin)

            if len(x_data) > 0:
                self.ax1.set_xlim(max(0, len(x_data) - self.WIDTH), len(x_data))
                self.ax2.set_xlim(max(0, len(x_data) - self.WIDTH), len(x_data))

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.data_updated = False

        except Exception as e:
            print(f"Error updating plot: {e}")

# ─────────────────────────── headset driver ──────────────────────────
class Headset:
    def __init__(self, mac: Optional[str], use_test: bool):
        self.mac = mac
        self.use_test = use_test
        self.plot = LivePlot()

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
        print(f"Sending: Feature={feature_id}, CID={pdu_specific_id}, Data={data.hex()}, Packet={packet.hex()}")
        await cli.write_gatt_char(RX_UUID, packet, response=False)

    def _parse_data_to_small_buffer(self, data: bytearray):
        """Parse exactly like the working code"""
        if len(data) < 4:
            print(f"Packet too short: {len(data)} bytes")
            return None, None, None, None

        type_val = data[0]
        length = data[1]
        msg_index = int.from_bytes(data[2:3], "little", signed=False)  # Exactly like working code
        value = data[4:length+4]  # Exactly like working code

        print(f"Parsed packet: type={type_val}, length={length}, msg_index={msg_index}, value_len={len(value)}")
        print(f"First 16 bytes of packet: {data[:16].hex()}")
        print(f"First 16 bytes of value: {value[:16].hex() if len(value) >= 16 else value.hex()}")

        return type_val, length, msg_index, value

    def _data_handler_ecg(self, data: bytearray, length: int):
        """Handle EEG data exactly like working code"""
        print(f"EEG handler: data_len={len(data)}, length_param={length}")

        if len(data) % 4 != 0:
            print(f"Invalid EEG data length: {len(data)} (not multiple of 4)")
            return

        num_samples = len(data) // 4
        print(f"Processing {num_samples} total samples")

        try:
            # Unpack as signed 32-bit integers, little endian (same as working code)
            ecg_data = struct.unpack(f"<{num_samples}i", data)

            print(f"First 10 raw samples: {ecg_data[:10]}")

            # Extract channels exactly like working code
            ch1_data = []
            ch2_data = []
            for i in range(0, num_samples, 2):
                ch1_data.append(ecg_data[i])
                if i + 1 < num_samples:
                    ch2_data.append(ecg_data[i + 1])

            print(f"Channel 1 samples: {len(ch1_data)}, first 5: {ch1_data[:5]}")
            print(f"Channel 2 samples: {len(ch2_data)}, first 5: {ch2_data[:5]}")
            print(f"Channel 1 range: {min(ch1_data) if ch1_data else 'N/A'} to {max(ch1_data) if ch1_data else 'N/A'}")
            print(f"Channel 2 range: {min(ch2_data) if ch2_data else 'N/A'} to {max(ch2_data) if ch2_data else 'N/A'}")

            self.plot.push(ch1_data, ch2_data)

        except struct.error as e:
            print(f"Error unpacking EEG data: {e}")
        except Exception as e:
            print(f"Unexpected error in EEG handler: {e}")

    def _on_notify(self, _h, data: bytearray):
        if len(data) < 2:
            return

        # Parse 16-bit BLE header
        command_id = int.from_bytes(data[:2], "big")
        feature_id = command_id >> 9
        pdu_type = (command_id >> 7) & 0x0003
        pdu_specific_id = command_id & 0x007F

        print(f"\n=== New Notification ===")
        print(f"Total packet length: {len(data)}")
        print(f"BLE header: {data[:2].hex()}")
        print(f"Parsed BLE: Feature={feature_id}, PDU={pdu_type}, PID={pdu_specific_id}")

        # Check if this is EEG stream data
        if (feature_id, pdu_type, pdu_specific_id) != (NEOCORE_SENSOR_STREAM_FEATURE_ID, PDU_TYPE_NOTIFICATION, NEOCORE_NOTIFY_ID_EEG_DATA):
            print(f"Ignoring: not EEG data")
            return

        # Parse the payload (everything after 2-byte BLE header)
        payload = data[2:]
        print(f"Payload length: {len(payload)}")
        print(f"Payload first 20 bytes: {payload[:20].hex()}")

        type_val, length, msg_index, value = self._parse_data_to_small_buffer(payload)

        if type_val is None:
            print("Failed to parse payload")
            return

        # Check if this is EEG data (type 2 according to working code)
        if type_val == 2:
            print("Processing as EEG data")
            self._data_handler_ecg(value, length)
        else:
            print(f"Unknown payload type: {type_val}")

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

            print("Plot window opened. Close it to stop streaming.")

            plot_update_counter = 0
            while plt.fignum_exists(self.plot.fig.number):
                plot_update_counter += 1
                if plot_update_counter >= 5:
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
    mac = next((a for a in sys.argv[1:] if not a.startswith("--")), None)

    print("Neocore EEG Plotter - Debug Version")
    print(f"Test mode: {'ON' if test else 'OFF'}")
    if mac:
        print(f"Target MAC: {mac}")
    else:
        print("Will scan for devices")

    try:
        asyncio.run(Headset(mac, test).run())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()