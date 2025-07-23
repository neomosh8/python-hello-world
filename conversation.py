#!/usr/bin/env python3
"""
Realtime voice chatbot -- EEG-driven tone adaptation.

Dependencies
------------
pip install bleak websockets sounddevice numpy scipy certifi
export OPENAI_API_KEY="sk-..."

"""

import os, ssl, asyncio, json, base64, struct, collections, random
import numpy as np
from scipy import signal
import sounddevice as sd
import certifi, websockets
from bleak import BleakScanner, BleakClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# ═════════════════════  CONSTANTS  ═════════════════════
RX_UUID="6e400002-b5a3-f393-e0a9-e50e24dcca9e"
TX_UUID="6e400003-b5a3-f393-e0a9-e50e24dcca9e"
TARGET_NAMES={"QCC5181","QCC5181-LE","NEOCORE"}

SAMPLE_RATE=250      # EEG
PCM_RATE   =24_000   # audio

# ═════════════════════  BLE HELPERS  ═══════════════════
def build_cmd(fid,pid,payload=b""):
    cid=(fid<<9)|(0<<7)|pid
    return cid.to_bytes(2,"big")+payload

def build_stream_cmd(start=True):
    return build_cmd(0x01,0x00,b"\x01" if start else b"\x00")

async def find_device():
    for d in await BleakScanner.discover(6):
        if d.name and any(n in d.name.upper() for n in TARGET_NAMES):
            print(f"EEG headset: {d.name} {d.address}")
            return d.address
    raise RuntimeError("Headset not found")

# ═════════  PACKET PARSE + FILTER  ═════════════════════
def parse_eeg(pkt:bytes):
    if len(pkt)<4 or pkt[0]!=0x02: return [],[]
    n=pkt[1]; samp=pkt[4:4+n]
    if len(samp)%8: return [],[]
    ch1=[struct.unpack("<i",samp[i:i+4])[0] for i in range(0,len(samp),8)]
    ch2=[struct.unpack("<i",samp[i+4:i+8])[0] for i in range(0,len(samp),8)]
    return ch1,ch2

class OnlineFilter:
    def __init__(self,fs):
        ny=fs/2
        self.bp=signal.butter(4,[0.5/ny,40/ny],btype="band",output="sos")
        b,a=signal.iirnotch(60,30,fs)
        self.nt=signal.tf2sos(b,a)
        self.zbp=[signal.sosfilt_zi(self.bp) for _ in range(2)]
        self.znt=[signal.sosfilt_zi(self.nt) for _ in range(2)]
    def __call__(self,x1,x2):
        x1,self.zbp[0]=signal.sosfilt(self.bp,x1,zi=self.zbp[0])
        x2,self.zbp[1]=signal.sosfilt(self.bp,x2,zi=self.zbp[1])
        x1,self.znt[0]=signal.sosfilt(self.nt,x1,zi=self.znt[0])
        x2,self.znt[1]=signal.sosfilt(self.nt,x2,zi=self.znt[1])
        return x1,x2

# ═════════════════  ENGAGEMENT  ═══════════════════════
class Engagement:
    def __init__(self,fs,win=4):
        self.buf=collections.deque(maxlen=fs*win); self.fs=fs
    def add(self,x): self.buf.extend(x)
    def score(self):
        if len(self.buf)<self.buf.maxlen//2: return 0.0
        x=np.asarray(self.buf,np.float32)
        f,p=signal.welch(x,self.fs,nperseg=self.fs)
        def band(lo,hi): idx=(f>=lo)&(f<=hi); return np.trapz(p[idx],f[idx])
        th,al,be=band(4,7),band(8,13),band(14,30)
        return be/(al+th) if al+th else 0.0

# ═══════════════  EEG STREAM  ═════════════════════════
class EEGStreamer:
    def __init__(self,eng):
        self.eng=eng; self.filt=OnlineFilter(SAMPLE_RATE)
    async def run(self):
        mac=await find_device()
        async with BleakClient(mac,timeout=15) as c:
            await c.start_notify(TX_UUID,self._cb)
            await c.write_gatt_char(RX_UUID,build_stream_cmd(True),response=False)
            print("EEG streaming …")
            while c.is_connected: await asyncio.sleep(0.1)
    def _cb(self,_s,d):
        if len(d)<8: return
        ch1,ch2=parse_eeg(d[2:])
        if not ch1: return
        f1,f2=self.filt(np.array(ch1),np.array(ch2))
        self.eng.add((f1+f2)/2)

# ═══════════════  STRATEGY  ═══════════════════════════
class Bandit:
    ARMS={
        "A":"Share a surprising fact in an upbeat tone.",
        "B":"Ask a direct question with curiosity.",
        "C":"Tell a short dramatic story.",
        "D":"Give a friendly compliment in a warm tone.",
        "E":"Issue an energetic call to action."
    }
    def __init__(self,eps=0.2):
        self.eps=eps; self.stats={k:(0,0.) for k in self.ARMS}
    def pick(self):
        if random.random()<self.eps: k=random.choice(list(self.ARMS))
        else: k=max(self.stats,key=lambda k:self.stats[k][1])
        return k,self.ARMS[k]
    def update(self,k,r):
        n,m=self.stats[k]; n+=1; m+=(r-m)/n; self.stats[k]=(n,m)

# ═══════════════  MIC UTIL  ═══════════════════════════
def record_mic(sec=4, rate=PCM_RATE, thr=400):
    try:
        print(f"Recording for {sec} seconds...")
        buf = sd.rec(int(sec * rate), samplerate=rate, channels=1, dtype='int16')
        sd.wait()
        a = buf.flatten()

        print(f"Recorded {len(a)} samples ({len(a) / rate:.2f} seconds)")

        if np.max(np.abs(a)) < thr:
            print("Audio below threshold")
            return None

        # Find audio boundaries but ensure minimum length
        idx = np.where(np.abs(a) > thr)[0]
        if idx.size == 0:
            print("No audio above threshold found")
            return None

        # Ensure minimum 200ms of audio (buffer for API requirement)
        min_samples = int(0.2 * rate)  # 200ms
        start_idx = max(0, idx[0] - int(0.1 * rate))  # Add 100ms padding before
        end_idx = min(len(a), idx[-1] + int(0.1 * rate))  # Add 100ms padding after

        # Ensure we have at least minimum length
        if end_idx - start_idx < min_samples:
            # Extend to minimum length, centered on the audio
            center = (start_idx + end_idx) // 2
            start_idx = max(0, center - min_samples // 2)
            end_idx = min(len(a), start_idx + min_samples)

        result = a[start_idx:end_idx]
        duration_ms = len(result) / rate * 1000
        print(f"Processed audio: {len(result)} samples ({duration_ms:.1f}ms)")

        return result if duration_ms >= 100 else None  # Ensure at least 100ms

    except Exception as e:
        print(f"Mic error: {e}")
        return None


def pcm64(a): return base64.b64encode(a.tobytes()).decode()

# ═══════════  REALTIME CLIENT  ════════════════════════
class RealtimeChat:
    URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    INIT = "You are an engaging, curious voice companion. Greet the user naturally."

    def __init__(self, bandit, eng):
        self.b = bandit;
        self.eng = eng;
        self.mu = 0.;
        self.sig = 1.

    async def run(self):
        # Use environment variable for API key instead of hardcoding
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        hdrs = {"Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1"}
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        try:
            async with websockets.connect(self.URL, additional_headers=hdrs,
                                          ssl=ssl_ctx, ping_interval=None) as ws:
                await self._wait(ws, "session.created")
                print("WebSocket connected successfully")

                # Configure session with correct parameters
                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "instructions": self.INIT,
                        "voice": "echo",
                        "input_audio_format": "pcm16",  # No sample_rate here!
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {
                            "model": "whisper-1"
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 200,
                            "create_response": True
                        },
                        "temperature": 0.8
                    }}))
                await self._wait(ws, "session.updated")
                print("Session configured successfully")

                # Initial greeting
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user",
                             "content": [{"type": "input_text", "text": "Hello, please introduce yourself"}]}
                }))
                await ws.send(json.dumps({
                    "type": "response.create",
                    "response": {"modalities": ["audio", "text"]}
                }))
                await self._play(ws)

                # Main conversation loop
                # In the main conversation loop, add validation:
                while True:
                    k, txt = self.b.pick()
                    aud = record_mic(sec=6, thr=200)  # Record longer, lower threshold

                    if aud is not None:
                        # Validate audio length before sending
                        duration_ms = len(aud) / PCM_RATE * 1000
                        if duration_ms < 100:
                            print(f"Audio too short ({duration_ms:.1f}ms), skipping...")
                            continue

                        # Send audio input using proper format
                        audio_b64 = pcm64(aud)
                        print(f"Sending {len(aud)} samples ({duration_ms:.1f}ms) of audio")

                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": audio_b64
                        }))

                        # Small delay to ensure audio is buffered
                        await asyncio.sleep(0.1)

                        # Commit the audio buffer to trigger processing
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.commit"
                        }))
                        print("Audio sent and committed")

                    else:
                        # Send text input
                        await ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "message", "role": "user",
                                     "content": [{"type": "input_text",
                                                  "text": f"{txt} Please respond with both audio and text."}]}
                        }))
                        await ws.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "instructions": f"You are lively. {txt}",
                                "modalities": ["audio", "text"]
                            }
                        }))
                        print("Text message sent")

                    await self._play(ws)
                    await asyncio.sleep(1)
                    r = self._reward()
                    self.b.update(k, r)

        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket error: {e}")
        except Exception as e:
            print(f"Connection error: {e}")

    async def _wait(self, ws, typ):
        while True:
            try:
                message = await ws.recv()
                event = json.loads(message)
                print(f"Waiting for {typ}, got: {event['type']}")
                if event["type"] == typ:
                    if typ == "session.updated":
                        print(f"Session configuration: {event.get('session', {})}")
                    break
                elif event["type"] == "error":
                    print(f"API Error: {event}")
                    raise Exception(f"API Error: {event}")
            except Exception as e:
                print(f"Error waiting for {typ}: {e}")
                raise

    async def _play(self, ws):
        buf = bytearray()
        audio_received = False

        while True:
            try:
                message = await ws.recv()
                ev = json.loads(message)
                t = ev["type"]
                print(f"EVENT: {t}")

                if t == "response.audio.delta":
                    buf.extend(base64.b64decode(ev["delta"]))
                    audio_received = True

                elif t == "response.audio.done":
                    if buf:
                        print(f"Playing {len(buf)} bytes of audio")
                        try:
                            # Create a copy of the buffer to avoid resize issues
                            audio_bytes = bytes(buf)
                            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).copy()

                            # Play audio in a separate thread to avoid blocking
                            def play_audio():
                                sd.play(audio_data, samplerate=PCM_RATE)
                                sd.wait()

                            # Run audio playback in thread
                            import threading
                            audio_thread = threading.Thread(target=play_audio)
                            audio_thread.start()

                            buf.clear()

                        except Exception as audio_error:
                            print(f"Audio playback error: {audio_error}")
                    else:
                        print("No audio data received")

                elif t == "response.done":
                    if not audio_received:
                        print("Warning: Response completed but no audio was received")
                    print("-- turn complete")
                    break

                elif t == "error":
                    print(f"API Error during playback: {ev}")
                    if "input_audio_buffer" in str(ev):
                        print("Tip: Try speaking longer or check microphone settings")
                    break

                elif t == "input_audio_buffer.speech_started":
                    print("Speech detected")
                elif t == "input_audio_buffer.speech_stopped":
                    print("Speech ended")
                elif t == "input_audio_buffer.committed":
                    print("Audio buffer committed")
                elif t == "response.created":
                    print("Response created")
                elif t == "response.output_item.added":
                    print("Output item added")
                elif t == "response.content_part.added":
                    print("Content part added")

            except Exception as e:
                print(f"Error during playback: {e}")
                break

    def _reward(self):
        e = self.eng.score()
        z = (e - self.mu) / (self.sig or 1)
        self.mu = self.mu * 0.9 + e * 0.1
        self.sig = self.sig * 0.9 + abs(e - self.mu) * 0.1
        print(f"Engagement={e:.3f}, z={z:.2f}")
        return z# ═══════════════  MAIN  ═══════════════════════════════
async def main():
    eng=Engagement(SAMPLE_RATE)
    await asyncio.gather(EEGStreamer(eng).run(),
                         RealtimeChat(Bandit(),eng).run())
if __name__=="__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("bye")
