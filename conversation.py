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
def record_mic(sec=4,rate=PCM_RATE,thr=400):
    try:
        buf=sd.rec(int(sec*rate),samplerate=rate,channels=1,dtype='int16')
        sd.wait(); a=buf.flatten()
        if np.max(np.abs(a))<thr: return None
        idx=np.where(np.abs(a)>thr)[0]
        return a[idx[0]:idx[-1]+1] if idx.size else None
    except Exception as e:
        print("Mic error:",e); return None
def pcm64(a): return base64.b64encode(a.tobytes()).decode()

# ═══════════  REALTIME CLIENT  ════════════════════════
class RealtimeChat:
    URL="wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2025-06-03"
    INIT="You are an engaging, curious voice companion. Greet the user naturally."
    def __init__(self,bandit,eng):
        self.b=bandit; self.eng=eng; self.mu=0.; self.sig=1.
    async def run(self):
        hdrs={"Authorization":f"Bearer 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",
              "OpenAI-Beta":"realtime=v1"}
        ssl_ctx=ssl.create_default_context(cafile=certifi.where())
        async with websockets.connect(self.URL,additional_headers=hdrs,
                                      ssl=ssl_ctx,ping_interval=None) as ws:
            await self._wait(ws,"session.created")
            # session config & turn-detect off
            await ws.send(json.dumps({
                "type":"session.update",
                "session":{
                    "instructions":self.INIT,
                    "voice":"echo",
                    "turn_detection":{"create_response":False},
                    "input_audio_format":{"type":"pcm","sample_rate":PCM_RATE,"channels":1},
                    "output_audio_format":{"type":"pcm","sample_rate":PCM_RATE,"channels":1}
                }}))
            await self._wait(ws,"session.updated")
            # initial nudge
            await ws.send(json.dumps({
                "type":"conversation.item.create",
                "item":{"type":"message","role":"user",
                        "content":[{"type":"input_text","text":"👋"}]}
            }))
            await ws.send(json.dumps({
                "type":"response.create",
                "response":{"modalities":["audio","text"]}
            }))
            await self._play(ws)
            while True:
                k,txt=self.b.pick()
                aud=record_mic()
                if aud is not None:
                    await ws.send(json.dumps({
                        "type":"conversation.item.create",
                        "item":{"type":"message","role":"user",
                                "content":[{"type":"input_audio","audio":pcm64(aud)}]}}))
                else:
                    await ws.send(json.dumps({
                        "type":"conversation.item.create",
                        "item":{"type":"message","role":"user",
                                "content":[{"type":"input_text","text":"👍"}]}}))
                await ws.send(json.dumps({
                    "type":"response.create",
                    "response":{
                        "instructions":f"You are lively. {txt}",
                        "modalities":["audio","text"]}}))
                await self._play(ws)
                await asyncio.sleep(1)
                r=self._reward(); self.b.update(k,r)
    # helpers
    async def _wait(self,ws,typ):
        while json.loads(await ws.recv())["type"]!=typ: pass
    async def _play(self,ws):
        buf=bytearray()
        while True:
            ev=json.loads(await ws.recv()); t=ev["type"]
            print("EVENT:",t)                       # debug
            if t=="response.audio.delta":
                buf.extend(base64.b64decode(ev["delta"]))
            elif t=="response.audio.done":
                sd.play(np.frombuffer(buf,np.int16),samplerate=PCM_RATE)
                sd.wait(); buf.clear()
            elif t=="response.done":
                print("-- turn complete")
                break
    def _reward(self):
        e=self.eng.score()
        z=(e-self.mu)/(self.sig or 1)
        self.mu=self.mu*0.9+e*0.1
        self.sig=self.sig*0.9+abs(e-self.mu)*0.1
        print(f"Engagement={e:.3f}, z={z:.2f}")
        return z

# ═══════════════  MAIN  ═══════════════════════════════
async def main():
    eng=Engagement(SAMPLE_RATE)
    await asyncio.gather(EEGStreamer(eng).run(),
                         RealtimeChat(Bandit(),eng).run())
if __name__=="__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("bye")
