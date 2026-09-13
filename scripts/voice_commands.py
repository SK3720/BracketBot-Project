# /// script
# dependencies = ["faster-whisper", "numpy<2"]
# ///
"""Shared real speech-recognition helpers, built on a local Whisper model
(faster-whisper, CPU, int8) instead of the RMS-only voice-activity detection
used earlier -- this actually transcribes words so callers can match against
specific phrases (stop/continue/thank-you) rather than just "some sound
happened".

Segmentation is RMS-gated: a segment starts once audio RMS crosses
START_RMS, and ends once RMS has stayed below STOP_RMS for SILENCE_HANG_S
(hysteresis so a brief dip mid-sentence doesn't cut it short). Each completed
segment is transcribed once. This is keyword/substring matching against a
handful of phrasings, not real language understanding -- good enough for a
short fixed set of commands, tune the *_PHRASES lists below against what
listen_transcribe.py (robot mic) or listen_transcribe_local.py (laptop mic)
actually shows you it heard.

Three entry points, two of which share the same gating/transcription core
(_run_segment_loop) and only differ in where raw audio chunks come from:
  - listen_continuous           -- robot mic, via bbos's mic.audio channel.
  - listen_continuous_local_mic -- this machine's own mic, via sounddevice.
bbos is only imported inside listen_continuous, lazily, so this module (and
the local-mic path) works fine on a laptop that doesn't have bbos installed.

The third, run_command_server / send_text_to_robot, is for when the robot's
own mic is too finicky about distance/volume to rely on for a demo: the
laptop captures + transcribes locally (listen_continuous_local_mic) and POSTs
each recognized phrase over the network to run_command_server, which just
dispatches already-recognized text -- the robot does no audio capture or
Whisper inference at all in that mode. See send_voice_commands.py (laptop
side) and full_delivery.py's start_voice_thread (robot side).
"""
import queue
import time

import numpy as np

START_RMS = 1200.0
STOP_RMS = 800.0
SILENCE_HANG_S = 0.6
MAX_SEGMENT_S = 8.0
MIN_SEGMENT_S = 0.3

THANKS_PHRASES = ["thank you", "thanks", "appreciate it", "appreciated", "thank u"]
FOOD_REQUEST_PHRASES = ["hey bracket, can i have my food", "hey bracket can i have my food",
                         "can i have my food"]

_model = None


def get_model(size="tiny.en"):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"Loading Whisper model ({size})...")
        t0 = time.time()
        _model = WhisperModel(size, device="cpu", compute_type="int8")
        print(f"  model loaded in {time.time() - t0:.1f}s")
    return _model


def transcribe_clip(clip_1d_int16, model=None):
    model = model or get_model()
    audio_f32 = clip_1d_int16.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_f32, language="en", beam_size=1)
    return " ".join(seg.text for seg in segments).strip()


def matches_any(text, phrases):
    text_lower = text.lower()
    return any(p in text_lower for p in phrases)


def describe_tags(text):
    tags = []
    if matches_any(text, THANKS_PHRASES):
        tags.append("THANKS")
    if matches_any(text, FOOD_REQUEST_PHRASES):
        tags.append("FOOD_REQUEST")
    return tags


def _rms(audio):
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _run_segment_loop(audio_chunks, on_segment, model, on_rms=None):
    """Shared RMS-gating + transcription loop, driven by any iterable of raw
    audio chunks (any shape -- flattened here; e.g. mic.audio hands back
    (N, 1) arrays, not flat 1D, which is exactly what silently broke this
    the first time around: no error, just an empty transcript every time)."""
    buf = []
    in_speech = False
    seg_start = None
    silence_start = None
    for audio in audio_chunks:
        audio = audio.reshape(-1)
        rms = _rms(audio)
        if on_rms is not None:
            on_rms(rms)
        now = time.monotonic()

        if not in_speech:
            if rms > START_RMS:
                in_speech = True
                seg_start = now
                silence_start = None
                buf = [audio]
            continue

        buf.append(audio)
        if rms < STOP_RMS:
            if silence_start is None:
                silence_start = now
            elif now - silence_start > SILENCE_HANG_S:
                in_speech = False
        else:
            silence_start = None
        if now - seg_start > MAX_SEGMENT_S:
            in_speech = False

        if not in_speech:
            duration = now - seg_start
            clip = np.concatenate(buf)
            buf = []
            if duration >= MIN_SEGMENT_S:
                try:
                    text = transcribe_clip(clip, model)
                except Exception as e:
                    print(f"    (segment: {duration:.1f}s, shape={clip.shape} dtype={clip.dtype} "
                          f"-- transcribe FAILED: {e!r})")
                    text = ""
                else:
                    print(f"    (segment: {duration:.1f}s, shape={clip.shape} dtype={clip.dtype} "
                          f"-> transcribed: \"{text}\")")
                if text:
                    on_segment(text)


def listen_continuous(on_segment, stop_event=None, model=None, on_rms=None):
    """Robot mic, via bbos's mic.audio channel. Blocking -- run in its own
    thread. See module docstring for on_rms/stop_event."""
    from bbos import Reader
    model = model or get_model()

    def chunks():
        with Reader("mic.audio") as r:
            while stop_event is None or not stop_event.is_set():
                if not r.ready():
                    time.sleep(0.01)
                    continue
                yield r.data["audio"]

    _run_segment_loop(chunks(), on_segment, model, on_rms)


VOICE_COMMAND_PORT = 8015


def run_command_server(on_segment, port=VOICE_COMMAND_PORT):
    """Robot side of the laptop-mic setup: receives already-transcribed text
    over HTTP (POST /voice {"text": "..."}) and dispatches it through
    on_segment exactly like listen_continuous would -- no mic, no Whisper,
    nothing audio-related on this machine at all. Blocking -- run in its own
    thread."""
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                text = json.loads(body).get("text", "")
            except Exception:
                text = ""
            if text:
                on_segment(text)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, format, *args):
            pass

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def send_text_to_robot(url, text, timeout=2.0):
    """Laptop side: POSTs one recognized phrase to run_command_server."""
    import json
    import urllib.request
    req = urllib.request.Request(
        url, data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()


def listen_continuous_local_mic(on_segment, stop_event=None, model=None, on_rms=None,
                                 samplerate=16000, blocksize=1600, device=None):
    """This machine's own mic, via sounddevice -- for fast local iteration
    on phrase-matching/thresholds without needing the robot at all. Same
    gating/transcription core as the robot path (_run_segment_loop), so
    whatever works here works there too."""
    import sounddevice as sd
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    model = model or get_model()

    def chunks():
        with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16",
                             blocksize=blocksize, device=device, callback=callback):
            while stop_event is None or not stop_event.is_set():
                try:
                    yield q.get(timeout=0.5)
                except queue.Empty:
                    continue

    _run_segment_loop(chunks(), on_segment, model, on_rms)
