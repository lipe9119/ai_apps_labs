"""Utilitarios de voz: microfone, VAD, deteccao de enunciados, embeddings de locutor (ECAPA),
transcricao (Whisper), desafio de digitos, sincronia labial e UI minima em OpenCV.

Imports pesados (torch, sounddevice, faster-whisper, webrtcvad) sao feitos so quando usados.
"""
import argparse
import json
import os
import queue
import random
import re
import time
import unicodedata
from collections import deque
from pathlib import Path

import cv2
import numpy as np

SR = 16000
FRAME = 480                       # 30 ms a 16 kHz (tamanho exigido pelo webrtcvad)
FRAME_S = FRAME / SR

VOICE_FILE = Path(__file__).parent / "data" / "voices.json"
MODEL_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------- ajustes
VOICE_THRESHOLD = 0.45            # similaridade (cosseno) minima p/ aceitar o locutor. Maior = mais rigido
MIN_SPEECH_S = 1.5                # fala minima (s) para cadastro
ENERGY_MIN = 0.004                # RMS minimo para um bloco contar como fala
WHISPER_MODEL = "base"            # tiny (mais rapido) | base | small (mais preciso)
N_DIGITS = 4                      # tamanho do desafio de digitos
MAX_DIGIT_ERRORS = 0              # erros tolerados no desafio (distancia de edicao)
LIPSYNC_MIN_CORR = 0.25           # correlacao minima entre abertura da boca e volume da voz
MIN_MAR_RANGE = 0.04              # variacao minima da abertura da boca (rosto parado = reprova)

# ---------------------------------------------------------------- cores (BGR) e canvas
GREEN = (60, 180, 60)
RED = (50, 50, 220)
YELLOW = (0, 170, 220)
GRAY = (90, 90, 90)
BG = (28, 28, 28)
FONT = cv2.FONT_HERSHEY_SIMPLEX
W, H = 640, 360


# ---------------------------------------------------------------- argumentos
def parse_args(desc=""):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--device", help="indice ou parte do nome do microfone")
    p.add_argument("--list-devices", action="store_true", help="lista os dispositivos de audio e sai")
    a = p.parse_args()
    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        raise SystemExit
    dev = a.device
    if dev is not None and dev.isdigit():
        dev = int(dev)
    return dev


# ---------------------------------------------------------------- microfone
class Mic:
    """Captura continua a 16 kHz mono. Cada bloco chega na fila como (timestamp, array float32 de 480 amostras)."""

    def __init__(self, device=None):
        import sounddevice as sd
        self.q = queue.Queue()
        try:
            self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME,
                                         device=device, callback=self._cb)
            self.stream.start()
        except Exception as e:
            raise SystemExit(f"Nao consegui abrir o microfone: {e}\nTente: python <script> --list-devices  e depois  --device N")

    def _cb(self, indata, frames, t, status):
        self.q.put((time.time(), indata[:, 0].copy()))

    def blocks(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def drain(self):
        self.blocks()

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class UtteranceDetector:
    """Junta blocos em 'enunciados' (uma frase). Comeca quando ha fala sustentada e termina apos silencio."""

    def __init__(self, aggressiveness=2, start_frames=5, end_silence=0.8, max_seconds=10.0, preroll=0.3):
        self.aggr, self.start_frames = aggressiveness, start_frames
        self.end_silence, self.max_seconds, self.preroll = end_silence, max_seconds, preroll
        self.vad = None
        self.reset()

    def reset(self):
        self.pre = deque(maxlen=max(1, int(self.preroll / FRAME_S)))
        self.buf = []
        self.active = False
        self.done = False
        self.run = 0
        self.silence = 0.0
        self.voiced = 0.0          # segundos de fala de fato
        self.audio = None

    def is_speech(self, block):
        if len(block) != FRAME:
            return False
        if float(np.sqrt(np.mean(block ** 2))) < ENERGY_MIN:
            return False
        if self.vad is None:
            import webrtcvad
            self.vad = webrtcvad.Vad(self.aggr)
        pcm = (np.clip(block, -1, 1) * 32767).astype(np.int16).tobytes()
        return self.vad.is_speech(pcm, SR)

    def feed(self, block):
        """Processa um bloco. Retorna True se este bloco e fala."""
        if self.done:
            return False
        sp = self.is_speech(block)
        if not self.active:
            self.pre.append(block)
            self.run = self.run + 1 if sp else 0
            if self.run >= self.start_frames:
                self.active = True
                self.buf = list(self.pre)
                self.silence = 0.0
                self.voiced = self.run * FRAME_S
        else:
            self.buf.append(block)
            if sp:
                self.voiced += FRAME_S
                self.silence = 0.0
            else:
                self.silence += FRAME_S
            if self.silence >= self.end_silence or len(self.buf) * FRAME_S >= self.max_seconds:
                self.done = True
                self.audio = np.concatenate(self.buf).astype(np.float32)
        return sp


# ---------------------------------------------------------------- locutor (ECAPA-TDNN via SpeechBrain)
class SpeakerEncoder:
    """Transforma um trecho de voz em um vetor de 192 numeros (normalizado). Mesma voz = vetores proximos."""

    def __init__(self):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:                                   # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        torch.set_num_threads(2)
        self.torch = torch
        kwargs = dict(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODEL_DIR / "ecapa"),
                      run_opts={"device": "cpu"})
        if os.name == "nt":                                   # evita erro de symlink no Windows
            try:
                from speechbrain.utils.fetching import LocalStrategy
                kwargs["local_strategy"] = LocalStrategy.COPY
            except Exception:
                pass
        try:
            self.model = EncoderClassifier.from_hparams(**kwargs)
        except TypeError:
            kwargs.pop("local_strategy", None)
            self.model = EncoderClassifier.from_hparams(**kwargs)

    def embed(self, wav):
        sig = self.torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
        with self.torch.no_grad():
            e = self.model.encode_batch(sig).squeeze().cpu().numpy().astype(np.float64)
        return e / (np.linalg.norm(e) + 1e-9)


# ---------------------------------------------------------------- banco de vozes
def load_voices():
    """{nome: array (n_amostras, 192)}"""
    if not VOICE_FILE.exists():
        return {}
    raw = json.loads(VOICE_FILE.read_text(encoding="utf-8"))
    return {n: np.array([s["embedding"] for s in samples]) for n, samples in raw.items()}


def save_voice(name, samples):
    """samples: lista de (id_da_frase, embedding). Substitui cadastro anterior com o mesmo nome."""
    raw = json.loads(VOICE_FILE.read_text(encoding="utf-8")) if VOICE_FILE.exists() else {}
    raw[name] = [{"phrase": p, "embedding": e.tolist()} for p, e in samples]
    VOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    VOICE_FILE.write_text(json.dumps(raw), encoding="utf-8")


def score_voice(emb, embs):
    """Media das 3 maiores similaridades (cosseno) com as amostras da pessoa."""
    sims = np.sort(embs @ emb)
    return float(sims[-min(3, len(sims)):].mean())


def identify_voice(emb, db):
    """Retorna (nome ou None, melhor score, {nome: score})."""
    scores = {n: score_voice(emb, e) for n, e in db.items()}
    best = max(scores, key=scores.get)
    return (best if scores[best] >= VOICE_THRESHOLD else None), scores[best], scores


# ---------------------------------------------------------------- fala -> texto e desafio de digitos
class Transcriber:
    def __init__(self, size=WHISPER_MODEL):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=2,
                                  download_root=str(MODEL_DIR / "whisper"))

    def transcribe(self, wav):
        segs, _ = self.model.transcribe(np.asarray(wav, dtype=np.float32), language="pt", beam_size=1,
                                        condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()


_WORDS = {"zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4, "cinco": 5,
          "seis": 6, "meia": 6, "sete": 7, "oito": 8, "nove": 9}


def ascii_only(s):
    """O OpenCV so desenha ASCII: remove acentos."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def digits_from_text(text):
    """'quatro, sete, 2 e nove' / '4-7-2-9' / '4729' -> [4, 7, 2, 9]"""
    out = []
    for tok in re.findall(r"\d|[a-z]+", ascii_only(text).lower()):
        if tok.isdigit():
            out.append(int(tok))
        elif tok in _WORDS:
            out.append(_WORDS[tok])
    return out


def digit_errors(expected, heard):
    """Distancia de edicao entre duas listas de digitos."""
    n, m = len(expected), len(heard)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (expected[i - 1] != heard[j - 1]))
            prev = cur
    return d[m]


def random_digits(n=N_DIGITS):
    d = [random.randrange(10)]
    while len(d) < n:
        x = random.randrange(10)
        if x != d[-1]:                 # sem repeticao seguida (facilita a fala e a transcricao)
            d.append(x)
    return d


# ---------------------------------------------------------------- sincronia labial (rosto + voz)
def lipsync_score(mar_ts, mar_vals, aud_ts, aud_rms):
    """Correlacao (maxima com folga de +-300 ms) entre abertura da boca e volume da voz.
    Retorna (correlacao, variacao_da_boca). Voz tocando num alto-falante com o rosto parado => variacao ~0."""
    if len(mar_ts) < 10 or len(aud_ts) < 10:
        return 0.0, 0.0
    mar_vals = np.asarray(mar_vals, dtype=float)
    mar_range = float(np.percentile(mar_vals, 95) - np.percentile(mar_vals, 5))
    t0, t1 = max(mar_ts[0], aud_ts[0]), min(mar_ts[-1], aud_ts[-1])
    if t1 - t0 < 0.8:
        return 0.0, mar_range
    grid = np.arange(t0, t1, 0.05)                                     # 20 Hz
    m = np.interp(grid, mar_ts, mar_vals)
    a = np.interp(grid, aud_ts, np.log10(np.asarray(aud_rms, dtype=float) + 1e-4))
    m, a = m - m.mean(), a - a.mean()
    if m.std() < 1e-6 or a.std() < 1e-6:
        return 0.0, mar_range
    m, a = m / m.std(), a / a.std()
    best = -1.0
    for lag in range(-6, 7):
        if lag >= 0:
            x, y = m[lag:], a[:len(a) - lag]
        else:
            x, y = m[:lag], a[-lag:]
        if len(x) >= 8:
            best = max(best, float(np.mean(x * y)))
    return best, mar_range


# ---------------------------------------------------------------- UI minima (OpenCV so aceita ASCII!)
def new_canvas():
    return np.full((H, W, 3), BG, np.uint8)


def put_text(img, text, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def center_text(img, text, y, color=(255, 255, 255), scale=1.0, thick=2):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    put_text(img, text, (max(8, (img.shape[1] - tw) // 2), y), color, scale, thick)


def banner(img, text, color=GRAY, sub=None):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 48), color, -1)
    put_text(img, text, (12, 33), scale=0.75)
    if sub:
        put_text(img, sub, (12, h - 14), scale=0.6)


class Rolling:
    """Ultimos N segundos de audio, para desenhar a onda."""

    def __init__(self, seconds=2.0):
        self.buf = np.zeros(int(SR * seconds) // W * W, np.float32)

    def push(self, block):
        n = len(block)
        self.buf = np.concatenate([self.buf[n:], block.astype(np.float32)])


def draw_wave(img, samples, top, height, color):
    cols = np.abs(samples.reshape(W, -1)).max(axis=1)
    mid = top + height // 2
    for x, v in enumerate(cols):
        a = int(min(1.0, v * 4) * height / 2)
        cv2.line(img, (x, mid - a), (x, mid + a), color, 1)


def level_frac(rms):
    return float(np.clip((20 * np.log10(rms + 1e-6) + 60) / 60, 0, 1))


def draw_meter(img, rms, x, y, w, h, active):
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 80, 80), 1)
    cv2.rectangle(img, (x + 1, y + 1), (x + 1 + int((w - 2) * level_frac(rms)), y + h - 1),
                  GREEN if active else (140, 140, 140), -1)


def edit_text(text, key, maxlen=24):
    if key in (8, 127):
        return text[:-1]
    if 32 <= key <= 126 and len(text) < maxlen:
        return text + chr(key)
    return text


def show(win, img):
    cv2.imshow(win, img)
    cv2.waitKey(1)
