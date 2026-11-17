"""Ditado Global: atalho de teclado ativa gravacao e digita no campo ativo.

Fluxo:
  1. Aperte o atalho global (padrao: Cmd+Shift+D no macOS, Ctrl+Shift+D em
     outros SO) para iniciar a gravacao. Aperte de novo para parar.
  2. Enquanto grava, cada trecho definitivo transcrito e digitado
     automaticamente no campo de texto que estiver em foco.
  3. Os arquivos de audio ficam em apps/data/transcricoes/ (opcional, pode
     desativar com --no-save).

Requisitos no macOS: conceda permissao de Acessibilidade ao terminal/IDE em
Preferencias do Sistema > Privacidade e Seguranca > Acessibilidade.

Uso:
  python apps/dictation_global.py
  python apps/dictation_global.py --model base --hotkey <cmd>+<shift>+d
  python apps/dictation_global.py --list-devices
  python apps/dictation_global.py --device 2 --model small
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "voice"
if str(VOICE_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_DIR))

import vcommon as vc  # noqa: E402  (mic, VAD, Whisper)
from PySide6.QtCore import (  # noqa: E402
    QObject,
    QThread,
    QTimer,
    Qt,
    QMetaObject,
    Signal,
    Slot,
)
from PySide6.QtWidgets import QApplication  # noqa: E402

import pyautogui  # noqa: E402  (injecao de texto no campo ativo)
from pynput import keyboard  # noqa: E402  (atalho global)

MODELS = ("tiny", "base", "small", "large-v3-turbo")
DEFAULT_MODEL = "base"
DEFAULT_HOTKEY = "<cmd>+<shift>+d" if sys.platform == "darwin" else "<ctrl>+<shift>+d"
SILENCE_END = 0.7
MAX_SEGMENT_S = 15.0
PARTIAL_EVERY = 1.2
POLL_MS = 20
SAVE_DIR = ROOT / "apps" / "data" / "transcricoes"


# ------------------------------------------------------------ transcritor
class Transcriber(vc.Transcriber):
    """Wrapper fino sobre vc.Transcriber com idioma configuravel."""

    def __init__(self, size=DEFAULT_MODEL, lang="pt", device="auto",
                 compute="auto", threads=None, beam=1):
        super().__init__(size, device=device, compute=compute,
                         threads=threads, beam=beam)
        self.lang = lang
        self.beam = beam

    def transcribe(self, wav):
        segs, _ = self.model.transcribe(
            np.asarray(wav, dtype=np.float32),
            language=self.lang or None,
            beam_size=self.beam,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs).strip()


# ------------------------------------------------------------ atalho global
class HotkeyListener(QObject):
    """Escuta um atalho global (mesmo sem o app em foco) e emite toggle."""

    toggle = Signal()
    error = Signal(str)

    def __init__(self, hotkey):
        super().__init__()
        self.hotkey = hotkey
        self._listener = None

    @Slot()
    def start(self):
        try:
            self._listener = keyboard.GlobalHotKeys({self.hotkey: self._on_activate})
            self._listener.start()
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Falha ao registrar atalho global '{self.hotkey}': {e}")

    def _on_activate(self):
        # Chamado na thread do pynput: marshal para a thread Qt.
        QMetaObject.invokeMethod(self, "_emit_toggle", Qt.QueuedConnection)

    @Slot()
    def _emit_toggle(self):
        self.toggle.emit()

    @Slot()
    def stop(self):
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


# ------------------------------------------------------------ motor de audio
class AudioEngine(QObject):
    """Microfone + VAD + Whisper numa QThread. Nao trava a interface."""

    status = Signal(str)
    segment = Signal(str)       # texto definitivo pronto pra injetar
    error = Signal(str)
    level = Signal(float, bool)
    recording_changed = Signal(bool)

    def __init__(self, device=None, model=DEFAULT_MODEL, lang="pt",
                 save_audio=True, out_dir=None, hotkey=""):
        super().__init__()
        self.device = device
        self.model_size = model
        self.lang = lang
        self.save_audio = save_audio
        self.out_dir = out_dir or SAVE_DIR
        self.hotkey = hotkey

        self.recording = False
        self.mic = None
        self.det = None
        self.stt = None
        self.session = None
        self._timer = None
        self._seg_start = None
        self._seg_id = 0
        self._last_t = None
        self._last_partial = 0.0
        self._partial_in_flight = False
        self._job_seq = 0
        self._last_job = None
        self._jobs_done = set()
        self._commits = deque()          # fila de commits pendentes
        self._pending_partial = None     # partial pendente (só o mais recente vale)
        self._busy = False               # Whisper ocupado?
        self._rolling = np.zeros(int(vc.SR * 4.0), np.float32)

    # ------------------------------------------------------------ ciclo
    @Slot()
    def setup(self):
        self.det = vc.UtteranceDetector(
            end_silence=SILENCE_END, max_seconds=MAX_SEGMENT_S
        )
        try:
            self.mic = vc.Mic(self.device)
        except SystemExit as e:
            self.error.emit(str(e))
            self.status.emit("Erro no microfone")
            return
        try:
            self.stt = Transcriber(self.model_size, self.lang)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Falha ao carregar modelo Whisper: {e}")
            self.status.emit("Erro no modelo")
            return
        self.status.emit("Pronto")
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    @Slot()
    def shutdown(self):
        if self._timer is not None:
            self._timer.stop()
        if self.recording:
            self.stop()
        if self.mic is not None:
            self.mic.close()
            self.mic = None

    # ------------------------------------------------------------ toggle
    @Slot()
    def toggle(self):
        if self.recording:
            self.stop()
        else:
            self.start_rec()

    @Slot()
    def start_rec(self):
        if self.recording or self.mic is None or self.stt is None:
            return
        if self.save_audio:
            self.session = Session(self.out_dir, self.model_size, self.lang)
            self.session.open()
        self.det.reset()
        self._seg_start = None
        self._seg_id = 0
        self._last_t = None
        self._last_partial = 0.0
        self._partial_in_flight = False
        self._last_job = None
        self._jobs_done = set()
        self._rec_t0 = time.time()
        self.recording = True
        self.status.emit("Gravando")
        self.recording_changed.emit(True)

    @Slot()
    def stop(self):
        if not self.recording:
            return
        self.recording = False
        self._partial_in_flight = False
        self._commit_pending()
        self._seg_start = None
        if self._last_job is None or self._last_job in self._jobs_done:
            self._finalize_session()
        else:
            self.status.emit("Finalizando...")
            self.recording_changed.emit(False)

    # ------------------------------------------------------------ resultados
    @Slot(int, str)
    def _on_partial_ready(self, seg_id, text):
        if seg_id == self._seg_id and self.recording:
            # partial nao e injetado; so o texto definitivo entra no campo
            pass
        self._partial_in_flight = False

    @Slot(int, str, float, float)
    def _on_segment_ready(self, seg_id, text, start, end):
        if self.save_audio and self.session is not None and text.strip():
            self.session.add(text, start, end)
        # injeta o texto definitivo no campo de texto ativo
        if text.strip():
            self.segment.emit(text)
        if seg_id == self._seg_id:
            pass
        self._partial_in_flight = False

    @Slot(int)
    def _on_commit_done(self, job_id):
        self._jobs_done.add(job_id)
        if (
            not self.recording
            and self._last_job is not None
            and self._last_job in self._jobs_done
            and self.session is not None
        ):
            QTimer.singleShot(0, self._finalize_session)

    # ------------------------------------------------------------ loop
    @Slot()
    def _poll(self):
        blocks = self.mic.blocks() if self.mic is not None else []
        joined = np.concatenate([b for _, b in blocks]) if blocks else None
        if joined is not None:
            n = min(len(joined), len(self._rolling))
            self._rolling = np.concatenate(
                [self._rolling[n:], joined[-n:].astype(np.float32)]
            )
            rms = float(np.sqrt(np.mean(joined**2)))
        else:
            rms = 0.0

        if self.recording and blocks:
            last_t = blocks[-1][0]
            self._last_t = last_t
            for _, b in blocks:
                if self.save_audio and self.session is not None:
                    self.session.audio(b)
                self.det.feed(b)
            if self.det.active:
                if self._seg_start is None:
                    self._seg_start = last_t - len(self.det.buf) * vc.FRAME_S
                    self._seg_id += 1
                    self._partial_in_flight = False
                self._maybe_partial()
            if self.det.done:
                self._submit_commit(
                    self.det.audio, self._seg_start, last_t - self.det.silence
                )
                self.det.reset()
                self._seg_start = None

        self.level.emit(rms, rms > vc.ENERGY_MIN)
        self._drain()

    def _maybe_partial(self):
        if self._partial_in_flight:
            return
        if len(self.det.buf) * vc.FRAME_S < 0.8:
            return
        now = time.time()
        if now - self._last_partial < PARTIAL_EVERY:
            return
        self._last_partial = now
        audio = np.concatenate(self.det.buf).astype(np.float32)
        self._partial_in_flight = True
        self._pending_partial = (self._seg_id, audio)

    def _submit_commit(self, audio, start_abs, end_abs):
        if audio is None or len(audio) < int(0.4 * vc.SR):
            return None
        t0 = self.session.t0 if self.session is not None else self._rec_t0
        start = (
            (start_abs - t0)
            if start_abs is not None
            else (end_abs - t0 - len(audio) / vc.SR)
        )
        end = max(start, end_abs - t0)
        job_id = self._job_seq
        self._job_seq += 1
        self._last_job = job_id
        self._commits.append((job_id, self._seg_id, audio, max(0.0, start), end))
        self._partial_in_flight = False
        return job_id

    def _drain(self):
        """Processa no maximo um item da fila de transcriçao (commit tem prioridade)."""
        if self._busy or self.stt is None:
            return
        if self._commits:
            job_id, seg_id, audio, start, end = self._commits.popleft()
            kind = "commit"
        elif self._pending_partial is not None:
            seg_id, audio = self._pending_partial
            start = end = 0.0
            self._pending_partial = None
            kind = "partial"
        else:
            return

        self._busy = True
        try:
            text = self.stt.transcribe(audio)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Falha na transcricao: {e}")
            text = ""
        self._busy = False

        if kind == "commit":
            self._on_segment_ready(seg_id, text, start, end)
            self._jobs_done.add(job_id)
        else:
            self._on_partial_ready(seg_id, text)

    def _commit_pending(self):
        if self.det.active and self.det.buf:
            audio = np.concatenate(self.det.buf).astype(np.float32)
            end = self._last_t if self._last_t is not None else time.time()
            return self._submit_commit(audio, self._seg_start, end)
        return None

    def _finalize_session(self):
        if self.save_audio and self.session is not None:
            self.session.close()
        self.session = None
        self.det.reset()
        self.status.emit("Pronto")
        self.recording_changed.emit(False)


# ------------------------------------------------------------ sessao de arquivo
class Session:
    """Escreve .wav enquanto grava e fecha .json/.txt ao final."""

    def __init__(self, out_dir, model, lang):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base, n = stamp, 1
        while (self.dir / f"{base}.txt").exists():
            n += 1
            base = f"{stamp}_{n}"
        self.txt_path = self.dir / f"{base}.txt"
        self.wav_path = self.dir / f"{base}.wav"
        self.json_path = self.dir / f"{base}.json"
        self.model, self.lang = model, lang
        self.t0 = time.time()
        self.segments = []
        self._txt = None
        self._wav = None

    def open(self):
        import wave
        self._txt = self.txt_path.open("w", encoding="utf-8")
        self._wav = wave.open(str(self.wav_path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(vc.SR)

    def audio(self, block):
        if self._wav is not None:
            pcm = (np.clip(block, -1, 1) * 32767).astype("<i2")
            self._wav.writeframes(pcm.tobytes())

    def add(self, text, start, end):
        text = text.strip()
        if not text:
            return
        self.segments.append(
            {"inicio_s": round(start, 2), "fim_s": round(end, 2), "texto": text}
        )
        if self._txt is not None:
            self._txt.write(text + "\n")
            self._txt.flush()

    def close(self):
        import wave
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        if self._txt is not None:
            self._txt.close()
            self._txt = None
        data = {
            "inicio": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
            "duracao_s": round(time.time() - self.t0, 2),
            "modelo": self.model,
            "idioma": self.lang,
            "segmentos": self.segments,
        }
        import json
        self.json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "txt": str(self.txt_path),
            "wav": str(self.wav_path),
            "json": str(self.json_path),
        }


# ------------------------------------------------------------ aplicacao
class DictationGlobalApp(QObject):
    """Orquestra hotkey global + audio + injecao de texto."""

    def __init__(self, device=None, model=DEFAULT_MODEL, lang="pt",
                 save_audio=True, hotkey=DEFAULT_HOTKEY):
        super().__init__()
        self.hotkey = hotkey
        self._app = QApplication(sys.argv)
        self._app.setQuitOnLastWindowClosed(False)

        self.engine_thread = QThread(self)
        self.engine = AudioEngine(device, model, lang, save_audio=save_audio)
        self.engine.moveToThread(self.engine_thread)
        self.engine_thread.started.connect(self.engine.setup)

        self.hotkey_thread = QThread(self)
        self.hotkey_listener = HotkeyListener(hotkey)
        self.hotkey_listener.moveToThread(self.hotkey_thread)
        self.hotkey_thread.started.connect(self.hotkey_listener.start)

        # sinais
        self.hotkey_listener.toggle.connect(self.engine.toggle)
        self.engine.segment.connect(self._on_segment)
        self.engine.status.connect(self._on_status)
        self.engine.error.connect(self._on_error)
        self.engine.recording_changed.connect(self._on_recording_changed)

        self.engine_thread.start()
        self.hotkey_thread.start()

        self._recording = False
        print(f"[dictation_global] atalho: {self.hotkey}")
        print("[dictation_global] pressione o atalho para iniciar/parar a gravacao")
        print("[dictation_global] pressione Ctrl+C neste terminal para sair")

    def _on_segment(self, text):
        # Digita o texto transcrito no campo de texto que estiver em foco.
        # pyautogui.write envia teclas para a janela ativa do SO.
        try:
            pyautogui.write(text)
        except Exception as e:  # noqa: BLE001
            print(f"[erro] falha ao injetar texto: {e}", file=sys.stderr)

    def _on_status(self, msg):
        print(f"[dictation_global] {msg}")

    def _on_error(self, msg):
        print(f"[erro] {msg}", file=sys.stderr)

    def _on_recording_changed(self, recording):
        self._recording = recording
        estado = "gravando" if recording else "parado"
        print(f"[dictation_global] {estado}")

    def run(self):
        # Mantem o loop do Qt rodando. Ctrl+C no terminal encerra.
        try:
            self._app.exec()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        QMetaObject.invokeMethod(self.engine, "shutdown", Qt.QueuedConnection)
        QMetaObject.invokeMethod(self.hotkey_listener, "stop", Qt.QueuedConnection)
        self.engine_thread.quit()
        self.engine_thread.wait(5000)
        self.hotkey_thread.quit()
        self.hotkey_thread.wait(5000)


# ------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(
        description="Ditado Global: atalho de teclado para digitar por voz"
    )
    p.add_argument("--device", help="indice ou nome (parcial) do microfone")
    p.add_argument("--list-devices", action="store_true", help="lista microfones e sai")
    p.add_argument(
        "--model", default=DEFAULT_MODEL, choices=MODELS, help="tamanho do Whisper"
    )
    p.add_argument("--lang", default="pt", help="idioma da fala (ex: pt, en)")
    p.add_argument(
        "--hotkey", default=DEFAULT_HOTKEY,
        help=f"atalho global (pynput format, padrao: {DEFAULT_HOTKEY})",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="nao grava arquivos de audio/texto em disco",
    )
    a = p.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    device = a.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    app = DictationGlobalApp(
        device=device,
        model=a.model,
        lang=a.lang,
        save_audio=not a.no_save,
        hotkey=a.hotkey,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
