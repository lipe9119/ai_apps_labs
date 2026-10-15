"""Ditado: transcreve sua fala ao vivo e salva .txt + .wav + .json.

Fluxo:
  1. Aperte ESPACO/ENTER (ou clique em GRAVAR) e comece a falar.
  2. O texto aparece na tela enquanto voce fala: o trecho em andamento
     fica em italico/cinza (provisorio) e vira definitivo quando voce
     faz uma pausa. Cada trecho definitivo ja e salvo no .txt na hora.
  3. Aperte de novo (ou ESC) para parar. Os arquivos ficam em
     apps/data/transcricoes/ com data e hora no nome.

Arquivos gerados por gravacao:
  AAAA-MM-DD_HHMMSS.txt   transcricao (uma linha/paragrafo por trecho)
  AAAA-MM-DD_HHMMSS.wav   audio 16 kHz mono
  AAAA-MM-DD_HHMMSS.json  inicio, duracao, modelo e os tempos de cada trecho

Uso:
  python apps/dictation_app.py
  python apps/dictation_app.py --list-devices
  python apps/dictation_app.py --device 2 --model base
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "voice"
if str(VOICE_DIR) not in sys.path:
    sys.path.insert(0, str(VOICE_DIR))

import vcommon as vc  # noqa: E402  (mic, VAD, Whisper e o transcritor do voice/)
from PySide6.QtCore import (  # noqa: E402
    QEventLoop,
    QMetaObject,
    QObject,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (  # noqa: E402
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QPen,
    QShortcut,
)
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

MODELS = ("tiny", "base", "small")
DEFAULT_MODEL = "base"
DEFAULT_LANG = "pt"
SILENCE_END = 0.7  # pausa (s) que fecha um trecho
MAX_SEGMENT_S = 15.0  # teto de duracao de um trecho
PARTIAL_EVERY = 1.2  # intervalo (s) entre transcricoes provisorias
POLL_MS = 20  # ritmo com que o worker consome o microfone
WAVE_SECONDS = 4.0


class Transcriber(vc.Transcriber):
    """O Transcriber do vcommon fixa portugues; aqui o idioma e configuravel."""

    def __init__(self, size=DEFAULT_MODEL, lang=DEFAULT_LANG):
        super().__init__(size)
        self.lang = lang

    def transcribe(self, wav):
        segs, _ = self.model.transcribe(
            np.asarray(wav, dtype=np.float32),
            language=self.lang or None,
            beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs).strip()


class Session:
    """Uma gravacao: escreve .txt e .wav na hora e fecha o .json no final."""

    def __init__(self, out_dir, model, lang):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base, n = stamp, 1
        while (self.dir / f"{base}.txt").exists():  # duas gravacoes no mesmo segundo
            n += 1
            base = f"{stamp}_{n}"
        self.txt_path = self.dir / f"{base}.txt"
        self.wav_path = self.dir / f"{base}.wav"
        self.json_path = self.dir / f"{base}.json"
        self.model, self.lang = model, lang
        self.t0 = time.time()
        self.segments = []
        self.words = 0
        self._txt = None
        self._wav = None

    def open(self):
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
        self.words += len(text.split())
        if self._txt is not None:
            self._txt.write(text + "\n")
            self._txt.flush()  # write-through: nada se perde se o app fechar

    def close(self):
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
        self.json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "txt": str(self.txt_path),
            "wav": str(self.wav_path),
            "json": str(self.json_path),
        }


class Stt(QObject):
    """Thread so para o Whisper: o loop do microfone nunca espera por ele.

    - commit (fechar um trecho): fila FIFO, nunca e descartado;
    - partial (texto provisorio): so o mais recente vale, o antigo e substituido.
    Os metodos submit_* sao chamados direto da thread do worker e so mexem em
    estruturas seguras sob GIL (deque / atributo simples).
    """

    ready = Signal(str)
    error = Signal(str)
    busy = Signal(bool)
    partial_ready = Signal(int, str)
    segment_ready = Signal(int, str, float, float)
    commit_done = Signal(int)
    selftest_done = Signal()

    def __init__(self, model_size, lang):
        super().__init__()
        self.model_size = model_size
        self.lang = lang
        self._stt = None
        self._timer = None
        self._commits = deque()
        self._pending_partial = None
        self._busy = False
        self._paused = False

    # ------------------------------------------------------------ enfileirar
    def submit_partial(self, seg_id, audio):
        self._pending_partial = (seg_id, audio)

    def submit_commit(self, job_id, seg_id, audio, start, end):
        if self._pending_partial is not None and self._pending_partial[0] == seg_id:
            self._pending_partial = None  # o commit ja vai cobrir esse trecho
        self._commits.append((job_id, seg_id, audio, start, end))

    @Slot()
    def start(self):
        try:
            self._stt = Transcriber(self.model_size, self.lang)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Falha ao carregar o modelo Whisper: {e}")
            return
        self.ready.emit(self.model_size)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._drain)
        self._timer.start()

    @Slot(str)
    def load_model(self, size):
        if self._busy or self._paused or size == self.model_size:
            return
        self._paused = True
        try:
            stt = Transcriber(size, self.lang)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Falha ao carregar o modelo Whisper: {e}")
            self._paused = False
            return
        self._stt, self.model_size = stt, size
        self._paused = False
        self.ready.emit(size)

    @Slot()
    def shutdown(self):
        if self._timer is not None:
            self._timer.stop()

    @Slot()
    def transcribe_selftest(self):
        try:
            self._stt.transcribe(np.zeros(vc.SR, np.float32))
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"selftest: {e}")
        finally:
            self.selftest_done.emit()

    # ------------------------------------------------------------ processar
    @Slot()
    def _drain(self):
        """Pega no maximo um trabalho (commit tem prioridade sobre partial) e roda."""
        if self._busy or self._paused or self._stt is None:
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
        self.busy.emit(True)
        try:
            text = self._stt.transcribe(audio)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"Falha na transcricao: {e}")
            text = ""
        self._busy = False

        if kind == "commit":
            self.segment_ready.emit(seg_id, text, start, end)
            self.commit_done.emit(job_id)
        else:
            self.partial_ready.emit(seg_id, text)

        # nao pisca o indicador se ja ha outro trabalho na fila
        if not self._commits and self._pending_partial is None:
            self.busy.emit(False)


class Worker(QObject):
    """Vive numa QThread: microfone + VAD + Whisper. Nunca trava a interface."""

    status = Signal(str)
    ready = Signal(str)
    error = Signal(str)
    level = Signal(float, bool)
    wave = Signal(object)
    partial = Signal(str)
    segment = Signal(str)
    saved = Signal(dict)
    stats = Signal(dict)

    def __init__(
        self, device=None, model=DEFAULT_MODEL, lang=DEFAULT_LANG, out_dir=None
    ):
        super().__init__()
        self.device = device
        self.model_size = model
        self.lang = lang
        self.out_dir = out_dir
        self.open_mic = True
        self.recording = False
        self.session = None
        self.mic = None
        self.stt = None  # o objeto Stt (thread do Whisper); so enfileiramos trabalhos
        self.det = None
        self._timer = None
        self._seg_start = None
        self._seg_id = 0
        self._last_t = None
        self._last_partial = 0.0
        self._partial_in_flight = False
        self._job_seq = 0      # id monotono dos commits (nao reseta entre gravacoes)
        self._last_job = None  # ultimo commit enfileirado nesta gravacao
        self._jobs_done = set()  # preenchido pelo handler, sempre na thread do worker
        self._rolling = np.zeros(int(vc.SR * WAVE_SECONDS), np.float32)

    # ------------------------------------------------------------ ciclo de vida
    @Slot()
    def setup(self):
        self.det = vc.UtteranceDetector(
            end_silence=SILENCE_END, max_seconds=MAX_SEGMENT_S
        )
        if self.open_mic:
            try:
                self.mic = vc.Mic(self.device)
            except SystemExit as e:
                self.error.emit(str(e))
                self.ready.emit(self.model_size)
                return
        self.ready.emit(self.model_size)
        self._timer = QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    @Slot()
    def shutdown(self):
        if self._timer is not None:
            self._timer.stop()
        if self.recording:
            self.stop_rec()
        if self.mic is not None:
            self.mic.close()
            self.mic = None

    # ------------------------------------------------------------ gravacao
    @Slot()
    def start_rec(self):
        if self.recording or self.mic is None or self.stt is None:
            return
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
        self.recording = True
        self.status.emit("Gravando")
        self.stats.emit({"recording": True, "elapsed": 0.0, "words": 0})

    @Slot()
    def stop_rec(self):
        if not self.recording:
            return
        self.recording = False
        self._partial_in_flight = False
        # Enfileira o que ainda estava falando e ESPERA TODA a fila de
        # transcricao terminar antes de fechar os arquivos: sem isso, trechos
        # que ja estavam na fila (VAD ja tinha fechado) chegavam depois do
        # close e se perdiam do .txt. O loop de eventos desta thread continua
        # rodando durante a espera, entao a interface nao trava e os resultados
        # vao gravando na hora.
        self._commit_pending()
        self._seg_start = None
        if self._last_job is not None and self._last_job not in self._jobs_done:
            self.status.emit("Finalizando...")
            self._wait_queue()
        paths = self.session.close() if self.session else None
        self.session = None
        self.det.reset()
        self.status.emit("Salvo")
        if paths:
            self.saved.emit(paths)
        self.stats.emit({"recording": False, "elapsed": 0.0, "words": 0})

    @Slot(object)
    def set_device(self, device):
        if self.recording:
            return
        if self.mic is not None:
            self.mic.close()
            self.mic = None
        try:
            self.mic = vc.Mic(device)
        except SystemExit as e:
            self.error.emit(str(e))
            return
        self.ready.emit(self.model_size)

    # ------------------------------------------------------------ resultados do Whisper
    @Slot(int, str)
    def _on_partial_ready(self, seg_id, text):
        if seg_id == self._seg_id and self.recording:
            self.partial.emit(text)
        self._partial_in_flight = False

    @Slot(int, str, float, float)
    def _on_segment_ready(self, seg_id, text, start, end):
        if self.session is not None and text.strip():
            self.session.add(text, start, end)
            self.segment.emit(text)
        if seg_id == self._seg_id:
            self.partial.emit("")
        self._partial_in_flight = False

    @Slot(int)
    def _on_commit_done(self, job_id):
        self._jobs_done.add(job_id)

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
            self.wave.emit(joined)
            rms = float(np.sqrt(np.mean(joined**2)))
        else:
            rms = 0.0

        if self.recording and blocks:
            last_t = blocks[-1][0]
            self._last_t = last_t
            for _, b in blocks:
                self.session.audio(b)
                self.det.feed(b)
            if self.det.active:
                if self._seg_start is None:
                    self._seg_start = last_t - len(self.det.buf) * vc.FRAME_S
                    self._seg_id += 1
                    self._partial_in_flight = False
                self._maybe_partial()
            if self.det.done:
                # tira do fim o silencio que fechou o trecho
                self._submit_commit(
                    self.det.audio, self._seg_start, last_t - self.det.silence
                )
                self.det.reset()
                self._seg_start = None

        self.level.emit(rms, rms > vc.ENERGY_MIN)
        if self.recording and self.session is not None:
            self.stats.emit(
                {
                    "recording": True,
                    "elapsed": time.time() - self.session.t0,
                    "words": self.session.words,
                }
            )

    def _maybe_partial(self):
        """Enfileira o texto provisorio - nunca bloqueia o loop esperando o Whisper."""
        if self._partial_in_flight:
            return  # back-pressure: so um por vez; se atrasar, atrasa o texto
        if len(self.det.buf) * vc.FRAME_S < 0.8:
            return
        now = time.time()
        if now - self._last_partial < PARTIAL_EVERY:
            return
        self._last_partial = now
        audio = np.concatenate(self.det.buf).astype(np.float32)
        self._partial_in_flight = True
        self.stt.submit_partial(self._seg_id, audio)

    def _submit_commit(self, audio, start_abs, end_abs):
        """Enfileira a transcricao definitiva de um trecho. Retorna o job (ou None se curto)."""
        if audio is None or len(audio) < int(0.4 * vc.SR) or self.session is None:
            return None
        t0 = self.session.t0
        start = (
            (start_abs - t0)
            if start_abs is not None
            else (end_abs - t0 - len(audio) / vc.SR)
        )
        end = max(start, end_abs - t0)
        job_id = self._job_seq
        self._job_seq += 1
        self._last_job = job_id
        self.stt.submit_commit(job_id, self._seg_id, audio, max(0.0, start), end)
        self._partial_in_flight = False
        return job_id

    def _commit_pending(self):
        """Fecha o trecho que ainda estava aberto quando o usuario parou."""
        if self.det.active and self.det.buf:
            audio = np.concatenate(self.det.buf).astype(np.float32)
            end = self._last_t if self._last_t is not None else time.time()
            return self._submit_commit(audio, self._seg_start, end)
        return None

    def _wait_queue(self):
        """Espera TODA a fila de transcricao terminar, processando os eventos
        desta thread (cada resultado chega e e gravado durante a espera)."""
        if self._last_job is None or self._last_job in self._jobs_done:
            return
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(50)
        deadline = time.time() + 120

        def check():
            if self._last_job in self._jobs_done or time.time() > deadline:
                timer.stop()
                loop.quit()

        timer.timeout.connect(check)
        timer.start()
        loop.exec()
        if self._last_job not in self._jobs_done:
            print(
                "[aviso] fila de transcricao nao terminou no tempo limite",
                file=sys.stderr,
            )


class Scope(QWidget):
    """Forma de onda dos ultimos segundos + medidor de nivel, desenhados no Qt."""

    def __init__(self):
        super().__init__()
        self.buf = np.zeros(int(vc.SR * WAVE_SECONDS), np.float32)
        self.rms, self.speaking, self.recording = 0.0, False, False
        self.setMinimumHeight(110)

    @Slot(object)
    def push(self, samples):
        n = min(len(samples), len(self.buf))
        if n > 0:
            self.buf = np.concatenate([self.buf[n:], samples[-n:].astype(np.float32)])
            self.update()

    @Slot(float, bool)
    def set_level(self, rms, speaking):
        self.rms, self.speaking = rms, speaking
        self.update()

    def set_recording(self, on):
        self.recording = on
        self.update()

    def paintEvent(self, _event):  # noqa: N802 (API do Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1a1d22"))
        p.drawRoundedRect(r, 10, 10)

        inner = r.adjusted(10, 10, -10, -30)
        mid = inner.center().y()
        p.setPen(QPen(QColor("#2a2f37"), 1))
        p.drawLine(inner.left(), mid, inner.right(), mid)

        if inner.width() > 0:
            step = max(1, len(self.buf) // inner.width())
            if self.recording:
                color = QColor("#3fb950") if self.speaking else QColor("#2f8f43")
            else:
                color = QColor("#5b6472")
            p.setPen(QPen(color, 1))
            for x in range(inner.width()):
                chunk = self.buf[x * step : (x + 1) * step]
                if chunk.size == 0:
                    continue
                amp = int(
                    min(1.0, float(np.abs(chunk).max()) * 4) * (inner.height() / 2)
                )
                p.drawLine(inner.left() + x, mid - amp, inner.left() + x, mid + amp)

        meter = r.adjusted(10, r.height() - 18, -10, -10)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#232830"))
        p.drawRoundedRect(meter, 4, 4)
        if meter.width() > 0:
            frac = float(np.clip((20 * np.log10(self.rms + 1e-6) + 60) / 60, 0, 1))
            if frac > 0:
                bar = meter.adjusted(0, 0, -int((1 - frac) * meter.width()), 0)
                p.setBrush(QColor("#3fb950" if frac < 0.75 else "#e5484d"))
                p.drawRoundedRect(bar, 4, 4)


class MainWindow(QMainWindow):
    req_start = Signal()
    req_stop = Signal()
    req_device = Signal(object)
    req_model = Signal(str)
    req_selftest = Signal()

    def __init__(
        self,
        device=None,
        model=DEFAULT_MODEL,
        lang=DEFAULT_LANG,
        out_dir=None,
        selftest=False,
    ):
        super().__init__()
        self.setWindowTitle("Ditado - transcricao ao vivo")
        self.resize(820, 640)
        self.out_dir = Path(out_dir)
        self._recording = False
        self._closing = False
        self._t0 = 0.0
        self._committed = []
        self._partial = ""
        self._ready_mic = False
        self._ready_model = False

        self._build_ui()
        self._populate_devices(device)
        self._start_worker(device, model, lang, open_mic=not selftest)
        self._on_status(f"Carregando modelo Whisper '{model}'...")

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(500)
        self._ui_timer.timeout.connect(self._tick)
        self._ui_timer.start()

    # ------------------------------------------------------------ interface
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(12)

        top = QHBoxLayout()
        self.pill = QLabel("Iniciando...")
        self.pill.setObjectName("pill")
        top.addWidget(self.pill)
        self.lbl_proc = QLabel("transcrevendo…")
        self.lbl_proc.setObjectName("proc")
        self.lbl_proc.hide()
        top.addWidget(self.lbl_proc)
        top.addStretch(1)
        top.addWidget(QLabel("Microfone"))
        self.cmb_device = QComboBox()
        self.cmb_device.setMinimumWidth(220)
        self.cmb_device.currentIndexChanged.connect(self._on_device)
        top.addWidget(self.cmb_device)
        top.addWidget(QLabel("Modelo"))
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(MODELS)
        self.cmb_model.setCurrentText(DEFAULT_MODEL)
        self.cmb_model.currentTextChanged.connect(self._on_model_change)
        top.addWidget(self.cmb_model)
        lay.addLayout(top)

        self.scope = Scope()
        lay.addWidget(self.scope)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setObjectName("transcript")
        self._render()
        lay.addWidget(self.text, 1)

        bottom = QHBoxLayout()
        self.btn_rec = QPushButton("GRAVAR")
        self.btn_rec.setObjectName("rec")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self._toggle)
        bottom.addWidget(self.btn_rec)
        self.btn_open = QPushButton("Abrir pasta")
        self.btn_open.clicked.connect(self._open_folder)
        bottom.addWidget(self.btn_open)
        bottom.addStretch(1)
        self.lbl_info = QLabel("00:00  ·  0 palavras")
        self.lbl_info.setObjectName("info")
        bottom.addWidget(self.lbl_info)
        lay.addLayout(bottom)

        self.hint = QLabel(
            "ESPACO / ENTER = gravar ou parar   |   ESC = sair   |   "
            f"salvando em {self.out_dir}"
        )
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._toggle)
        QShortcut(QKeySequence(Qt.Key_Return), self, activated=self._toggle)
        QShortcut(QKeySequence(Qt.Key_Enter), self, activated=self._toggle)

    def _populate_devices(self, device):
        self.cmb_device.addItem("Microfone padrao do sistema", None)
        try:
            import sounddevice as sd

            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    self.cmb_device.addItem(f"{i}: {d['name']}", i)
        except Exception:  # noqa: BLE001
            pass
        if device is not None:
            idx = self.cmb_device.findData(device)
            if idx >= 0:
                self.cmb_device.setCurrentIndex(idx)

    def _start_worker(self, device, model, lang, open_mic):
        # thread do Whisper (transcricao) - separada do microfone de proposito
        self.stt_thread = QThread(self)
        self.stt = Stt(model, lang)
        self.stt.moveToThread(self.stt_thread)
        self.stt_thread.started.connect(self.stt.start)

        # thread do microfone (captura + VAD + escrita dos arquivos)
        self.thread = QThread(self)
        self.worker = Worker(device, model, lang, self.out_dir)
        self.worker.open_mic = open_mic
        self.worker.stt = self.stt
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.setup)

        # pedidos da interface -> worker / stt
        self.req_start.connect(self.worker.start_rec)
        self.req_stop.connect(self.worker.stop_rec)
        self.req_device.connect(self.worker.set_device)
        self.req_model.connect(self.stt.load_model)
        self.req_selftest.connect(self.stt.transcribe_selftest)

        # worker -> interface
        self.worker.status.connect(self._on_status)
        self.worker.error.connect(self._on_error)
        self.worker.level.connect(self.scope.set_level)
        self.worker.wave.connect(self.scope.push)
        self.worker.partial.connect(self._on_partial)
        self.worker.segment.connect(self._on_segment)
        self.worker.saved.connect(self._on_saved)
        self.worker.stats.connect(self._on_stats)
        self.worker.ready.connect(self._on_worker_ready)

        # stt -> worker (resultados chegam na thread do microfone)
        self.stt.partial_ready.connect(self.worker._on_partial_ready)
        self.stt.segment_ready.connect(self.worker._on_segment_ready)
        self.stt.commit_done.connect(self.worker._on_commit_done)

        # stt -> interface
        self.stt.ready.connect(self._on_model_ready)
        self.stt.error.connect(self._on_error)
        self.stt.busy.connect(self._on_busy)
        self.stt.selftest_done.connect(self._selftest_done)

        self.stt_thread.start()
        self.thread.start()

    # ------------------------------------------------------------ acoes
    def _toggle(self):
        if self._recording:
            self._stop()
        else:
            self._start()

    def _start(self):
        self._committed, self._partial = [], ""
        self._render()
        self.btn_rec.setChecked(True)
        self.req_start.emit()

    def _stop(self):
        self.btn_rec.setChecked(False)
        self.req_stop.emit()

    def _on_device(self, _idx):
        self.req_device.emit(self.cmb_device.currentData())

    def _open_folder(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out_dir)))

    # ------------------------------------------------------------ sinais do worker
    def _on_status(self, msg):
        colors = {"Pronto": "#3fb950", "Gravando": "#e5484d", "Salvo": "#3fb950"}
        waiting = msg.startswith(("Carregando", "Finalizando"))
        color = colors.get(msg, "#d0a215" if waiting else "#8b929c")
        self.pill.setText(msg)
        self.pill.setStyleSheet(
            f"background: {color}22; color: {color}; border: 1px solid {color}55;"
            "border-radius: 12px; padding: 4px 14px; font-weight: 600;"
        )
        if msg == "Gravando":
            self._recording, self._t0 = True, time.time()
            self.scope.set_recording(True)
            self.btn_rec.setText("PARAR")
            self.cmb_device.setEnabled(False)
            self.cmb_model.setEnabled(False)
        elif msg == "Finalizando...":
            self.btn_rec.setEnabled(False)
        elif msg in ("Pronto", "Salvo"):
            self.btn_rec.setEnabled(True)
            if self._recording:
                self._recording = False
                self.scope.set_recording(False)
            self.btn_rec.setText("GRAVAR")
            self.btn_rec.setChecked(False)
            self.cmb_device.setEnabled(True)
            self.cmb_model.setEnabled(True)

    def _on_worker_ready(self, _model=""):
        self._ready_mic = True
        self._maybe_ready()

    def _on_model_ready(self, _model=""):
        self._ready_model = True
        self._maybe_ready()

    def _maybe_ready(self):
        if self._ready_mic and self._ready_model and not self._recording:
            self.btn_rec.setEnabled(True)
            self._on_status("Pronto")

    def _on_model_change(self, size):
        if not hasattr(self, "stt") or size == self.stt.model_size or self._recording:
            return
        self._ready_model = False
        self.btn_rec.setEnabled(False)
        self._on_status(f"Carregando modelo Whisper '{size}'...")
        self.req_model.emit(size)

    @Slot(bool)
    def _on_busy(self, busy):
        self.lbl_proc.setVisible(bool(busy))

    def _on_error(self, msg):
        low = msg.lower()
        if "microfone" in low or "modelo" in low:  # erro que impede gravar
            self.btn_rec.setEnabled(False)
        self.pill.setText("Erro")
        self.pill.setStyleSheet(
            "background: #e5484d22; color: #e5484d; border: 1px solid #e5484d55;"
            "border-radius: 12px; padding: 4px 14px; font-weight: 600;"
        )
        self.lbl_info.setText(msg.splitlines()[0][:120])
        print(f"[erro] {msg}", file=sys.stderr)

    @Slot(str)
    def _on_partial(self, text):
        self._partial = text
        self._render()

    @Slot(str)
    def _on_segment(self, text):
        self._committed.append(text)
        self._partial = ""
        self._render()

    @Slot(dict)
    def _on_saved(self, paths):
        self.lbl_info.setText(f"{getattr(self, '_words', 0)} palavras")
        self.hint.setText(f"ultimo arquivo: {paths['txt']}")

    @Slot(dict)
    def _on_stats(self, stats):
        if stats.get("recording"):
            self._words = stats.get("words", 0)
        else:
            self._words = len(" ".join(self._committed).split())

    # ------------------------------------------------------------ selftest
    @Slot(str)
    def _selftest_ready(self, _model):
        QTimer.singleShot(300, self.req_selftest.emit)

    @Slot()
    def _selftest_done(self):
        print("selftest: ok (interface + 2 threads + modelo + transcricao)", flush=True)
        self.close()

    # ------------------------------------------------------------ util
    def _render(self):
        blocks = "".join(
            f"<p style='margin:0 0 10px 0'>{html.escape(t)}</p>"
            for t in self._committed
        )
        if self._partial:
            blocks += (
                "<p style='margin:0 0 10px 0;color:#8b929c;font-style:italic'>"
                f"{html.escape(self._partial)}</p>"
            )
        if not blocks:
            blocks = (
                "<p style='color:#5b6472'>Aperte GRAVAR e comece a falar. "
                "O texto aparece aqui enquanto voce fala.</p>"
            )
        self.text.setHtml(blocks)
        bar = self.text.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _tick(self):
        elapsed = (time.time() - self._t0) if self._recording else 0.0
        words = getattr(self, "_words", len(" ".join(self._committed).split()))
        self.lbl_info.setText(
            f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}  ·  {words} palavras"
        )

    def keyPressEvent(self, event):  # noqa: N802 (API do Qt)
        if event.key() == Qt.Key_Escape:
            if self._recording:
                self._stop()
            else:
                self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):  # noqa: N802 (API do Qt)
        if not self._closing:
            self._closing = True
            self._ui_timer.stop()
            QMetaObject.invokeMethod(
                self.worker, "shutdown", Qt.BlockingQueuedConnection
            )
            QMetaObject.invokeMethod(self.stt, "shutdown", Qt.BlockingQueuedConnection)
            self.thread.quit()
            self.thread.wait(5000)
            self.stt_thread.quit()
            self.stt_thread.wait(5000)
        super().closeEvent(event)


STYLE = """
#root { background: #14161a; }
QLabel { color: #cbd2dc; }
QLabel#pill { background: #232830; color: #cbd2dc; border: 1px solid #2f3640;
              border-radius: 12px; padding: 4px 14px; font-weight: 600; }
QLabel#info { color: #8b929c; }
QLabel#proc { color: #d0a215; font-size: 12px; }
QLabel#hint { color: #5b6472; font-size: 11px; }
QTextEdit#transcript { background: #1a1d22; border: 1px solid #2a2f37; border-radius: 10px;
                       color: #e8eaed; font-size: 16px; padding: 12px; }
QComboBox { background: #1d2026; border: 1px solid #2a2f37; border-radius: 8px;
            padding: 5px 10px; color: #dfe3e8; }
QComboBox:hover { border-color: #3a4049; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView { background: #1d2026; color: #dfe3e8; border: 1px solid #2a2f37;
                              selection-background-color: #2f3640; }
QPushButton { background: #232830; border: 1px solid #2f3640; border-radius: 8px;
              padding: 8px 18px; color: #dfe3e8; font-weight: 600; }
QPushButton:hover { background: #2b313a; }
QPushButton#rec { background: #e5484d; border: none; border-radius: 26px;
                  padding: 14px 34px; color: white; font-size: 15px; font-weight: 700; }
QPushButton#rec:hover { background: #f2555a; }
QPushButton#rec:checked { background: #3a3f47; color: #ffb3b6; }
"""


def main():
    p = argparse.ArgumentParser(
        description="Ditado: transcricao ao vivo para .txt/.wav/.json"
    )
    p.add_argument("--device", help="indice ou nome (parcial) do microfone")
    p.add_argument(
        "--list-devices", action="store_true", help="lista os microfones e sai"
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL, choices=MODELS, help="tamanho do Whisper"
    )
    p.add_argument("--lang", default=DEFAULT_LANG, help="idioma da fala (ex: pt, en)")
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "data" / "transcricoes"),
        help="pasta onde salvar as gravacoes",
    )
    p.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    device = a.device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    if a.selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    families = QFontDatabase.families()
    family = next(
        (
            f
            for f in ("SF Pro Text", "Helvetica Neue", "Inter", "Segoe UI")
            if f in families
        ),
        "",
    )
    app.setFont(QFont(family, 13))
    app.setStyleSheet(STYLE)

    win = MainWindow(device, a.model, a.lang, a.out, selftest=a.selftest)
    win.show()

    if a.selftest:
        win.stt.ready.connect(win._selftest_ready)
        QTimer.singleShot(60000, win.close)

    code = app.exec()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
