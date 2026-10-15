"""Nivel 4: ROSTO + VOZ juntos.

Fluxo:
  1. FACE   -> o nivel 3 do Face Lab (reconhece o rosto + desafio de piscar/virar) ate liberar
  2. VOZ    -> aparecem digitos aleatorios; voce os diz em voz alta enquanto a camera continua te vendo
  3. Checagens (todas precisam passar):
       - conteudo: os digitos ditos sao os pedidos (Whisper)
       - voz: o locutor e a MESMA pessoa reconhecida pelo rosto (ECAPA)
       - labios: a abertura da boca acompanha o volume da voz (sincronia labial).
         Uma gravacao tocando num alto-falante com o rosto parado nao passa.
  4. GRANTED -> liberado enquanto o rosto continuar la (checagem leve). Aperte V para refazer a prova de voz.

Requisitos: a pasta face/ ao lado desta (no mesmo projeto) com rostos cadastrados,
e voz cadastrada com o MESMO NOME. Sair: q / ESC.   Opcoes: --list-devices | --device N
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FACE_DIR = Path(os.environ.get("FACE_LAB_DIR", ROOT / "face"))
if not (FACE_DIR / "common.py").exists():
    raise SystemExit(
        f"Nao achei o Face Lab em: {FACE_DIR}\n"
        "Esperava a pasta 'face' ao lado de 'voice' no mesmo projeto."
    )
# face/ e um pacote "flat": os modulos dele importam `from common import ...`,
# entao o direitorio face/ entra no path (e nao a raiz do projeto).
sys.path.insert(0, str(FACE_DIR))

import vcommon as vc  # noqa: E402

from common import (  # noqa: E402
    GRAY,
    GREEN,
    RED,
    YELLOW,
    FaceTracker,
    draw_banner,
    draw_face_box,
    draw_text,
    open_camera,
    read_mirrored,
)
from common import load_db as load_face_db  # noqa: E402
from level3_liveness import Auth  # noqa: E402  (do Face Lab)

WIN = "Nivel 4 - Rosto + Voz"
PREP_S = 1.2
LISTEN_TIMEOUT = 10.0
DENY_S = 4.0
FACE_CHECK_EVERY = 0.4  # segundos entre checagens do rosto durante a prova de voz
UPPER_LIP, LOWER_LIP, MOUTH_L, MOUTH_R = 13, 14, 78, 308


def mouth_open_ratio(face):
    p = face.pts
    return float(
        np.linalg.norm(p[UPPER_LIP] - p[LOWER_LIP])
        / (np.linalg.norm(p[MOUTH_L] - p[MOUTH_R]) + 1e-6)
    )


def main():
    dev = vc.parse_args("Nivel 4: rosto + voz")
    face_db, voice_db = load_face_db(), vc.load_voices()
    if not face_db:
        raise SystemExit(
            "Nenhum rosto cadastrado. No Face Lab: python level2_enroll.py"
        )
    if not voice_db:
        raise SystemExit("Nenhuma voz cadastrada. Aqui: python level2_enroll_voice.py")
    voice_names = {n.lower(): n for n in voice_db}
    if not {n.lower() for n in face_db} & set(voice_names):
        raise SystemExit(
            f"Nenhum nome em comum. Rostos: {list(face_db)} | Vozes: {list(voice_db)}\n"
            "Cadastre rosto e voz com o MESMO nome."
        )

    print("Carregando modelos (a primeira vez baixa os modelos, pode demorar)...")
    enc, stt = vc.SpeakerEncoder(), vc.Transcriber()
    cap, tracker = open_camera(), FaceTracker(max_faces=2)
    mic = vc.Mic(dev)
    det = vc.UtteranceDetector(end_silence=0.9, max_seconds=10.0)
    auth = Auth(face_db)

    phase, t0, last_face_check = "FACE", 0.0, 0.0
    digits, face_user, checks, deny_msg = [], None, [], ""
    mar_ts, mar_vals, aud_ts, aud_rms = [], [], [], []

    def begin_voice(now):
        nonlocal phase, t0, digits
        digits, phase, t0 = vc.random_digits(), "PREP", now

    while True:
        frame = read_mirrored(cap)
        if frame is None:
            break
        now = time.time()
        h, w = frame.shape[:2]
        faces = tracker.process(frame)
        face = faces[0] if len(faces) == 1 else None
        text, color, sub = "", GRAY, ""
        delay = 1
        fail = None
        if phase != "LISTEN":
            mic.drain()  # evita a fila de audio crescer fora da escuta

        if phase == "FACE":
            text, color, sub = auth.update(frame, faces, now)
            if auth.state == "VERIFIED":
                face_user = auth.user
                if face_user.lower() not in voice_names:
                    fail = f"{face_user} tem rosto, mas nao tem voz cadastrada"
                else:
                    begin_voice(now)

        elif phase in ("PREP", "LISTEN"):
            if (
                now - last_face_check >= FACE_CHECK_EVERY
            ):  # rosto continua sendo o mesmo?
                last_face_check = now
                auth.update(frame, faces, now)
                if auth.state != "VERIFIED":
                    fail = "Rosto perdido ou trocado durante a prova de voz"

            if phase == "PREP":
                text, color, sub = (
                    f"Ola, {face_user}! Prepare-se...",
                    YELLOW,
                    "Voce vai dizer 4 numeros",
                )
                if now - t0 >= PREP_S:
                    det.reset()
                    mic.drain()
                    mar_ts, mar_vals, aud_ts, aud_rms = [], [], [], []
                    phase, t0 = "LISTEN", now
            else:
                text, color, sub = (
                    "DIGA OS NUMEROS EM VOZ ALTA",
                    YELLOW,
                    "olhe para a camera",
                )
                if face is not None:
                    mar_ts.append(now)
                    mar_vals.append(mouth_open_ratio(face))
                for t, b in mic.blocks():
                    aud_ts.append(t)
                    aud_rms.append(float(np.sqrt(np.mean(b**2))))
                    det.feed(b)
                if det.done:
                    cv2.rectangle(frame, (0, 0), (w, 48), YELLOW, -1)
                    draw_text(frame, "Analisando voz e labios...", (12, 33), scale=0.75)
                    cv2.imshow(WIN, frame)
                    cv2.waitKey(1)
                    audio, voiced = det.audio, det.voiced
                    heard_text = stt.transcribe(audio)
                    heard = vc.digits_from_text(heard_text)
                    ok_content = vc.digit_errors(digits, heard) <= vc.MAX_DIGIT_ERRORS
                    spk, score, _ = vc.identify_voice(enc.embed(audio), voice_db)
                    ok_voice = spk is not None and spk.lower() == face_user.lower()
                    corr, mar_range = vc.lipsync_score(
                        mar_ts, mar_vals, aud_ts, aud_rms
                    )
                    ok_lips = (
                        corr >= vc.LIPSYNC_MIN_CORR and mar_range >= vc.MIN_MAR_RANGE
                    )
                    checks = [
                        (
                            ok_content,
                            f"Conteudo: pedido {' '.join(map(str, digits))} | ouvi {' '.join(map(str, heard)) or '-'}",
                        ),
                        (
                            ok_voice,
                            f"Voz: {spk or 'desconhecida'} ({score:.2f}) vs rosto: {face_user}",
                        ),
                        (
                            ok_lips,
                            f"Labios: sincronia {corr:.2f} (min {vc.LIPSYNC_MIN_CORR}), movimento {mar_range:.2f}",
                        ),
                    ]
                    if voiced < 1.0:
                        fail = "Fala muito curta"
                    elif not all(c[0] for c in checks):
                        fail = "Falhou: " + ", ".join(
                            n
                            for (ok, _), n in zip(checks, ("conteudo", "voz", "labios"))
                            if not ok
                        )
                    else:
                        phase, t0 = "GRANTED", now
                    mic.drain()
                elif not det.active and now - t0 > LISTEN_TIMEOUT:
                    fail = "Nenhuma fala detectada"

        elif phase == "GRANTED":
            delay = 400  # modo economia, como no Face Lab
            auth.update(frame, faces, now)
            if auth.state != "VERIFIED":
                phase = "FACE"
                auth.reset()
            text, color, sub = (
                f"ACESSO LIBERADO: {face_user} (rosto + voz)",
                GREEN,
                "V = refazer prova de voz | q / ESC = sair",
            )

        elif phase == "DENIED":
            text, color, sub = "ACESSO NEGADO", RED, deny_msg
            if now - t0 >= DENY_S:
                auth.reset()
                phase = "FACE"

        if fail:
            phase, t0, deny_msg = "DENIED", now, fail
            text, color, sub = "ACESSO NEGADO", RED, fail

        for f in faces:
            draw_face_box(frame, f, color)
        draw_banner(frame, text, color, sub or "q / ESC = sair")
        if phase in ("PREP", "LISTEN"):
            vc.center_text(
                frame,
                "  ".join(map(str, digits)),
                h // 2 + 120,
                (255, 255, 255),
                scale=2.4,
                thick=5,
            )
        if phase in ("GRANTED", "DENIED"):
            for i, (ok, line) in enumerate(checks):
                draw_text(
                    frame,
                    ("[OK] " if ok else "[X] ") + line,
                    (12, h - 70 + 22 * i),
                    GREEN if ok else RED,
                    scale=0.5,
                    thick=1,
                )
        cv2.imshow(WIN, frame)

        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("v") and phase == "GRANTED":
            begin_voice(time.time())

    cap.release()
    mic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
