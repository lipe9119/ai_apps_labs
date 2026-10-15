"""Nivel 2 (parte 1): cadastro guiado.

1. Digite o nome na propria janela e aperte ENTER.
2. Siga as poses (frente, esquerda, direita). O programa so conta 3-2-1 quando detecta que voce
   realmente esta na pose, e tira a foto ao fim da contagem.
Sair: ESC.
"""

import math
import time

import cv2

from common import (
    FRONT_RANGE,
    GRAY,
    GREEN,
    LEFT_RANGE,
    MIN_FACE_PX,
    RED,
    RIGHT_RANGE,
    YELLOW,
    FaceTracker,
    draw_banner,
    draw_face_box,
    draw_text,
    embedding,
    open_camera,
    read_mirrored,
    save_person,
)

# (id, instrucao, faixa de yaw)  -- so ASCII, o OpenCV nao desenha acentos
POSES = [
    ("frente", "Olhe de FRENTE para a camera", FRONT_RANGE),
    ("esquerda", "Vire o rosto para a ESQUERDA da tela", LEFT_RANGE),
    ("direita", "Vire o rosto para a DIREITA da tela", RIGHT_RANGE),
]
HOLD_SECONDS = 0.6  # tempo estavel na pose antes de comecar a contagem
COUNTDOWN = 3  # 3-2-1
SAMPLES_PER_POSE = 3  # embeddings guardados por pose


def pose_ok(face, rng):
    return face is not None and face.width >= MIN_FACE_PX and rng[0] < face.yaw < rng[1]


def main():
    cap = open_camera()
    tracker = FaceTracker(max_faces=2)
    name, stage, pose_idx, t0 = "", "name", 0, None
    samples, grabbed = [], []
    flash_until = 0.0

    while True:
        frame = read_mirrored(cap)
        if frame is None:
            break
        now = time.time()
        h, w = frame.shape[:2]
        faces = tracker.process(frame)
        face = faces[0] if len(faces) == 1 else None
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if stage == "name":
            draw_banner(
                frame, "CADASTRO - digite seu nome e aperte ENTER", GRAY, "ESC = sair"
            )
            draw_text(frame, "Nome: " + name + "_", (20, h // 2), scale=1.0)
            if key == 13 and name.strip():
                name, stage, t0 = name.strip(), "hold", None
            elif key in (8, 127):
                name = name[:-1]
            elif 32 <= key <= 126 and len(name) < 24:
                name += chr(key)

        elif stage == "done":
            draw_banner(frame, f"Cadastro de {name} concluido!", GREEN)
            if now - t0 > 2.0:
                break

        else:
            pose_id, msg, rng = POSES[pose_idx]
            ok = pose_ok(face, rng)

            if len(faces) > 1:
                hint = "Apenas um rosto na imagem"
            elif face is None:
                hint = "Procurando rosto..."
            elif face.width < MIN_FACE_PX:
                hint = "Aproxime-se da camera"
            elif not ok:
                hint = msg
            else:
                hint = "Isso! Fique assim"

            if stage == "hold":
                if ok:
                    t0 = t0 or now
                    if now - t0 >= HOLD_SECONDS:
                        stage, t0 = "count", now
                else:
                    t0 = None
            elif stage == "count":
                if not ok:
                    stage, t0 = "hold", None
                else:
                    left = COUNTDOWN - (now - t0)
                    if left <= 0:
                        stage, grabbed = "grab", []
                    else:
                        draw_text(
                            frame,
                            str(math.ceil(left)),
                            (w // 2 - 30, h // 2 + 30),
                            color=YELLOW,
                            scale=4,
                            thick=6,
                        )
            if stage == "grab":
                if ok:
                    emb = embedding(frame, face)
                    if emb is not None:
                        grabbed.append(emb)
                    if len(grabbed) >= SAMPLES_PER_POSE:
                        samples += [(pose_id, e) for e in grabbed]
                        flash_until = now + 0.25
                        pose_idx += 1
                        if pose_idx >= len(POSES):
                            save_person(name, samples)
                            stage, t0 = "done", now
                        else:
                            stage, t0 = "hold", None
                else:
                    stage, t0, grabbed = "hold", None, []

            if stage != "done":
                if face is not None:
                    draw_face_box(frame, face, GREEN if ok else RED)
                draw_banner(
                    frame,
                    f"Pose {pose_idx + 1}/{len(POSES)}: {hint}",
                    GREEN if ok else GRAY,
                    f"Cadastrando: {name}   (ESC = cancelar)",
                )
            if now < flash_until:
                frame[:] = cv2.addWeighted(frame, 0.3, frame * 0 + 255, 0.7, 0)

        cv2.imshow("Nivel 2 - Cadastro", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
