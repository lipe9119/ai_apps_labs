"""Nivel 2 (parte 2): reconhecimento de varias pessoas cadastradas. Sair: q ou ESC.

Para economizar CPU, o embedding (parte cara) so e calculado ~4x por segundo;
a deteccao/desenho acontece a cada frame.
"""

import time

import cv2
import numpy as np

from common import (
    FPS,
    GRAY,
    GREEN,
    RED,
    FaceTracker,
    draw_banner,
    draw_face_box,
    embedding,
    identify,
    load_db,
    open_camera,
    read_mirrored,
)

RECOG_INTERVAL = 0.25  # segundos entre calculos de embedding


def main():
    db = load_db()
    if not db:
        raise SystemExit(
            "Nenhum rosto cadastrado. Rode primeiro: python level2_enroll.py"
        )
    cap = open_camera()
    tracker = FaceTracker(max_faces=3)
    fps = FPS()
    labels = []  # [(centro, nome ou None, distancia)]
    last = 0.0

    while True:
        frame = read_mirrored(cap)
        if frame is None:
            break
        now = time.time()
        faces = tracker.process(frame)

        if now - last >= RECOG_INTERVAL:
            last = now
            labels = []
            for f in faces:
                emb = embedding(frame, f)
                name, dist = identify(emb, db) if emb is not None else (None, 9.0)
                labels.append((f.center, name, dist))

        for f in faces:
            lab = min(
                labels, key=lambda l: np.linalg.norm(l[0] - f.center), default=None
            )
            if lab is None or np.linalg.norm(lab[0] - f.center) > f.width:
                draw_face_box(frame, f, GRAY, "...")
            elif lab[1] is None:
                draw_face_box(frame, f, RED, f"Desconhecido ({lab[2]:.2f})")
            else:
                draw_face_box(frame, f, GREEN, f"{lab[1]} ({lab[2]:.2f})")

        draw_banner(
            frame,
            f"Cadastrados: {', '.join(db)}",
            GRAY,
            f"FPS {fps.tick():.0f}   |   q / ESC = sair",
        )
        cv2.imshow("Nivel 2 - Reconhecimento", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
