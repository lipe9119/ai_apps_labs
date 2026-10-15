"""Nivel 1: abre a webcam e detecta rostos. Sem reconhecimento ainda. Sair: q ou ESC."""

import cv2

from common import (
    FPS,
    GRAY,
    GREEN,
    FaceTracker,
    draw_banner,
    draw_face_box,
    draw_text,
    open_camera,
    read_mirrored,
)


def main():
    cap = open_camera()
    tracker = FaceTracker(max_faces=4)
    fps = FPS()
    while True:
        frame = read_mirrored(cap)
        if frame is None:
            break
        faces = tracker.process(frame)
        for f in faces:
            draw_face_box(frame, f, GREEN)
            # alguns pontos do rosto so para visualizar
            for i in (1, 33, 263, 61, 291, 152):
                x, y = f.pts[i]
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
            draw_text(
                frame,
                f"yaw {f.yaw:.2f}  olho {f.ear:.2f}",
                (f.bbox[0], f.bbox[3] + 22),
                scale=0.55,
            )
        draw_banner(
            frame,
            f"Rostos: {len(faces)}    FPS: {fps.tick():.0f}",
            GRAY,
            "q / ESC = sair",
        )
        cv2.imshow("Nivel 1 - Deteccao", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
