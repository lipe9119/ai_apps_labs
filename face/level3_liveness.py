"""Nivel 3: reconhecimento + prova de vivacidade (anti foto/video).

Fluxo:
  IDENTIFY  -> reconhece uma pessoa cadastrada, de frente, varias vezes seguidas
  CHALLENGE -> sorteia 2 desafios aleatorios (piscar 2x / virar p/ esquerda / virar p/ direita)
  FINAL     -> pede para voltar a olhar de frente e reconfirma que e a MESMA pessoa
  VERIFIED  -> liberado; entra em modo economia (checa ~2.5x/s). Se sumir/trocar de pessoa, recomeca
  DENIED    -> falhou em algum passo; mostra o motivo e recomeca

Uma foto ou video gravado nao sabe qual desafio vai ser pedido e nao reage a ele.
Sair: q ou ESC.
"""
import random
import time
from collections import deque

import cv2
import numpy as np

from common import (GRAY, GREEN, MIN_FACE_PX, RED, TURN_LEFT_BELOW, TURN_RIGHT_ABOVE, YELLOW, FaceTracker,
                    draw_banner, draw_face_box, embedding, identify, load_db, open_camera, read_mirrored)

CHALLENGE_TEXT = {
    "blink": "Pisque 2 vezes",
    "left": "Vire o rosto para a ESQUERDA da tela",
    "right": "Vire o rosto para a DIREITA da tela",
}
N_CHALLENGES = 2
CHALLENGE_TIMEOUT = 6.0   # segundos por desafio
ID_STABLE = 3             # reconhecimentos seguidos antes de liberar o desafio
CHECK_EVERY_MS = 400      # ritmo no modo VERIFIED
MISS_LIMIT = 4            # checagens falhas seguidas p/ encerrar a sessao (~1.6 s)
DENY_SECONDS = 2.5


class Auth:
    def __init__(self, db):
        self.db = db
        self.reset()

    def reset(self):
        self.state = "IDENTIFY"
        self.id_name, self.id_hits, self.unknown, self.last_embed = None, 0, False, 0.0
        self.ear_samples = deque(maxlen=40)
        self.blink_thr = 0.2
        self.challenges, self.idx, self.ch_start = [], 0, 0.0
        self.blinks, self.closed, self.turn_frames = 0, 0, 0
        self.lost_since, self.prev_center = None, None
        self.final_start = 0.0
        self.user, self.misses = None, 0
        self.deny_until, self.reason = 0.0, ""

    # ------------------------------------------------------------ utilitarios
    @staticmethod
    def _frontal(face, tol=0.08):
        return face is not None and face.width >= MIN_FACE_PX and abs(face.yaw - 0.5) < tol

    def _match(self, frame, face):
        emb = embedding(frame, face)
        return identify(emb, self.db)[0] if emb is not None else None

    def _fail(self, reason, now):
        self.state, self.reason, self.deny_until = "DENIED", reason, now + DENY_SECONDS
        return "ACESSO NEGADO", RED, reason

    # ------------------------------------------------------------ maquina de estados
    def update(self, frame, faces, now):
        face = faces[0] if len(faces) == 1 else None
        if self.state == "DENIED":
            if now >= self.deny_until:
                self.reset()
            return "ACESSO NEGADO", RED, self.reason
        if self.state == "IDENTIFY":
            return self._identify(frame, faces, face, now)
        if self.state == "CHALLENGE":
            return self._challenge(frame, faces, face, now)
        if self.state == "FINAL":
            return self._final(frame, face, now)
        return self._verified(frame, faces, face)

    def _identify(self, frame, faces, face, now):
        if len(faces) > 1:
            self.id_name, self.id_hits = None, 0
            return "Apenas uma pessoa na frente da camera", YELLOW, ""
        if face is None:
            self.id_name, self.id_hits, self.unknown = None, 0, False
            return "Procurando rosto...", GRAY, ""
        if not self._frontal(face):
            return "Olhe de frente para a camera", YELLOW, ""

        self.ear_samples.append(face.ear)
        if now - self.last_embed >= 0.3:
            self.last_embed = now
            name = self._match(frame, face)
            self.unknown = name is None
            if name is None:
                self.id_name, self.id_hits = None, 0
            elif name == self.id_name:
                self.id_hits += 1
            else:
                self.id_name, self.id_hits = name, 1

        if self.unknown:
            return "Rosto desconhecido", RED, ""
        if self.id_hits >= ID_STABLE and len(self.ear_samples) >= 15:
            baseline = float(np.percentile(self.ear_samples, 75))   # olho aberto "normal" desta pessoa
            self.blink_thr = baseline * 0.75
            self.challenges = random.sample(list(CHALLENGE_TEXT), N_CHALLENGES)
            self.state, self.idx, self.ch_start = "CHALLENGE", 0, now
            self.blinks = self.closed = self.turn_frames = 0
            self.lost_since = self.prev_center = None
            return f"Ola, {self.id_name}! Prova de vida...", YELLOW, ""
        return "Verificando identidade...", YELLOW, ""

    def _challenge(self, frame, faces, face, now):
        kind = self.challenges[self.idx]
        title = f"Desafio {self.idx + 1}/{len(self.challenges)}: {CHALLENGE_TEXT[kind]}"
        w = frame.shape[1]

        if len(faces) > 1:
            return self._fail("Mais de um rosto no quadro", now)
        if face is None:
            self.lost_since = self.lost_since or now
            if now - self.lost_since > 0.8:
                return self._fail("Rosto perdido durante o desafio", now)
        else:
            self.lost_since = None
            c = face.center
            if self.prev_center is not None and np.linalg.norm(c - self.prev_center) > 0.25 * w:
                return self._fail("Troca de rosto detectada", now)
            self.prev_center = c

            done = False
            if kind == "blink":
                if abs(face.yaw - 0.5) < 0.10:                       # so conta piscada com rosto de frente
                    if face.ear < self.blink_thr:
                        self.closed += 1
                    elif face.ear > self.blink_thr * 1.1:
                        if self.closed >= 1:
                            self.blinks += 1
                        self.closed = 0
                done = self.blinks >= 2
            else:
                turned = face.yaw < TURN_LEFT_BELOW if kind == "left" else face.yaw > TURN_RIGHT_ABOVE
                self.turn_frames = self.turn_frames + 1 if turned else 0
                done = self.turn_frames >= 4
            if done:
                self.idx += 1
                self.blinks = self.closed = self.turn_frames = 0
                self.ch_start = now
                if self.idx >= len(self.challenges):
                    self.state, self.final_start = "FINAL", now
                    return "Volte a olhar de frente", YELLOW, ""

        if now - self.ch_start > CHALLENGE_TIMEOUT:
            return self._fail("Tempo esgotado no desafio", now)
        sub = f"piscadas: {self.blinks}/2" if kind == "blink" else ""
        return title, YELLOW, sub

    def _final(self, frame, face, now):
        if now - self.final_start > 4.0:
            return self._fail("Nao confirmou a identidade", now)
        if not self._frontal(face):
            return "Volte a olhar de frente", YELLOW, ""
        name = self._match(frame, face)
        if name is None:
            return "Confirmando...", YELLOW, ""
        if name == self.id_name:
            self.state, self.user, self.misses = "VERIFIED", name, 0
            return f"LIBERADO: {name}", GREEN, ""
        return self._fail("Identidade diferente apos o desafio", now)

    def _verified(self, frame, faces, face):
        if len(faces) > 1 or face is None:
            self.misses += 1                                  # alguem entrou ou voce saiu
        elif abs(face.yaw - 0.5) > 0.15:
            pass                                              # olhando de lado: nao conta como falha
        else:
            self.misses = 0 if self._match(frame, face) == self.user else self.misses + 1
        if self.misses >= MISS_LIMIT:
            self.reset()
            return "Sessao encerrada", GRAY, ""
        return f"LIBERADO: {self.user}", GREEN, "modo economia: checagem a cada 0.4 s"


def main():
    db = load_db()
    if not db:
        raise SystemExit("Nenhum rosto cadastrado. Rode primeiro: python level2_enroll.py")
    cap = open_camera()
    tracker = FaceTracker(max_faces=2)
    auth = Auth(db)

    while True:
        frame = read_mirrored(cap)
        if frame is None:
            break
        faces = tracker.process(frame)
        text, color, sub = auth.update(frame, faces, time.time())
        for f in faces:
            draw_face_box(frame, f, color)
        draw_banner(frame, text, color, sub or "q / ESC = sair")
        cv2.imshow("Nivel 3 - Vivacidade", frame)
        delay = CHECK_EVERY_MS if auth.state == "VERIFIED" else 1
        if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
