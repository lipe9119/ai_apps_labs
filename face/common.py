"""Utilitarios compartilhados: camera, FaceMesh (MediaPipe), embeddings e banco de rostos."""
import json
import os
import time
from pathlib import Path

import cv2
import face_recognition
import mediapipe as mp
import numpy as np

DATA_FILE = Path(__file__).parent / "data" / "faces.json"

# ---------------------------------------------------------------- ajustes
MATCH_TOLERANCE = 0.50        # distancia maxima p/ considerar a mesma pessoa (dlib: 0.6 e o padrao, 0.5 e mais rigido)
MIN_FACE_PX = 110             # largura minima do rosto (px) p/ processar
FRONT_RANGE = (0.45, 0.55)    # faixa do "yaw" para rosto de frente
LEFT_RANGE = (0.28, 0.42)     # rosto virado p/ ESQUERDA da tela
RIGHT_RANGE = (0.58, 0.72)    # rosto virado p/ DIREITA da tela
TURN_LEFT_BELOW = 0.42        # limiar do desafio "vire p/ esquerda"
TURN_RIGHT_ABOVE = 0.58       # limiar do desafio "vire p/ direita"

# ---------------------------------------------------------------- cores (BGR)
GREEN = (60, 180, 60)
RED = (50, 50, 220)
YELLOW = (0, 170, 220)
GRAY = (90, 90, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------- indices do Face Mesh
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]
NOSE, CHEEK_L, CHEEK_R, FOREHEAD, CHIN = 1, 234, 454, 10, 152


def _ear(p):
    """Eye Aspect Ratio: ~0.3 olho aberto, <0.2 fechado."""
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return v / (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6)


class Face:
    """Um rosto detectado, com metricas derivadas dos landmarks (coordenadas em pixels)."""

    def __init__(self, pts):
        self.pts = pts
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        self.bbox = (int(x1), int(y1), int(x2), int(y2))

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @property
    def yaw(self):
        """~0.5 = de frente; <0.5 = nariz p/ esquerda da tela; >0.5 = p/ direita."""
        p = self.pts
        return float((p[NOSE][0] - p[CHEEK_L][0]) / (p[CHEEK_R][0] - p[CHEEK_L][0] + 1e-6))

    @property
    def pitch(self):
        p = self.pts
        return float((p[NOSE][1] - p[FOREHEAD][1]) / (p[CHIN][1] - p[FOREHEAD][1] + 1e-6))

    @property
    def ear(self):
        return float((_ear(self.pts[R_EYE]) + _ear(self.pts[L_EYE])) / 2.0)


class FaceTracker:
    """MediaPipe Face Mesh: roda so em CPU, entrega 468 pontos por rosto."""

    def __init__(self, max_faces=2):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        faces = []
        for fl in res.multi_face_landmarks or []:
            pts = np.array([[l.x, l.y] for l in fl.landmark]) * np.array([w, h])
            faces.append(Face(pts))
        return faces


# ---------------------------------------------------------------- camera
def open_camera(index=0, width=640, height=480):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir a webcam (indice {index}).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_mirrored(cap):
    """Le um frame ja espelhado (como um espelho), para a interface ficar natural."""
    ok, frame = cap.read()
    return cv2.flip(frame, 1) if ok else None


# ---------------------------------------------------------------- embeddings / banco
def embedding(frame_bgr, face):
    """Vetor de 128 numeros que representa o rosto (usa a caixa do MediaPipe, sem rodar o detector do dlib)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox
    y1 = y1 + int(0.12 * (y2 - y1))  # a caixa do mesh inclui a testa; o dlib espera algo mais proximo das sobrancelhas
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w - 1), min(y2, h - 1)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb, known_face_locations=[(y1, x2, y2, x1)], num_jitters=1)
    return encs[0] if encs else None


def load_db():
    """Retorna {nome: [embeddings]}."""
    if not DATA_FILE.exists():
        return {}
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return {n: [np.array(s["embedding"]) for s in samples] for n, samples in raw.items()}


def save_person(name, samples):
    """samples: lista de (pose, embedding). Substitui o cadastro anterior com o mesmo nome."""
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8")) if DATA_FILE.exists() else {}
    raw[name] = [{"pose": p, "embedding": e.tolist()} for p, e in samples]
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(raw), encoding="utf-8")


def identify(emb, db):
    """Retorna (nome ou None, menor distancia)."""
    best_name, best = None, 9.0
    for name, embs in db.items():
        d = float(np.min(face_recognition.face_distance(embs, emb)))
        if d < best:
            best_name, best = name, d
    if best_name is not None and best <= MATCH_TOLERANCE:
        return best_name, best
    return None, best


# ---------------------------------------------------------------- desenho (OpenCV so aceita ASCII!)
def draw_text(img, text, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_banner(img, text, color=GRAY, sub=None):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 48), color, -1)
    draw_text(img, text, (12, 33), scale=0.75)
    if sub:
        draw_text(img, sub, (12, h - 14), scale=0.6)


def draw_face_box(img, face, color, label=None):
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        draw_text(img, label, (x1, max(y1 - 8, 60)), color=color, scale=0.65)


class FPS:
    def __init__(self):
        self.t = time.time()
        self.value = 0.0

    def tick(self):
        now = time.time()
        dt = now - self.t
        self.t = now
        if dt > 0:
            self.value = 0.9 * self.value + 0.1 * (1.0 / dt)
        return self.value
