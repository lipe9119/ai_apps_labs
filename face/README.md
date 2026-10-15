# Face Lab

Estudo de reconhecimento facial em 3 niveis. Tudo roda em CPU.

## Instalar (Python >= 3.11)

Na raiz do repositorio existe **uma unica env** para face + voice + apps
(`pyproject.toml` + `uv.lock`, geridos pelo [uv](https://docs.astral.sh/uv/)):

```
uv sync
```

- Na primeira sync o `dlib` e compilado a partir do fonte (uns minutos): no macOS
  `xcode-select --install` e `brew install cmake`; no Linux `apt install build-essential cmake`.
  O `dlib-bin` (wheel) cobre o modulo; o `dlib` vem do `face-recognition`.
- Prefere venv classico? `python -m venv .venv && source .venv/bin/activate && pip install -e .`
  (mesma compilacao do dlib).

## Usar

| Nivel | Comando | O que faz |
|---|---|---|
| 1 | `python level1_detect.py` | Webcam + deteccao de rostos, FPS, yaw e abertura dos olhos |
| 2 | `python level2_enroll.py` | Cadastro guiado (nome na janela, poses frente/esquerda/direita com 3-2-1) |
| 2 | `python level2_recognize.py` | Reconhece varias pessoas cadastradas |
| 3 | `python level3_liveness.py` | Reconhece + desafio aleatorio (piscar / virar). Foto e video nao passam |

Os cadastros ficam em `data/faces.json` (so vetores de 128 numeros, nenhuma foto).
Para cadastrar mais gente, rode `level2_enroll.py` de novo com outro nome.

## Como funciona

- **MediaPipe Face Mesh** acha o rosto e da 468 pontos: serve para detectar, medir a pose da cabeca (yaw) e o piscar (EAR).
- **face_recognition (dlib)** transforma o rosto em um vetor de 128 numeros. Mesma pessoa = vetores proximos.
  Se a menor distancia ate um cadastro for <= `MATCH_TOLERANCE` (0.50), e reconhecido.
- **Poses no cadastro**: o yaw e a posicao do nariz entre as bordas do rosto (0.5 = de frente). So conta 3-2-1 quando o yaw esta na faixa da pose.
- **Vivacidade (nivel 3)**: desafios sorteados na hora (piscar 2x, virar para um lado), com continuidade do rosto
  e reconfirmacao da identidade no final. O embedding (parte cara) roda ~3x/s; no modo liberado, so ~2.5 checagens/s.

## Ajustes (no topo de `common.py` e `level3_liveness.py`)

- Reconhece pouco / confunde pessoas: `MATCH_TOLERANCE` (menor = mais rigido).
- Pose nao e aceita: ajuste `LEFT_RANGE` / `RIGHT_RANGE` / `FRONT_RANGE`.
- Piscada nao conta: o limiar e adaptativo (75% do olho aberto), ajuste o `0.75` em `_identify`.

## Limites (honestos)

Webcam comum e 2D. O desafio aleatorio barra foto e video comum, mas nao um deepfake em tempo real
nem uma mascara 3D. Proximo passo: modelo de anti-spoofing passivo (ex.: Silent-Face-Anti-Spoofing, ONNX).
