# ai_apps_labs

Laboratorio pessoal de IA em tres partes: **reconhecimento facial**, **reconhecimento de voz** e
**aplicativos de desktop** construidos em cima deles. Tudo roda em **CPU** (sem GPU), com niveis
progressivos e comentarios explicando o que acontece em cada passo.

```bash
uv sync
uv run python apps/dictation_app.py   # ditado com transcricao ao vivo
```

## Estrutura

```
face/    Face Lab  - reconhecimento facial em 3 niveis (webcam)
voice/   Voice Lab - reconhecimento de voz em 4 niveis (microfone)
apps/    aplicativos de verdade feitos em cima dos labs
```

Cada pasta tem o proprio `README.md` com detalhes de implementacao, ajustes finos e limites
honestos. Este aqui e so o mapa geral.

## Instalacao

Requisitos: **Python >= 3.11** e o [**uv**](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lipe9119/ai_apps_labs.git
cd ai_apps_labs
uv sync
```

- No Linux, instale antes o PortAudio: `sudo apt install libportaudio2`.
- Os modelos baixam na primeira execucao: Whisper `base` (~140 MB) e ECAPA (~80 MB) para `voice/models/`.
- Prefere venv classico? `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- Microfone com varios aparelhos: todos os scripts aceitam `--list-devices` e `--device N`.

## Face Lab (`face/`)

| Nivel | Comando | O que faz |
|---|---|---|
| 1 | `uv run python face/level1_detect.py` | Webcam + deteccao de rostos, FPS, yaw e abertura dos olhos |
| 2 | `uv run python face/level2_enroll.py` | Cadastro guiado (nome + poses frente/esquerda/direita, 3-2-1) |
| 2 | `uv run python face/level2_recognize.py` | Reconhece varias pessoas cadastradas |
| 3 | `uv run python face/level3_liveness.py` | Reconhecimento + desafio anti-foto (piscar / virar) |

MediaPipe Face Mesh acha o rosto e os 468 pontos (pose e piscar); `face_recognition` (dlib) vira o
rosto num **vetor de 128 numeros**. Cadastros em `face/data/faces.json` - so numeros, nenhuma foto.

## Voice Lab (`voice/`)

| Nivel | Comando | O que faz |
|---|---|---|
| 1 | `uv run python voice/level1_mic.py` | Captura, onda, medidor e VAD (detecta fala) |
| 2 | `uv run python voice/level2_enroll_voice.py` | Cadastro guiado: 5 frases com contagem 3-2-1 |
| 2 | `uv run python voice/level2_verify_voice.py` | Fale e veja quem e, com o score de cada pessoa |
| 3 | `uv run python voice/level3_voice_challenge.py` | Digitos aleatorios: conteudo (Whisper) + locutor (ECAPA) |
| 4 | `uv run python voice/level4_face_voice.py` | Rosto + voz juntos: liveness, digitos e sincronia labial |

VAD (webrtcvad) separa fala de silencio e recorta frases; SpeechBrain ECAPA-TDNN vira a voz num
**vetor de 192 numeros**; `faster-whisper` transcreve. Cadastros em `voice/data/voices.json`.
O nivel 4 combina com o Face Lab: cadastre rosto e voz com o **mesmo nome**.

## Apps (`apps/`)

| App | O que faz |
|---|---|
| `uv run python apps/dictation_app.py` | **Ditado**: aperte, fale, o texto aparece na tela ao vivo e salva `.txt` + `.wav` + `.json` (com os timestamps de cada trecho) |

App de desktop em PySide6 (tema escuro, waveform e medidor desenhados no Qt). A transcricao roda
numa thread separada da do microfone, entao a janela nunca trava; ao parar, ele espera toda a fila
de transcricao terminar antes de fechar os arquivos. Detalhes e opcoes em `apps/README.md`.

## Como o codigo e organizado

- **Niveis como scripts avulsos**: cada `levelN_*.py` roda sozinho; e a melhor porta de entrada.
- **Bibliotecas compartilhadas**: `face/common.py` e `voice/vcommon.py` (mic, VAD, modelos, tema).
- **Cadastro so com numeros**: JSONs guardam vetores, nunca fotos nem audio.
- **Ajustes no topo dos arquivos**: tolerancias, limiares e modelos ficam em constantes comentadas.
- **Testes**: `uv run python apps/dictation_app.py --selftest` valida interface + threads + modelo sem usar audio.

## Limites (honestos)

Sao labs de estudo, nao produtos: webcam 2D nao para deepfake em tempo real; voz e sensivel a
microfone e ruido; gravacao/clonagem por TTS passa pelos desafios simples. Os READMEs de cada pasta
detalham o que ja foi coberto e qual seria o proximo passo (ex.: anti-spoofing passivo com ONNX).
