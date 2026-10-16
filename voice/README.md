# Voice Lab

Estudo de reconhecimento de voz em 4 niveis, no mesmo formato do Face Lab. Por padrao roda em CPU; opcionalmente o Whisper (transcricao) usa a GPU — veja "Whisper: modelo e GPU".

## Instalar (Python >= 3.11)

Na raiz do repositorio existe **uma unica env** para face + voice + apps
(`pyproject.toml` + `uv.lock`, geridos pelo [uv](https://docs.astral.sh/uv/)):

```
uv sync
```

- Linux: instale antes o PortAudio -> `sudo apt install libportaudio2`
- Linux: torch so CPU (evita baixar CUDA, ~2 GB), rode depois do sync:
  `uv pip install "torch==2.8" "torchaudio==2.8" --index-url https://download.pytorch.org/whl/cpu`

Na primeira execucao dos niveis 2 a 4 os modelos sao baixados (ECAPA ~80 MB, Whisper base ~140 MB) para `models/`.
Se der erro de permissao/symlink no Windows, ative o Modo Desenvolvedor ou rode o terminal como administrador.

## Usar

| Nivel | Comando | O que faz |
|---|---|---|
| 1 | `python level1_mic.py` | Captura do microfone, onda, medidor e VAD (detecta fala). `P` toca a ultima frase |
| 2 | `python level2_enroll_voice.py` | Cadastro guiado: 5 frases com contagem 3-2-1, valida volume/duracao |
| 2 | `python level2_verify_voice.py` | Fale uma frase e veja quem e, com o score de cada pessoa cadastrada |
| 3 | `python level3_voice_challenge.py` | Sorteia 4 digitos: confere o CONTEUDO (Whisper) e o LOCUTOR (ECAPA) |
| 4 | `python level4_face_voice.py` | ROSTO + VOZ: face liveness, depois digitos + voz + sincronia labial |

Opcoes em todos: `--list-devices` e `--device N` para escolher o microfone.
Nos niveis 3 e 4, o Whisper aceita `--whisper MODELO`, `--whisper-device auto|cuda|cpu`,
`--whisper-compute`, `--whisper-threads` e `--whisper-beam` (veja a proxima secao).
Os cadastros ficam em `data/voices.json` (so vetores de 192 numeros, nenhum audio).

## Whisper: modelo e GPU (opcional)

O modelo padrao e o `base` (CPU, rapido e leve). Modelos maiores transcrevem melhor os digitos,
mas pesam mais. Para escolher sem adivinhar, compare na sua maquina:

| Modelo | Tamanho aprox. | Quando usar |
|---|---|---|
| `tiny` | ~75 MB | so para testar o fluxo |
| `base` | ~140 MB | padrao, CPU |
| `small` | ~500 MB | melhor precisao, ainda leve |
| `medium` | ~1.5 GB | preciso; melhor com GPU |
| `large-v3` | ~3 GB | maxima precisao; GPU recomendada |
| `distil-large-v3` | ~1.5 GB | quase large e mais rapido (GPU) |
| `large-v3-turbo` | ~1.6 GB | otimo custo/beneficio em GPU |

Comparar velocidades (grava 6 s do microfone e transcreve com varios modelos):

```
python bench_whisper.py --seconds 6 --models tiny,base,small,medium,large-v3-turbo
python bench_whisper.py --wav minha_fala.wav --models large-v3-turbo,distil-large-v3 --whisper-device cuda --runs 3
```

Usar num nivel (ex.: nivel 3 na GPU com o turbo):

```
python level3_voice_challenge.py --whisper large-v3-turbo --whisper-device cuda
```

Na GPU o `compute_type` vira `float16` sozinho; `auto` tenta a GPU e, se faltar lib CUDA, cai para a CPU
com um aviso. As libs CUDA 12 (cublas/cudnn) instaladas via pip no venv ja sao carregadas automaticamente;
fora do venv, garanta `libcublas.so.12` e `libcudnn.so.9` no `LD_LIBRARY_PATH`.

Tambem da para fixar tudo por variaveis de ambiente, sem mexer no codigo:

```
WHISPER_MODEL=small WHISPER_DEVICE=cuda python level3_voice_challenge.py
```

## Nivel 4: requisitos

- A pasta `face_lab` precisa estar AO LADO de `voice_lab` (ou defina `FACE_LAB_DIR=/caminho/face_lab`).
- Cadastre rosto (`face_lab/level2_enroll.py`) e voz (`level2_enroll_voice.py`) com o MESMO NOME.

## Como funciona

- **VAD (webrtcvad)** separa fala de silencio e recorta cada frase; so entao rodam as partes caras.
- **Locutor (SpeechBrain ECAPA-TDNN)**: a voz vira um vetor de 192 numeros. Mesma pessoa = vetores proximos
  (similaridade por cosseno >= `VOICE_THRESHOLD`, 0.45). O score da pessoa e a media das 3 melhores amostras.
- **Desafio de digitos (nivel 3)**: digitos aleatorios na hora + transcricao (faster-whisper). Gravacao antiga nao sabe o que sera pedido.
- **Sincronia labial (nivel 4)**: mede a abertura da boca (landmarks 13/14/78/308 do MediaPipe) e correlaciona com o volume da voz
  (com folga de +-300 ms para latencias). Voz tocando num alto-falante com rosto parado reprova.
- O nivel 4 exige que o locutor seja a MESMA pessoa reconhecida pelo rosto e que o rosto continue la durante a fala.

## Ajustes (topo de `vcommon.py`)

- Voce nao e reconhecido / outros passam: `VOICE_THRESHOLD` (use o nivel 2 verify para ver os scores reais).
- Whisper erra os digitos: modelo maior (`--whisper small` ou `--whisper large-v3-turbo` na GPU) ou `MAX_DIGIT_ERRORS = 1`.
  A lista de hotwords com os numerais (`DIGIT_HOTWORDS`) e o `beam_size` tambem ajudam.
- Sincronia labial reprova voce ou aprova demais: `LIPSYNC_MIN_CORR` e `MIN_MAR_RANGE` (os valores medidos aparecem na tela).
- Microfone ruim / ambiente barulhento: `ENERGY_MIN` e o agressividade do VAD (`UtteranceDetector(aggressiveness=...)`).

## Limites (honestos)

- Voz e menos estavel que rosto: gripe, ruido e microfone diferente mexem no score. Cadastre e teste no mesmo microfone.
- O desafio de digitos barra gravacoes, mas nao uma voz clonada por IA (TTS) gerada na hora. O nivel 4 ajuda
  (a sincronia labial e o rosto vivo), mas um deepfake de video em tempo real ainda passaria.
  Proximo passo: modelo anti-spoofing de voz (ex.: AASIST / ASVspoof).
