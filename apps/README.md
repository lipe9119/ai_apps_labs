# Apps

Ferramentas uteis (nao sao niveis de aprendizado). Usam o codigo de `voice/` (mic + Whisper)
e/ou `face/`.

## Ditado (`dictation_app.py`)

App de desktop (PySide6) que transcreve sua fala ao vivo e salva o resultado.

- Aperte **ESPACO / ENTER** (ou clique em **GRAVAR**) e comece a falar.
- O texto aparece na tela enquanto voce fala: o trecho em andamento fica em
  italico/cinza (provisorio) e vira definitivo quando voce faz uma pausa.
  Cada trecho definitivo ja e gravado no `.txt` na hora (nada se perde se o app fechar).
  O selo **transcrevendo…** (ao lado do status) acende enquanto o Whisper trabalha.
- Aperte de novo (ou **ESC**) para parar. Ao parar, o status mostra
  **"Finalizando..."** e o app espera **toda a fila de transcricao** terminar
  antes de fechar os arquivos — nada do que voce ja falou se perde, mesmo que
  voce aperte parar logo apos terminar de falar.

Cada gravacao gera 3 arquivos em `apps/data/transcricoes/` (mude com `--out`):

| Arquivo | Conteudo |
|---|---|
| `AAAA-MM-DD_HHMMSS.txt` | a transcricao |
| `AAAA-MM-DD_HHMMSS.wav` | o audio (16 kHz mono) |
| `AAAA-MM-DD_HHMMSS.json` | inicio, duracao, modelo e o tempo de cada trecho |

```
python apps/dictation_app.py                 # abre o app
python apps/dictation_app.py --list-devices  # lista os microfones
python apps/dictation_app.py --device 2 --model base
```

Opcoes: `--device N|nome`, `--model tiny|base|small`, `--lang pt`, `--out DIR`.

Detalhes de implementacao — sao **duas threads alem da interface**, de proposito:

- `Worker` (thread do microfone): captura, VAD, waveform e escrita dos arquivos.
  Roda a ~50 Hz e **nunca espera pelo Whisper**.
- `Stt` (thread do Whisper): carrega o modelo e transcreve. Recebe trabalhos por
  fila: os trechos fechados entram numa fila FIFO (nunca sao descartados) e o
  texto provisorio usa um unico slot (so o mais recente vale, o antigo e
  substituido). O `Worker` so enfileira e segue lendo o microfone.

Enquanto a fala continua, o app pede um texto provisorio a cada ~1,2 s (com
"back-pressure": se o anterior ainda nao voltou, nao pede outro); quando ha uma
pausa de ~0,7 s, o trecho e transcrito de forma definitiva e anexado ao arquivo.
Assim o custo fica limitado (nunca reescreve minutos de audio) e o display fica
suave mesmo com o modelo `base` demorando ~1,3 s por transcricao (medido: o loop
do microfone roda a ~50/s mesmo durante a transcricao; com a versao que
transcrevia dentro do loop, o mesmo cenario caia para 0 chamadas/s - travando
waveform e medidor).

Ao parar, o `stop_rec` espera **toda a fila** de commits (cada trabalho tem um
id; espera pelo ultimo, que so termina depois de todos os anteriores, ja que a
fila e FIFO) dentro de um `QEventLoop` aninhado - os resultados continuam
chegando e gravando durante a espera, sem congelar a interface. Sem essa espera,
trechos cujo texto ainda estava na fila perdiam a corrida contra o `close()` dos
arquivos e sumiam do `.txt` (reproduzido e corrigido: parar logo apos falar
deixava o .txt sem a ultima frase).

Testes: `python apps/dictation_app.py --selftest` roda sem audio (offscreen),
carrega o modelo e transcreve um trecho mudo para validar as duas threads.
