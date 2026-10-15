"""Nivel 2 (parte 2): verificacao de locutor. Fale uma frase (1 a 8 s) e veja quem o sistema acha que e.

Mostra o score (similaridade) de cada pessoa cadastrada: use isso para calibrar VOICE_THRESHOLD em vcommon.py.
Sair: q / ESC.   Opcoes: --list-devices | --device N
"""
import time
from collections import deque

import cv2
import numpy as np

import vcommon as vc

WIN = "Nivel 2 - Verificacao de voz"


def main():
    dev = vc.parse_args("Nivel 2: verificacao de locutor")
    db = vc.load_voices()
    if not db:
        raise SystemExit("Nenhuma voz cadastrada. Rode primeiro: python level2_enroll_voice.py")
    print("Carregando modelo de voz...")
    enc = vc.SpeakerEncoder()
    mic = vc.Mic(dev)
    det = vc.UtteranceDetector(end_silence=0.8, max_seconds=8.0)
    roll = vc.Rolling(2.0)
    result, history = None, deque(maxlen=4)      # result = (titulo, cor, scores, ate_quando)

    while True:
        now = time.time()
        speaking, rms = False, 0.0
        for _, b in mic.blocks():
            roll.push(b)
            rms = float(np.sqrt(np.mean(b ** 2)))
            speaking = det.feed(b) or speaking
            if det.done:
                audio, voiced = det.audio, det.voiced
                det.reset()
                if voiced < 1.0:
                    result = ("Fala muito curta", vc.GRAY, {}, now + 2.5)
                else:
                    img = vc.new_canvas()
                    vc.banner(img, "Analisando...", vc.YELLOW)
                    vc.show(WIN, img)
                    name, score, scores = vc.identify_voice(enc.embed(audio), db)
                    title = f"{name}  ({score:.2f})" if name else f"Desconhecido  ({score:.2f})"
                    history.appendleft(title)
                    result = (title, vc.GREEN if name else vc.RED, scores, time.time() + 6.0)
                mic.drain()
                break

        img = vc.new_canvas()
        if result and time.time() < result[3]:
            vc.banner(img, result[0], result[1], "q / ESC = sair")
            for i, (n, s) in enumerate(sorted(result[2].items(), key=lambda kv: -kv[1])):
                vc.put_text(img, f"{n}: {s:.2f}", (20, 250 + 24 * i), scale=0.6)
        else:
            vc.banner(img, "Fale uma frase...", vc.GREEN if speaking else vc.GRAY,
                      f"Cadastrados: {', '.join(db)}   |   limiar {vc.VOICE_THRESHOLD}")
        vc.draw_wave(img, roll.buf, 60, 130, vc.GREEN if speaking else (150, 150, 150))
        vc.draw_meter(img, rms, 20, 205, vc.W - 40, 14, speaking)
        for i, h in enumerate(history):
            vc.put_text(img, f"- {h}", (360, 250 + 24 * i), (170, 170, 170), scale=0.55)
        cv2.imshow(WIN, img)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break

    mic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
