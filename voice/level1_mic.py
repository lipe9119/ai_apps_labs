"""Nivel 1: captura do microfone + VAD (detecta quando ha fala).

Mostra a onda dos ultimos 2 s, o medidor de volume e se voce esta falando.
Cada frase detectada e contada; aperte P para ouvir a ultima. Sair: q / ESC.
Opcoes: --list-devices | --device N
"""
import cv2
import numpy as np

import vcommon as vc


def main():
    dev = vc.parse_args("Nivel 1: microfone + VAD")
    mic = vc.Mic(dev)
    det = vc.UtteranceDetector()
    roll = vc.Rolling(2.0)
    n_utt, last_audio, speaking, rms = 0, None, False, 0.0

    while True:
        for _, b in mic.blocks():
            roll.push(b)
            rms = float(np.sqrt(np.mean(b ** 2)))
            speaking = det.feed(b)
            if det.done:
                n_utt += 1
                last_audio = det.audio
                det.reset()

        img = vc.new_canvas()
        vc.banner(img, "Nivel 1 - Microfone e VAD", vc.GREEN if speaking else vc.GRAY,
                  "q / ESC = sair   |   P = ouvir a ultima frase")
        vc.draw_wave(img, roll.buf, 60, 150, vc.GREEN if speaking else (150, 150, 150))
        vc.draw_meter(img, rms, 20, 225, W_METER, 18, speaking)
        vc.put_text(img, "FALANDO" if speaking else "silencio", (20, 280),
                    vc.GREEN if speaking else (170, 170, 170), scale=0.9)
        info = f"frases detectadas: {n_utt}"
        if last_audio is not None:
            info += f"   ultima: {len(last_audio) / vc.SR:.1f} s"
        vc.put_text(img, info, (20, 320), scale=0.6)
        cv2.imshow("Nivel 1 - Microfone", img)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p") and last_audio is not None:
            import sounddevice as sd
            sd.play(last_audio, vc.SR)

    mic.close()
    cv2.destroyAllWindows()


W_METER = vc.W - 40

if __name__ == "__main__":
    main()
