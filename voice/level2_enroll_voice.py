"""Nivel 2 (parte 1): cadastro guiado de voz.

1. Digite o nome na janela e aperte ENTER (use o MESMO nome do cadastro de rosto, para o nivel 4).
2. Para cada frase: contagem 3-2-1, depois leia em voz alta. O programa avisa se a fala foi curta,
   baixa demais ou saturada e repete a frase.
Sair: ESC.   Opcoes: --list-devices | --device N
"""
import time

import cv2
import numpy as np

import vcommon as vc

PHRASES = [  # sem acentos: o OpenCV nao desenha acentos
    "O rato roeu a roupa do rei de Roma",
    "Hoje o dia esta lindo e eu vou programar",
    "Tres pratos de trigo para tres tigres tristes",
    "Minha voz e a minha senha de acesso",
    "Um dois tres quatro cinco seis sete oito nove",
]
COUNTDOWN = 3.0
LISTEN_TIMEOUT = 10.0
WIN = "Nivel 2 - Cadastro de voz"


def main():
    dev = vc.parse_args("Nivel 2: cadastro de voz")
    print("Carregando modelo de voz (na primeira vez ele e baixado)...")
    enc = vc.SpeakerEncoder()
    mic = vc.Mic(dev)
    det = vc.UtteranceDetector(end_silence=0.9, max_seconds=10.0)
    roll = vc.Rolling(2.0)

    name, stage, idx, t0, hint, key = "", "name", 0, 0.0, "", 255
    samples, summary = [], ""

    while True:
        now = time.time()
        img = vc.new_canvas()
        speaking, rms = False, 0.0
        for _, b in mic.blocks():
            roll.push(b)
            rms = float(np.sqrt(np.mean(b ** 2)))
            if stage == "listen":
                speaking = det.feed(b) or speaking

        if key == 27:
            break

        if stage == "name":
            vc.banner(img, "CADASTRO DE VOZ - digite o nome e aperte ENTER", vc.GRAY, "ESC = sair")
            vc.put_text(img, "Nome: " + name + "_", (20, 180), scale=1.0)
            if key == 13 and name.strip():
                name, stage, t0 = name.strip(), "prep", now
            else:
                name = vc.edit_text(name, key)

        elif stage == "prep":
            left = np.ceil(COUNTDOWN - (now - t0))
            vc.banner(img, f"Frase {idx + 1}/{len(PHRASES)} - prepare-se", vc.GRAY, f"Cadastrando: {name}")
            vc.center_text(img, PHRASES[idx], 120, scale=0.7)
            vc.center_text(img, str(int(left)), 240, vc.YELLOW, scale=3.5, thick=6)
            if hint:
                vc.center_text(img, hint, 320, vc.YELLOW, scale=0.6)
            if now - t0 >= COUNTDOWN:
                det.reset()
                mic.drain()
                stage, t0 = "listen", now

        elif stage == "listen":
            vc.banner(img, f"Frase {idx + 1}/{len(PHRASES)} - LEIA EM VOZ ALTA",
                      vc.GREEN if speaking else vc.YELLOW, f"Cadastrando: {name}")
            vc.center_text(img, PHRASES[idx], 100, scale=0.7)
            vc.draw_wave(img, roll.buf, 140, 120, vc.GREEN if speaking else (150, 150, 150))
            vc.draw_meter(img, rms, 20, 285, vc.W - 40, 16, speaking)
            if det.done:
                vc.center_text(img, "Processando...", 335, vc.YELLOW, scale=0.7)
                vc.show(WIN, img)
                audio = det.audio
                peak, level = float(np.abs(audio).max()), float(np.sqrt(np.mean(audio ** 2)))
                if det.voiced < vc.MIN_SPEECH_S:
                    hint = "Fala muito curta: leia a frase inteira"
                elif peak >= 0.99:
                    hint = "Audio estourou: afaste-se do microfone"
                elif level < 0.01:
                    hint = "Muito baixo: fale mais alto ou aproxime-se"
                else:
                    samples.append((f"frase{idx + 1}", enc.embed(audio)))
                    idx += 1
                    hint = "Ok!"
                if idx >= len(PHRASES):
                    vc.save_voice(name, samples)
                    embs = np.array([e for _, e in samples])
                    sim = embs @ embs.T
                    cons = float((sim.sum() - len(embs)) / (len(embs) * (len(embs) - 1)))
                    summary = f"Consistencia entre amostras: {cons:.2f}"
                    if cons < 0.5:
                        summary += "  (baixa: repita num ambiente silencioso)"
                    stage, t0 = "done", now
                else:
                    stage, t0 = "prep", now
                mic.drain()
            elif now - t0 > LISTEN_TIMEOUT and not det.active:
                hint, stage, t0 = "Nao ouvi nada. Vamos de novo", "prep", now

        else:  # done
            vc.banner(img, f"Cadastro de {name} concluido!", vc.GREEN)
            vc.center_text(img, summary, 180, scale=0.65)
            if now - t0 > 4.0:
                break

        cv2.imshow(WIN, img)
        key = cv2.waitKey(30) & 0xFF

    mic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
