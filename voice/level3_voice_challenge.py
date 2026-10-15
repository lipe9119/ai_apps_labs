"""Nivel 3: verificacao de voz com desafio aleatorio (anti-gravacao).

O sistema sorteia digitos na hora. Voce precisa:
  1) dizer EXATAMENTE esses digitos (Whisper transcreve e confere o conteudo), e
  2) a voz tem que ser de alguem cadastrado (ECAPA confere o locutor).
Uma gravacao antiga nao sabe quais digitos serao pedidos.
ENTER/ESPACO = novo desafio. Sair: q / ESC.   Opcoes: --list-devices | --device N
"""
import time

import cv2
import numpy as np

import vcommon as vc

WIN = "Nivel 3 - Desafio de voz"
PREP_S = 1.5
LISTEN_TIMEOUT = 10.0
RESULT_S = 5.0


def analyze(audio, voiced, digits, enc, stt, db):
    """Retorna (liberado, titulo, linhas[(ok, texto)])."""
    text = stt.transcribe(audio)
    heard = vc.digits_from_text(text)
    errs = vc.digit_errors(digits, heard)
    ok_content = errs <= vc.MAX_DIGIT_ERRORS
    name, score, _ = vc.identify_voice(enc.embed(audio), db)
    ok_voice = name is not None
    lines = [
        (ok_content, f"Conteudo: pedido {' '.join(map(str, digits))} | ouvi {' '.join(map(str, heard)) or '-'}"),
        (ok_voice, f"Voz: {name or 'desconhecida'} (score {score:.2f}, minimo {vc.VOICE_THRESHOLD})"),
    ]
    if voiced < 1.0:
        return False, "ACESSO NEGADO - fala muito curta", lines
    if ok_content and ok_voice:
        return True, f"LIBERADO: {name}", lines
    why = "conteudo e voz" if not ok_content and not ok_voice else ("conteudo errado" if not ok_content else "voz nao reconhecida")
    return False, f"ACESSO NEGADO - {why}", lines


def main():
    dev = vc.parse_args("Nivel 3: desafio de voz")
    db = vc.load_voices()
    if not db:
        raise SystemExit("Nenhuma voz cadastrada. Rode primeiro: python level2_enroll_voice.py")
    print("Carregando modelos (a primeira vez baixa os modelos, pode demorar)...")
    enc, stt = vc.SpeakerEncoder(), vc.Transcriber()
    mic = vc.Mic(dev)
    det = vc.UtteranceDetector(end_silence=0.9, max_seconds=10.0)
    roll = vc.Rolling(2.0)

    stage, t0, digits, result, key = "idle", 0.0, [], None, 255

    while True:
        now = time.time()
        img = vc.new_canvas()
        speaking, rms = False, 0.0
        for _, b in mic.blocks():
            roll.push(b)
            rms = float(np.sqrt(np.mean(b ** 2)))
            if stage == "listen":
                speaking = det.feed(b) or speaking

        if key in (ord("q"), 27):
            break
        if key in (13, 32) and stage == "idle":
            digits, stage, t0 = vc.random_digits(), "prep", now

        if stage == "idle":
            vc.banner(img, "Nivel 3 - Desafio de voz", vc.GRAY, f"Cadastrados: {', '.join(db)}   |   q / ESC = sair")
            vc.center_text(img, "ENTER ou ESPACO para iniciar", 180, scale=0.9)

        elif stage in ("prep", "listen"):
            big = "   ".join(map(str, digits))
            if stage == "prep":
                vc.banner(img, "Prepare-se...", vc.YELLOW)
                if now - t0 >= PREP_S:
                    det.reset()
                    mic.drain()
                    stage, t0 = "listen", now
            else:
                vc.banner(img, "DIGA OS NUMEROS EM VOZ ALTA", vc.GREEN if speaking else vc.YELLOW)
            vc.center_text(img, big, 130, scale=2.4, thick=5)
            vc.draw_wave(img, roll.buf, 170, 110, vc.GREEN if speaking else (150, 150, 150))
            vc.draw_meter(img, rms, 20, 300, vc.W - 40, 16, speaking)

            if stage == "listen":
                if det.done:
                    vc.banner(img, "Analisando...", vc.YELLOW)
                    vc.show(WIN, img)
                    result = analyze(det.audio, det.voiced, digits, enc, stt, db) + (time.time() + RESULT_S,)
                    stage = "result"
                    mic.drain()
                elif not det.active and now - t0 > LISTEN_TIMEOUT:
                    result = (False, "ACESSO NEGADO - nenhuma fala detectada", [], now + RESULT_S)
                    stage = "result"

        else:  # result
            ok, title, lines, until = result
            vc.banner(img, title, vc.GREEN if ok else vc.RED)
            for i, (good, text) in enumerate(lines):
                vc.put_text(img, ("[OK] " if good else "[X]  ") + text, (20, 120 + 40 * i),
                            vc.GREEN if good else vc.RED, scale=0.55)
            if now >= until:
                stage = "idle"

        cv2.imshow(WIN, img)
        key = cv2.waitKey(30) & 0xFF

    mic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
