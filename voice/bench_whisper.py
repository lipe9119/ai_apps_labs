"""Compara modelos do Whisper (velocidade x texto) na SUA maquina.

Grava um trecho do microfone (ou usa um .wav) e transcreve com varios modelos,
mostrando tempo de carga, tempo de transcricao e o texto que cada um devolveu.
Serve para escolher o modelo antes de rodar os niveis 3 e 4.

Exemplos:
  python bench_whisper.py --seconds 6
  python bench_whisper.py --wav minha_fala.wav --models tiny,base,small,medium
  python bench_whisper.py --whisper-device cuda --whisper-compute float16 --models large-v3-turbo,distil-large-v3 --runs 3
  python bench_whisper.py --models tiny,base,small --whisper-device cpu   # sem GPU

Numa RTX 3060 (12 GB) cabem large-v3 e large-v3-turbo em float16. Se aparecer
erro de cuDNN/CUDA, falta a lib do ctranslate2 (o script avisa e cai para a CPU).
"""
import argparse
import gc
import time

import numpy as np

import vcommon as vc


def record(dev, seconds):
    """Grava `seconds` do microfone e devolve float32 mono a 16 kHz."""
    mic = vc.Mic(dev)
    chunks = []
    print(f"Gravando {seconds:.0f}s... fale agora (os digitos, se possivel).")
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _, b in mic.blocks():
            chunks.append(b)
        time.sleep(0.01)
    mic.close()
    if not chunks:
        raise SystemExit("Nada gravado. O microfone esta mudo ou o dispositivo esta errado (--list-devices).")
    audio = np.concatenate(chunks).astype(np.float32)
    print(f"Gravado: {len(audio) / vc.SR:.1f}s\n")
    return audio


def load_wav(path):
    from faster_whisper.audio import decode_audio
    audio = np.asarray(decode_audio(path, sampling_rate=vc.SR), dtype=np.float32)
    print(f"Audio: {path} ({len(audio) / vc.SR:.1f}s)\n")
    return audio


def main():
    defaults = ",".join(vc.WHISPER_MODELS)
    p = argparse.ArgumentParser(description="Compara modelos do Whisper (velocidade x texto)",
                                epilog="Modelos: " + defaults)
    p.add_argument("--device", help="microfone (indice ou parte do nome)")
    p.add_argument("--list-devices", action="store_true", help="lista os microfones e sai")
    p.add_argument("--wav", help="usa um arquivo .wav em vez de gravar")
    p.add_argument("--seconds", type=float, default=6.0, help="duracao da gravacao (padrao: 6)")
    p.add_argument("--models", default="tiny,base,small,medium,large-v3-turbo",
                   help=f"modelos separados por virgula (padrao: tiny,base,small,medium,large-v3-turbo)")
    p.add_argument("--whisper-device", choices=("auto", "cuda", "cpu"), default=vc.WHISPER_DEVICE)
    p.add_argument("--whisper-compute", choices=("auto", "int8", "int8_float16", "float16", "float32"),
                   default=vc.WHISPER_COMPUTE)
    p.add_argument("--whisper-threads", type=int, default=vc.WHISPER_THREADS)
    p.add_argument("--whisper-beam", type=int, default=vc.WHISPER_BEAM)
    p.add_argument("--runs", type=int, default=2, help="transcricoes por modelo; mostra a melhor (padrao: 2)")
    a = p.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    dev = int(a.device) if a.device and a.device.isdigit() else a.device
    audio = load_wav(a.wav) if a.wav else record(dev, a.seconds)

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    print(f"{'modelo':>18} | {'carga':>7} | {'transcricao':>11} | texto")
    print("-" * 72)
    for name in models:
        try:
            t0 = time.time()
            stt = vc.Transcriber(size=name, device=a.whisper_device, compute=a.whisper_compute,
                                 threads=a.whisper_threads, beam=a.whisper_beam)
            load = time.time() - t0
            times, text = [], ""
            for _ in range(max(1, a.runs)):
                t0 = time.time()
                text = stt.transcribe(audio)
                times.append(time.time() - t0)
            del stt
            gc.collect()
            print(f"{name:>18} | {load:6.1f}s | {min(times):10.2f}s | {text or '-'}")
        except Exception as e:
            print(f"{name:>18} | falhou: {e}")


if __name__ == "__main__":
    main()
