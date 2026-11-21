import time
from src.audio.capture import AudioStreamCapture
from src.audio.spectral import analyze_frame


def main() -> None:
    capture = AudioStreamCapture()
    capture.start()
    print("Escuchando micrófono... Silba o toca una nota (Ctrl+C para salir).\n")

    prev_mag = None
    try:
        while True:
            samples = capture.get_latest_samples()
            frame = analyze_frame(samples, prev_magnitudes=prev_mag, smoothing=0.5)
            prev_mag = frame.magnitudes

            volumen_pct = min(int(frame.rms * 400), 100)
            print(
                f"\rVolumen: {volumen_pct:3d}% | Frecuencia pico: {frame.peak_freq:7.1f} Hz   ",
                end="",
                flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()