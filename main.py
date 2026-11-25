import time
from src.analysis.pitch import detect_pitch_yin
from src.analysis.tuner import analyze_pitch
from src.audio.capture import AudioStreamCapture
from src.audio.spectral import analyze_frame, compute_rms
from src.config import SAMPLE_RATE


def main() -> None:
    capture = AudioStreamCapture()
    capture.start()
    print("Escuchando micrófono... Silba o toca una nota (Ctrl+C para salir).\n")

    prev_mag = None
    try:
        while True:
            samples = capture.get_latest_samples()

            # ==============================================================
            # OPCIÓN 1: Módulo 1 - Captura de audio y FFT (Ventana de Hann)
            # (Selecciona este bloque y presiona Ctrl+/ para comentarlo)
            # ==============================================================
            # frame = analyze_frame(samples, prev_magnitudes=prev_mag, smoothing=0.5)
            # prev_mag = frame.magnitudes
            # volumen_pct = min(int(frame.rms * 350), 100)
            # print(
            #     f"\r[FFT] Volumen: {volumen_pct:3d}% | Frecuencia pico: {frame.peak_freq:7.1f} Hz   ",
            #     end="",
            #     flush=True,
            # )

            # ==============================================================
            # OPCIÓN 2: Módulo 2A - Afinador cromático en tiempo real
            # (Selecciona estas líneas y presiona Ctrl+/ para descomentarlas)
            # ==============================================================
            rms = compute_rms(samples)
            volumen_pct = min(int(rms * 350), 100)
            freq, conf = detect_pitch_yin(samples, sample_rate=SAMPLE_RATE)
            reading = analyze_pitch(freq, confidence=conf)
            if reading is None:
                print(
                    f"\r[AFINADOR] Volumen: {volumen_pct:3d}% | Nota: --         | En espera de señal...              ",
                    end="",
                    flush=True,
                )
            else:
                estado = "AFINADO" if reading.in_tune else ("SUBE ▲" if reading.cents < 0 else "BAJA ▼")
                print(
                    f"\r[AFINADOR] Volumen: {volumen_pct:3d}% | Nota: {reading.note_en:3s} ({reading.note_es:4s}) | {reading.detected_freq:6.1f} Hz | {reading.cents:+5.1f} cents [{estado}]   ",
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