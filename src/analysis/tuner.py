### Módulo 2:
"""
Lógica musical del afinador: conversión de frecuencia (Hz) a nota, octava,
frecuencia objetivo y desviación en cents según afinación temperada igual.
"""

import math
from dataclasses import dataclass
from typing import Optional

NOTE_NAMES_EN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_ES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

# Presets con números MIDI de cada cuerda
INSTRUMENT_PRESETS: dict[str, Optional[list[tuple[int, str]]]] = {
    "Cromático": None,
    "Guitarra (Estándar)": [
        (40, "6.ª cuerda (Mi2)"),
        (45, "5.ª cuerda (La2)"),
        (50, "4.ª cuerda (Re3)"),
        (55, "3.ª cuerda (Sol3)"),
        (59, "2.ª cuerda (Si3)"),
        (64, "1.ª cuerda (Mi4)"),
    ],
    "Bajo (4 cuerdas)": [
        (28, "4.ª cuerda (Mi1)"),
        (33, "3.ª cuerda (La1)"),
        (38, "2.ª cuerda (Re2)"),
        (43, "1.ª cuerda (Sol2)"),
    ],
    "Ukelele (Estándar)": [
        (67, "4.ª cuerda (Sol4)"),
        (60, "3.ª cuerda (Do4)"),
        (64, "2.ª cuerda (Mi4)"),
        (69, "1.ª cuerda (La4)"),
    ],
}


@dataclass
class TunerReading:
    """Resultado completo de afinación para una frecuencia detectada."""

    detected_freq: float
    target_freq: float
    note_en: str
    note_es: str
    cents: float
    in_tune: bool
    confidence: float
    string_label: Optional[str] = None


def analyze_pitch(
    freq: float,
    confidence: float = 1.0,
    a4_ref: float = 440.0,
    preset: str = "Cromático",
) -> Optional[TunerReading]:
    """Convierte una frecuencia en Hz a la nota más cercana y su error en cents."""
    if freq <= 0.0:
        return None

    midi_float = 69.0 + 12.0 * math.log2(freq / a4_ref)
    string_label: Optional[str] = None

    strings = INSTRUMENT_PRESETS.get(preset)
    if strings:
        target_midi, string_label = min(strings, key=lambda item: abs(midi_float - item[0]))
    else:
        target_midi = int(round(midi_float))

    target_freq = a4_ref * (2.0 ** ((target_midi - 69) / 12.0))
    cents = 1200.0 * math.log2(freq / target_freq)

    note_idx = target_midi % 12
    octave = (target_midi // 12) - 1

    return TunerReading(
        detected_freq=freq,
        target_freq=target_freq,
        note_en=f"{NOTE_NAMES_EN[note_idx]}{octave}",
        note_es=f"{NOTE_NAMES_ES[note_idx]}{octave}",
        cents=cents,
        in_tune=abs(cents) <= 5.0,
        confidence=confidence,
        string_label=string_label,
    )