# -*- coding: utf-8 -*-
"""Alice Smart System — Body Module (Sensory Organs + Motor Organs)"""

from alice.body.eye import AliceEye
from alice.body.ear import AliceEar
from alice.body.hand import AliceHand
from alice.body.mouth import AliceMouth
from alice.body.cochlea import (
    CochlearFilterBank,
    TonotopicActivation,
    generate_tone,
    generate_complex_tone,
    generate_noise,
    generate_vowel,
)

__all__ = [
    "AliceEye",
    "AliceEar",
    "AliceHand",
    "AliceMouth",
    # Cochlear filter bank (Phase 4.1)
    "CochlearFilterBank",
    "TonotopicActivation",
    "generate_tone",
    "generate_complex_tone",
    "generate_noise",
    "generate_vowel",
]
