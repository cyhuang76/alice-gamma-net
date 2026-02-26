# -*- coding: utf-8 -*-
"""Alice Smart System — Body Module (Sensory Organs + Motor Organs + Visceral Systems)"""

from alice.body.eye import AliceEye
from alice.body.ear import AliceEar
from alice.body.hand import AliceHand
from alice.body.mouth import AliceMouth
from alice.body.lung import AliceLung
from alice.body.skin import AliceSkin
from alice.body.nose import AliceNose
from alice.body.vestibular import VestibularSystem
from alice.body.interoception import InteroceptionOrgan
from alice.body.cardiovascular import CardiovascularSystem
from alice.body.cochlea import (
    CochlearFilterBank,
    TonotopicActivation,
    generate_tone,
    generate_complex_tone,
    generate_noise,
    generate_vowel,
)
# Visceral / metabolic systems (Phase 31+)
from alice.body.immune import ImmuneSystem
from alice.body.digestive import DigestiveSystem
from alice.body.endocrine import EndocrineSystem
from alice.body.kidney import KidneySystem
from alice.body.liver import LiverSystem
from alice.body.lymphatic import LymphaticSystem
from alice.body.reproductive import ReproductiveSystem

__all__ = [
    # Sensory organs
    "AliceEye",
    "AliceEar",
    "AliceHand",
    "AliceMouth",
    "AliceLung",
    "AliceSkin",
    "AliceNose",
    "VestibularSystem",
    "InteroceptionOrgan",
    "CardiovascularSystem",
    # Cochlear filter bank (Phase 4.1)
    "CochlearFilterBank",
    "TonotopicActivation",
    "generate_tone",
    "generate_complex_tone",
    "generate_noise",
    "generate_vowel",
    # Visceral / metabolic systems (Phase 31+)
    "ImmuneSystem",
    "DigestiveSystem",
    "EndocrineSystem",
    "KidneySystem",
    "LiverSystem",
    "LymphaticSystem",
    "ReproductiveSystem",
]
