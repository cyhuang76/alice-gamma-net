# -*- coding: utf-8 -*-
"""
Semantic Field Engine -- Concepts as Attractors in State Space
Phase 4.2: The Physics of Meaning

Core Philosophy:

  A concept is NOT a token, NOT an embedding vector learned by gradient descent.
  A concept is a RESONANT ATTRACTOR in the brain's state space.

  When you repeatedly hear the word "apple" paired with red, round, sweet,
  a basin of attraction forms in semantic space.
  Any future input that falls into this basin gets pulled toward the
  concept's fixed point -- that IS "understanding".

  Understanding = Gamma_semantic -> 0
  Confusion     = Gamma_semantic -> 1

Physical Model:

  1. State Space = N-dimensional fingerprint space
     (N = cochlear channels = 24 for auditory modality)
     Each point = a possible sensory pattern.

  2. Attractor = a resonant LC circuit at position c_i with mass M_i
     Q_i = 1 + alpha * ln(1 + M_i)   (quality factor)
     More exposure -> higher Q -> sharper discrimination.

  3. Recognition = impedance matching
     Gamma_semantic = 1 - sim(x, c_i)^Q_i
     sim=1 -> Gamma=0 (perfect resonance)
     sim=0 -> Gamma=1 (total reflection)

  4. Competitive Dynamics = gravitational field
     F_i(x) = G * M_i * (c_i - x) / ||c_i - x||^3
     Input state "rolls downhill" toward the strongest attractor.
     Winner = concept with lowest Gamma.

  5. Contrastive Learning = anti-Hebbian repulsion
     When two attractors are too similar, push apart.
     This is how the brain sharpens categorical boundaries.

  6. Multi-Modal Binding
     Each attractor stores multiple modality centroids:
       auditory -> cochlear fingerprint
       visual -> retinal fingerprint
       motor -> proprioceptive fingerprint
     Cross-modal prediction: hearing "apple" -> predicts visual pattern.

Equations:
  Gamma_semantic(i,x) = 1 - sim(x, c_i)^Q_i
  E_absorbed(i,x) = 1 - Gamma^2
  Q_i = 1 + alpha * ln(1 + M_i)
  F_grav(i,x) = G * M_i * (c_i - x) / ||c_i - x||^3
  V(x) = - sum_i  G * M_i / ||x - c_i||
  Centroid update: c_i += eta * (x - c_i) / (1 + M_i)
  Contrastive:    c_i -= beta * (c_j - c_i) / ||c_j - c_i||  when sim(c_i,c_j) > threshold
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# --- Attractor impedance ---
FIELD_Z0 = 100.0                 # Base impedance (Ohm)
FIELD_INPUT_Z = 50.0             # Input signal impedance (Ohm)

# --- Mass dynamics ---
INITIAL_MASS = 1.0               # Mass at concept birth
MASS_PER_EXPOSURE = 1.0          # Mass gained per exposure
MASS_DECAY_RATE = 0.001          # Mass decay per tick (forgetting)
MASS_FLOOR = 0.1                 # Minimum mass (never fully forgotten)

# --- Quality factor ---
SHARPNESS_ALPHA = 0.5            # Q = 1 + alpha * ln(1 + M)
                                 # Controls how fast discrimination improves

# --- Centroid learning ---
CENTROID_LEARNING_RATE = 0.1     # EMA rate for centroid updates
VARIANCE_INITIAL = 1.0           # Initial variance (high uncertainty)
VARIANCE_FLOOR = 0.01            # Minimum variance
VARIANCE_SHRINK_RATE = 0.95      # Variance *= this per exposure

# --- Competitive dynamics ---
CONTRASTIVE_STRENGTH = 0.05      # Repulsion rate between similar concepts
CONTRASTIVE_THRESHOLD = 0.85     # Push apart when similarity > this
MERGE_THRESHOLD = 0.99           # Merge attractors when similarity > this

# --- Gravitational dynamics ---
GRAVITATIONAL_CONSTANT = 1.0     # G
EVOLUTION_DAMPING = 0.8          # Damping factor for state evolution
EVOLUTION_DT = 0.1               # Time step
EVOLUTION_STEPS = 20             # Steps per recognition evolution
EVOLUTION_CONVERGENCE = 1e-6     # Convergence threshold

# --- Capacity ---
MAX_ATTRACTORS = 200             # Maximum number of concepts


# ============================================================================
# Gamma Semantic - the impedance matching equation for concepts
# ============================================================================


def gamma_semantic(similarity: float, mass: float,
                   sharpness: float = SHARPNESS_ALPHA) -> float:
    """
    Semantic reflection coefficient.

    Physical analogy: concept = LC resonator with Q proportional to mass.
    Higher Q = narrower bandwidth = steeper Gamma curve.

    Gamma = 1 - sim^Q    where Q = 1 + alpha * ln(1 + M)

    Properties:
      sim=1, any mass: Gamma = 0  (perfect resonance)
      sim=0, any mass: Gamma = 1  (no resonance at all)
      More mass: curve steepens  (more discriminative)

    Returns:
        float in [0, 1] where 0 = perfect match, 1 = no match.
    """
    similarity = float(np.clip(similarity, 0.0, 1.0))
    q_factor = 1.0 + sharpness * math.log(1.0 + max(mass, 0.0))
    gamma = 1.0 - similarity ** q_factor
    return gamma


def energy_absorbed(gamma: float) -> float:
    """Energy transfer = 1 - Gamma^2."""
    return 1.0 - gamma * gamma


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


# ============================================================================
# SemanticAttractor - a concept in state space
# ============================================================================


@dataclass
class SemanticAttractor:
    """
    A single concept -- an attractor in the brain's state space.

    Physical analogy:
      A tuned LC circuit sitting at position c (centroid) in state space.
      Its mass M determines gravitational pull and quality factor Q.
      Incoming signals that match the resonance frequency are absorbed;
      mismatched signals are reflected.

    Multi-modal:
      Each attractor stores centroids in multiple modalities.
      Hearing "apple" matches the auditory centroid;
      Seeing a red sphere matches the visual centroid.
      Both map to the SAME concept.
    """
    label: str

    # Per-modality centroids: {modality_name: centroid_vector}
    modality_centroids: Dict[str, np.ndarray] = field(default_factory=dict)

    # Per-modality exposure counts
    modality_masses: Dict[str, float] = field(default_factory=dict)

    # Total mass (sum of all modality masses)
    total_mass: float = INITIAL_MASS

    # Uncertainty radius (shrinks with exposure)
    variance: float = VARIANCE_INITIAL

    # Emotional valence (-1 = negative, 0 = neutral, +1 = positive)
    valence: float = 0.0

    # Timing
    creation_time: float = 0.0
    last_activated: float = 0.0
    activation_count: int = 0

    # ------------------------------------------------------------------
    def quality_factor(self) -> float:
        """Q = 1 + alpha * ln(1 + M) -- resonance sharpness."""
        return 1.0 + SHARPNESS_ALPHA * math.log(1.0 + self.total_mass)

    def impedance(self) -> float:
        """Z_concept = Z_0 / (1 + sqrt(M)) -- decreases with mass."""
        return FIELD_Z0 / (1.0 + math.sqrt(self.total_mass))

    # ------------------------------------------------------------------
    def gamma(self, fingerprint: np.ndarray, modality: str) -> float:
        """
        Compute Gamma_semantic for this concept given an input fingerprint.

        High similarity + high mass -> Gamma near 0 (perfect match).
        Low similarity or low mass -> Gamma near 1 (poor match).
        """
        if modality not in self.modality_centroids:
            return 1.0
        centroid = self.modality_centroids[modality]
        sim = cosine_similarity(fingerprint, centroid)
        mass = self.modality_masses.get(modality, self.total_mass)
        return gamma_semantic(sim, mass)

    def energy(self, fingerprint: np.ndarray, modality: str) -> float:
        """Energy absorbed = 1 - Gamma^2."""
        g = self.gamma(fingerprint, modality)
        return energy_absorbed(g)

    def similarity_to(self, fingerprint: np.ndarray, modality: str) -> float:
        """Raw cosine similarity to the given fingerprint."""
        if modality not in self.modality_centroids:
            return 0.0
        return cosine_similarity(fingerprint, self.modality_centroids[modality])

    # ------------------------------------------------------------------
    def absorb(self, fingerprint: np.ndarray, modality: str,
               valence_update: float = 0.0):
        """
        Learn from a new observation.

        Physics: the attractor pulls its centroid toward the new data point,
        weighted by learning rate.  Mass increases, variance decreases.
        This is how concepts get sharper and more specific over time.
        """
        fp = np.asarray(fingerprint, dtype=np.float64)

        if modality not in self.modality_centroids:
            self.modality_centroids[modality] = fp.copy()
            self.modality_masses[modality] = INITIAL_MASS
        else:
            # Exponential moving average centroid update
            alpha = CENTROID_LEARNING_RATE / (1.0 + self.modality_masses[modality] * 0.1)
            self.modality_centroids[modality] = (
                (1.0 - alpha) * self.modality_centroids[modality]
                + alpha * fp
            )
            self.modality_masses[modality] += MASS_PER_EXPOSURE

        # Update aggregates
        self.total_mass = sum(self.modality_masses.values())
        self.variance = max(VARIANCE_FLOOR,
                            self.variance * VARIANCE_SHRINK_RATE)
        self.activation_count += 1
        self.last_activated = time.monotonic()

        # Emotional valence EMA
        if valence_update != 0.0:
            self.valence = 0.9 * self.valence + 0.1 * valence_update

    # ------------------------------------------------------------------
    def force_on(self, point: np.ndarray, modality: str) -> np.ndarray:
        """
        Gravitational force exerted by this attractor on a point.

        F = G * M * (c - x) / ||c - x||^3

        Physical: massive, well-learned concepts pull harder.
        """
        if modality not in self.modality_centroids:
            return np.zeros_like(point)

        centroid = self.modality_centroids[modality]
        diff = centroid - point
        dist = float(np.linalg.norm(diff))
        if dist < 1e-10:
            return np.zeros_like(point)

        force = (GRAVITATIONAL_CONSTANT * self.total_mass * diff
                 / (dist ** 3 + 1e-10))
        return force

    def potential_at(self, point: np.ndarray, modality: str) -> float:
        """
        Gravitational potential at a point.

        V = -G * M / ||c - x||
        """
        if modality not in self.modality_centroids:
            return 0.0

        centroid = self.modality_centroids[modality]
        dist = float(np.linalg.norm(centroid - point))
        if dist < 1e-10:
            return -GRAVITATIONAL_CONSTANT * self.total_mass * 1e10
        return -GRAVITATIONAL_CONSTANT * self.total_mass / dist

    # ------------------------------------------------------------------
    def distance_to(self, other: "SemanticAttractor",
                    modality: Optional[str] = None) -> float:
        """
        Semantic distance to another attractor.
        Uses cosine distance = 1 - cosine_similarity.
        If modality is None, averages across shared modalities.
        """
        if modality is not None:
            if (modality not in self.modality_centroids or
                    modality not in other.modality_centroids):
                return 1.0
            return 1.0 - cosine_similarity(
                self.modality_centroids[modality],
                other.modality_centroids[modality],
            )

        # Average across shared modalities
        shared = set(self.modality_centroids.keys()) & set(other.modality_centroids.keys())
        if not shared:
            return 1.0
        total = 0.0
        for mod in shared:
            total += 1.0 - cosine_similarity(
                self.modality_centroids[mod],
                other.modality_centroids[mod],
            )
        return total / len(shared)

    # ------------------------------------------------------------------
    def decay(self):
        """Natural mass decay -- forgetting."""
        for mod in self.modality_masses:
            self.modality_masses[mod] = max(
                MASS_FLOOR,
                self.modality_masses[mod] * (1.0 - MASS_DECAY_RATE),
            )
        self.total_mass = sum(self.modality_masses.values())

    # ------------------------------------------------------------------
    def predict_cross_modal(self, from_modality: str, to_modality: str,
                            input_fp: np.ndarray) -> Optional[np.ndarray]:
        """
        Cross-modal prediction: given what I hear, predict what I should see.

        If the attractor matches in from_modality, return the centroid
        of to_modality.  This IS semantic understanding:
        hearing "apple" -> seeing the apple.
        """
        if from_modality not in self.modality_centroids:
            return None
        if to_modality not in self.modality_centroids:
            return None

        sim = cosine_similarity(input_fp, self.modality_centroids[from_modality])
        if sim < 0.3:
            return None

        # Scale prediction confidence by similarity and mass
        return self.modality_centroids[to_modality].copy()

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "total_mass": round(self.total_mass, 2),
            "quality_factor": round(self.quality_factor(), 3),
            "impedance": round(self.impedance(), 2),
            "variance": round(self.variance, 4),
            "valence": round(self.valence, 3),
            "modalities": list(self.modality_centroids.keys()),
            "modality_masses": {
                k: round(v, 2) for k, v in self.modality_masses.items()
            },
            "activation_count": self.activation_count,
        }


# ============================================================================
# SemanticField - the gravitational field of concepts
# ============================================================================


class SemanticField:
    """
    Semantic Field -- the brain's concept landscape.

    Physics:
      A collection of attractors (concepts) in state space.
      Each attractor creates a gravitational well.
      Input signals flow through this landscape,
      settling in the deepest accessible well = recognition.

    Key methods:
      register_concept() -- create new attractor
      absorb()           -- learn from observation
      recognize()        -- find best matching concept(s)
      evolve_state()     -- simulate attractor dynamics
      contrastive_update()  -- push apart similar concepts
      gamma_landscape()  -- compute Gamma for all concepts
      get_constellation() -- full concept map with distances
    """

    def __init__(self, max_attractors: int = MAX_ATTRACTORS):
        self.attractors: Dict[str, SemanticAttractor] = {}
        self.max_attractors = max_attractors

        # Statistics
        self.total_recognitions = 0
        self.total_absorptions = 0
        self.total_merges = 0
        self.total_contrastive_updates = 0

    # ------------------------------------------------------------------
    # Concept management
    # ------------------------------------------------------------------

    def register_concept(
        self,
        label: str,
        fingerprint: np.ndarray,
        modality: str,
        valence: float = 0.0,
    ) -> SemanticAttractor:
        """
        Register a new concept or update an existing one.

        If the label already exists, treat as new exposure.
        If not, create a new attractor at the fingerprint's position.
        """
        fp = np.asarray(fingerprint, dtype=np.float64)

        if label in self.attractors:
            self.attractors[label].absorb(fp, modality, valence)
            self.total_absorptions += 1
            return self.attractors[label]

        # Capacity management: remove weakest concept
        if len(self.attractors) >= self.max_attractors:
            weakest = min(self.attractors.values(), key=lambda a: a.total_mass)
            del self.attractors[weakest.label]

        attractor = SemanticAttractor(
            label=label,
            modality_centroids={modality: fp.copy()},
            modality_masses={modality: INITIAL_MASS},
            total_mass=INITIAL_MASS,
            variance=VARIANCE_INITIAL,
            valence=valence,
            creation_time=time.monotonic(),
            last_activated=time.monotonic(),
            activation_count=1,
        )
        self.attractors[label] = attractor
        self.total_absorptions += 1
        return attractor

    # ------------------------------------------------------------------
    def absorb(self, label: str, fingerprint: np.ndarray, modality: str,
               valence: float = 0.0):
        """Learn from observation -- update or create concept."""
        self.register_concept(label, fingerprint, modality, valence)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(
        self,
        fingerprint: np.ndarray,
        modality: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float, float]]:
        """
        Recognize input -- find best matching concepts.

        Physical process:
          Input fingerprint is broadcast to all attractors.
          Each attractor computes its Gamma (impedance mismatch).
          The attractor with lowest Gamma absorbs the most energy
          = the recognized concept.

        Returns:
            List of (label, gamma, energy_absorbed) sorted by gamma ascending.
        """
        self.total_recognitions += 1
        fp = np.asarray(fingerprint, dtype=np.float64)

        results = []
        for label, attractor in self.attractors.items():
            g = attractor.gamma(fp, modality)
            e = energy_absorbed(g)
            results.append((label, g, e))

        # Sort by gamma (lowest = best match)
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def best_match(
        self,
        fingerprint: np.ndarray,
        modality: str,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Find the single best matching concept.

        Returns:
            (label, gamma, energy) or None if no concepts registered.
        """
        results = self.recognize(fingerprint, modality, top_k=1)
        if not results:
            return None
        return results[0]

    # ------------------------------------------------------------------
    def multi_modal_recognize(
        self,
        fingerprints: Dict[str, np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[str, float, float]]:
        """
        Multi-modal recognition -- combine evidence from multiple senses.

        Physics: each modality provides an independent Gamma measurement.
        The total Gamma = RMS of per-modality Gammas (like total mismatch
        across multiple cable segments in series).

        This is how cross-modal binding works:
          hearing "apple" + seeing red sphere = two low Gammas = strong binding.
        """
        results = []
        for label, attractor in self.attractors.items():
            gammas = []
            for mod, fp in fingerprints.items():
                g = attractor.gamma(fp, mod)
                gammas.append(g)

            if not gammas:
                continue

            # RMS of per-modality Gammas
            total_gamma = math.sqrt(sum(g * g for g in gammas) / len(gammas))
            total_energy = energy_absorbed(total_gamma)
            results.append((label, total_gamma, total_energy))

        results.sort(key=lambda x: x[1])
        return results[:top_k]

    # ------------------------------------------------------------------
    # Attractor Dynamics
    # ------------------------------------------------------------------

    def evolve_state(
        self,
        initial_state: np.ndarray,
        modality: str,
        steps: int = EVOLUTION_STEPS,
        dt: float = EVOLUTION_DT,
    ) -> Dict[str, Any]:
        """
        Simulate the input state evolving through the gravitational field.

        The state "rolls downhill" in the potential landscape.
        After convergence, it sits in the basin of the winning concept.

        Physical process:
          dx/dt = F_total - damping * v
          F_total = sum of gravitational forces from all attractors
          Damping prevents oscillation.

        Returns:
            {
                "final_state": np.ndarray,
                "trajectory": List[np.ndarray],
                "settled_in": str or None (label of basin),
                "convergence_step": int,
                "final_gamma": float,
            }
        """
        x = np.asarray(initial_state, dtype=np.float64).copy()
        v = np.zeros_like(x)
        trajectory = [x.copy()]

        settled_in = None
        convergence_step = steps

        for step in range(steps):
            # Compute total force from all attractors
            total_force = np.zeros_like(x)
            for attractor in self.attractors.values():
                total_force += attractor.force_on(x, modality)

            # Damped dynamics
            v = EVOLUTION_DAMPING * v + total_force * dt
            x = x + v * dt
            trajectory.append(x.copy())

            # Check convergence
            if float(np.linalg.norm(v)) < EVOLUTION_CONVERGENCE:
                convergence_step = step + 1
                break

        # Determine which basin we settled in
        best_match = self.best_match(x, modality)
        if best_match:
            settled_in = best_match[0]
            final_gamma = best_match[1]
        else:
            final_gamma = 1.0

        return {
            "final_state": x,
            "trajectory": trajectory,
            "settled_in": settled_in,
            "convergence_step": convergence_step,
            "final_gamma": final_gamma,
        }

    # ------------------------------------------------------------------
    def gamma_landscape(
        self,
        fingerprint: np.ndarray,
        modality: str,
    ) -> Dict[str, float]:
        """
        Compute Gamma for all concepts -- the "impedance landscape".

        Returns:
            {concept_label: gamma_value} for each registered concept.
        """
        fp = np.asarray(fingerprint, dtype=np.float64)
        return {
            label: attractor.gamma(fp, modality)
            for label, attractor in self.attractors.items()
        }

    # ------------------------------------------------------------------
    def force_at(self, point: np.ndarray, modality: str) -> np.ndarray:
        """Total gravitational force at a point."""
        total = np.zeros_like(point)
        for attractor in self.attractors.values():
            total += attractor.force_on(point, modality)
        return total

    def potential_at(self, point: np.ndarray, modality: str) -> float:
        """Total gravitational potential at a point."""
        total = 0.0
        for attractor in self.attractors.values():
            total += attractor.potential_at(point, modality)
        return total

    # ------------------------------------------------------------------
    # Contrastive Learning
    # ------------------------------------------------------------------

    def contrastive_update(self):
        """
        Push apart concepts that are too similar.

        Physics: two resonators at nearly the same frequency
        interfere destructively.  Natural selection pushes one
        to shift its resonance -> sharpens categorical boundaries.

        This is anti-Hebbian learning between competing concepts.
        """
        labels = list(self.attractors.keys())
        n = len(labels)
        if n < 2:
            return

        updates_made = 0
        for i in range(n):
            for j in range(i + 1, n):
                a = self.attractors[labels[i]]
                b = self.attractors[labels[j]]

                # Check each shared modality
                shared_mods = (set(a.modality_centroids.keys()) &
                               set(b.modality_centroids.keys()))
                for mod in shared_mods:
                    sim = cosine_similarity(
                        a.modality_centroids[mod],
                        b.modality_centroids[mod],
                    )

                    if sim > MERGE_THRESHOLD:
                        # Too similar -> merge into the more massive one
                        if a.total_mass >= b.total_mass:
                            a.absorb(b.modality_centroids[mod], mod)
                            # Mark b for removal
                        else:
                            b.absorb(a.modality_centroids[mod], mod)
                        self.total_merges += 1
                    elif sim > CONTRASTIVE_THRESHOLD:
                        # Push apart -- anti-Hebbian
                        diff = a.modality_centroids[mod] - b.modality_centroids[mod]
                        dist = float(np.linalg.norm(diff))
                        if dist < 1e-10:
                            continue
                        direction = diff / dist
                        # Heavier concept moves less
                        push_a = CONTRASTIVE_STRENGTH / (1.0 + a.modality_masses.get(mod, 1.0))
                        push_b = CONTRASTIVE_STRENGTH / (1.0 + b.modality_masses.get(mod, 1.0))
                        a.modality_centroids[mod] += push_a * direction
                        b.modality_centroids[mod] -= push_b * direction
                        updates_made += 1

        self.total_contrastive_updates += updates_made

    # ------------------------------------------------------------------
    # Cross-Modal Prediction
    # ------------------------------------------------------------------

    def predict_cross_modal(
        self,
        input_fp: np.ndarray,
        from_modality: str,
        to_modality: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Cross-modal prediction: given input in one modality,
        predict the expected pattern in another modality.

        This IS the physical basis of understanding:
          hearing "apple" -> predicting what an apple looks like.

        Returns:
            {
                "concept": str,
                "predicted_fingerprint": np.ndarray,
                "gamma": float,
                "confidence": float,
            }
        """
        match = self.best_match(input_fp, from_modality)
        if match is None:
            return None

        label, gamma_in, energy_in = match
        attractor = self.attractors[label]

        predicted = attractor.predict_cross_modal(
            from_modality, to_modality, input_fp
        )
        if predicted is None:
            return None

        # Confidence = how well we matched in the source modality
        # scaled by how much experience we have in the target modality
        target_mass = attractor.modality_masses.get(to_modality, 0.0)
        confidence = energy_in * min(1.0, target_mass / 10.0)

        return {
            "concept": label,
            "predicted_fingerprint": predicted,
            "gamma": gamma_in,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Semantic Distance
    # ------------------------------------------------------------------

    def semantic_distance(
        self,
        label_a: str,
        label_b: str,
        modality: Optional[str] = None,
    ) -> float:
        """
        Distance between two concepts in semantic space.
        Uses cosine distance = 1 - similarity.
        """
        if label_a not in self.attractors or label_b not in self.attractors:
            return 1.0
        return self.attractors[label_a].distance_to(
            self.attractors[label_b], modality
        )

    def get_neighbors(
        self,
        label: str,
        modality: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find nearest neighbors to a concept in semantic space.

        Returns:
            List of (neighbor_label, distance) sorted by distance.
        """
        if label not in self.attractors:
            return []

        target = self.attractors[label]
        distances = []
        for other_label, other in self.attractors.items():
            if other_label == label:
                continue
            d = target.distance_to(other, modality)
            distances.append((other_label, d))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def tick(self):
        """
        Per-cognitive-cycle maintenance.

        1. Decay all attractor masses (forgetting)
        2. Remove dead attractors (mass below floor)
        3. Run contrastive update (push apart similar concepts)
        """
        dead = []
        for label, attractor in self.attractors.items():
            attractor.decay()
            if attractor.total_mass < MASS_FLOOR * 1.5:
                dead.append(label)

        for label in dead:
            del self.attractors[label]

        # Contrastive update (every tick for now)
        if len(self.attractors) > 1:
            self.contrastive_update()

    # ------------------------------------------------------------------
    # State & Constellation
    # ------------------------------------------------------------------

    def get_constellation(self, modality: Optional[str] = None) -> Dict[str, Any]:
        """
        Full concept constellation -- all concepts and their mutual distances.

        This is the "semantic map" of Alice's brain.
        """
        labels = list(self.attractors.keys())
        # Build pairwise distance matrix
        distances = {}
        for i, la in enumerate(labels):
            for lb in labels[i + 1:]:
                d = self.semantic_distance(la, lb, modality)
                distances[f"{la}-{lb}"] = round(d, 4)

        return {
            "n_concepts": len(self.attractors),
            "concepts": {
                label: att.to_dict() for label, att in self.attractors.items()
            },
            "pairwise_distances": distances,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_attractors": len(self.attractors),
            "total_recognitions": self.total_recognitions,
            "total_absorptions": self.total_absorptions,
            "total_merges": self.total_merges,
            "total_contrastive_updates": self.total_contrastive_updates,
            "top_concepts": [
                {"label": a.label, "mass": round(a.total_mass, 2),
                 "Q": round(a.quality_factor(), 3)}
                for a in sorted(self.attractors.values(),
                                key=lambda x: x.total_mass, reverse=True)[:10]
            ],
        }


# ============================================================================
# SemanticFieldEngine -- high-level interface
# ============================================================================


class SemanticFieldEngine:
    """
    High-level semantic field engine for integration with AliceBrain.

    Wraps SemanticField with convenient methods for:
    - Processing sensory input (fingerprint -> concept recognition)
    - Learning from labeled observations
    - Cross-modal prediction
    - Concept inventory management
    """

    def __init__(self, max_attractors: int = MAX_ATTRACTORS):
        self.field = SemanticField(max_attractors=max_attractors)

        # Track what was last recognized (for decision-making)
        self._last_recognition: Optional[Tuple[str, float, float]] = None
        self._last_modality: Optional[str] = None

    # ------------------------------------------------------------------
    def process_fingerprint(
        self,
        fingerprint: np.ndarray,
        modality: str,
        label: Optional[str] = None,
        valence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Main entry point: process a sensory fingerprint.

        If label is provided -> supervised learning (absorb into concept).
        Always does recognition (find best matching concept).

        Returns:
            {
                "best_concept": str or None,
                "gamma": float,
                "energy": float,
                "top_matches": List[(label, gamma, energy)],
                "is_novel": bool,
            }
        """
        fp = np.asarray(fingerprint, dtype=np.float64)

        # If labeled, register/absorb
        if label is not None:
            self.field.absorb(label, fp, modality, valence)

        # Recognize
        matches = self.field.recognize(fp, modality, top_k=5)
        if matches:
            best = matches[0]
            self._last_recognition = best
            self._last_modality = modality

            # Is this novel? (high Gamma = never seen before)
            is_novel = best[1] > 0.7

            return {
                "best_concept": best[0],
                "gamma": best[1],
                "energy": best[2],
                "top_matches": [
                    {"label": m[0], "gamma": round(m[1], 4),
                     "energy": round(m[2], 4)}
                    for m in matches
                ],
                "is_novel": is_novel,
            }

        return {
            "best_concept": None,
            "gamma": 1.0,
            "energy": 0.0,
            "top_matches": [],
            "is_novel": True,
        }

    # ------------------------------------------------------------------
    def predict_from_hearing(
        self,
        auditory_fp: np.ndarray,
        target_modality: str = "visual",
    ) -> Optional[Dict[str, Any]]:
        """
        Hear something -> predict what it looks like.

        The physical basis of language understanding:
        hearing a word -> resonating the matching concept attractor
        -> reading out the visual centroid = "seeing" what was described.
        """
        return self.field.predict_cross_modal(
            auditory_fp, "auditory", target_modality
        )

    # ------------------------------------------------------------------
    def tick(self):
        """Per-cognitive-cycle maintenance."""
        self.field.tick()

    def get_state(self) -> Dict[str, Any]:
        state = self.field.get_state()
        if self._last_recognition:
            state["last_recognition"] = {
                "concept": self._last_recognition[0],
                "gamma": round(self._last_recognition[1], 4),
                "modality": self._last_modality,
            }
        return state
