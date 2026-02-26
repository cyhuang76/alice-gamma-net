# -*- coding: utf-8 -*-
"""
Wernicke's Area Engine -- Sequence Comprehension & Proto-Grammar
Phase 5.2: The Physics of Understanding

Core Philosophy:

  Wernicke's area is NOT a "language understanding module" in the
  symbolic AI sense.  It is a SEQUENCE PREDICTION ENGINE.

  Broca (Phase 4.3) plans HOW to say a concept (motor compilation).
  The Semantic Field (Phase 4.2) knows WHAT a concept IS (attractor basin).
  Wernicke knows what comes NEXT (temporal prediction).

  This is the physical basis of "understanding":
    If I say "the cat sat on the ___",
    your Wernicke area predicts "mat/chair/floor" with high probability.
    Low prediction error = "I understand".
    High prediction error = "That doesn't make sense".

Physical Model:

  1. Transition Matrix = Impedance Network Between Attractors
     P(j|i) = probability of concept j following concept i.
     This is learned from hippocampal episode sequences.
     High P(j|i) = low impedance path from i to j.
     Low P(j|i) = high impedance = "surprising" transition.

  2. Gamma_syntactic = Prediction Error
     Γ_syn = 1 - P(observed_next | current)
     Γ_syn ≈ 0 : "makes sense" (smooth comprehension)
     Γ_syn ≈ 1 : "huh?!" (surprise / confusion)
     This is N400 in neuroscience -- the ERP component that spikes
     when a semantically unexpected word appears.

  3. Sequence Parsing = Rolling State Through Attractor Space
     As concepts arrive one by one, Wernicke tracks the trajectory.
     Each transition generates a Γ_syn.
     Mean(Γ_syn) over a sequence = comprehension score.
     Low mean Γ_syn = high comprehension.

  4. Chunk Formation = Frequently Co-occurring Subsequences
     When a sequence of concepts occurs frequently,
     it forms a "chunk" -- a higher-order attractor.
     "good morning" is not good+morning, it's a single chunk.
     Physics: low average Γ_syn within chunk = it flows as one unit.

  5. Prediction = Generating the Next Concept
     Given current context (last N concepts), predict the most likely next.
     This is proto-sentence generation: Wernicke generates,
     Broca articulates, Semantic Field verifies.

Equations:
  P(j|i) = count(i→j) / count(i→*)
  Γ_syntactic(i,j) = 1 - P(j|i)
  Comprehension(seq) = 1 - mean(Γ_syntactic for each transition)
  Surprise(j|context) = -log₂(P(j|context))  [in bits]
  Chunk_score(subseq) = count(subseq) * (1 - mean_Γ_within)
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# --- Transition learning ---
TRANSITION_LEARNING_RATE = 1.0    # How much each observation increases count
TRANSITION_DECAY_RATE = 0.001     # Per-tick decay of transition counts
MIN_TRANSITION_COUNT = 0.01       # Floor for transition counts

# --- Comprehension ---
SURPRISE_THRESHOLD = 0.8          # Γ_syn > this = "what?!"
COMPREHENSION_GOOD = 0.3          # Mean Γ_syn < this = fluent understanding
N400_THRESHOLD = 0.7              # Γ_syn > this triggers N400-like response

# --- Chunking ---
CHUNK_MIN_OCCURRENCES = 3         # Minimum times a bigram must occur to be a chunk
CHUNK_MAX_LENGTH = 4              # Maximum chunk length (words)
CHUNK_GAMMA_THRESHOLD = 0.3       # Mean Γ_syn within chunk < this = chunk formed

# --- Context ---
CONTEXT_WINDOW = 5                # Number of recent concepts for prediction
MAX_VOCABULARY = 200              # Maximum concepts trackable

# --- Sequence capacity ---
MAX_STORED_SEQUENCES = 200        # Maximum stored training sequences


# ============================================================================
# TransitionMatrix - the impedance network between concepts
# ============================================================================

class TransitionMatrix:
    """
    Directed transition counts between concepts.

    Physical analogy: a network of transmission lines between attractors.
    High count(i→j) = low impedance path = "natural" transition.
    Low count(i→j) = high impedance = "surprising".

    Normalized row-wise to get probabilities.
    """

    def __init__(self):
        # counts[from_concept][to_concept] = float count
        self.counts: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.total_from: Dict[str, float] = defaultdict(float)
        self.total_transitions: int = 0

    def observe(self, from_concept: str, to_concept: str,
                weight: float = TRANSITION_LEARNING_RATE):
        """Record an observed transition."""
        self.counts[from_concept][to_concept] += weight
        self.total_from[from_concept] += weight
        self.total_transitions += 1

    def probability(self, from_concept: str, to_concept: str) -> float:
        """P(to | from) = count(from→to) / count(from→*)."""
        total = self.total_from.get(from_concept, 0.0)
        if total < 1e-10:
            return 0.0
        count = self.counts.get(from_concept, {}).get(to_concept, 0.0)
        return count / total

    def gamma_syntactic(self, from_concept: str, to_concept: str) -> float:
        """
        Syntactic reflection coefficient.

        Γ_syn = 1 - P(to | from)

        Physics: how "surprised" the system is by this transition.
        Γ_syn ≈ 0 : expected (low impedance path)
        Γ_syn ≈ 1 : totally unexpected (high impedance, N400)
        """
        return 1.0 - self.probability(from_concept, to_concept)

    def predict_next(self, from_concept: str, top_k: int = 5
                     ) -> List[Tuple[str, float]]:
        """
        Predict most likely next concepts.

        Returns:
            List of (concept_label, probability) sorted descending.
        """
        if from_concept not in self.counts:
            return []

        total = self.total_from.get(from_concept, 1e-10)
        candidates = [
            (to_c, count / total)
            for to_c, count in self.counts[from_concept].items()
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def get_all_transitions(self, from_concept: str
                            ) -> Dict[str, float]:
        """Get all transition probabilities from a concept."""
        total = self.total_from.get(from_concept, 1e-10)
        return {
            to_c: count / total
            for to_c, count in self.counts.get(from_concept, {}).items()
        }

    def decay(self):
        """Natural decay of transition counts (forgetting)."""
        dead_from = []
        for from_c in list(self.counts.keys()):
            dead_to = []
            for to_c in list(self.counts[from_c].keys()):
                self.counts[from_c][to_c] *= (1.0 - TRANSITION_DECAY_RATE)
                if self.counts[from_c][to_c] < MIN_TRANSITION_COUNT:
                    dead_to.append(to_c)
            for to_c in dead_to:
                self.total_from[from_c] -= self.counts[from_c][to_c]
                del self.counts[from_c][to_c]
            if not self.counts[from_c]:
                dead_from.append(from_c)
        for from_c in dead_from:
            del self.counts[from_c]
            del self.total_from[from_c]

    @property
    def vocabulary_size(self) -> int:
        """Number of unique concepts in the matrix."""
        concepts = set(self.counts.keys())
        for targets in self.counts.values():
            concepts.update(targets.keys())
        return len(concepts)

    def to_dict(self) -> Dict[str, Any]:
        # Top transitions for summary
        top = []
        for from_c in self.counts:
            total = self.total_from.get(from_c, 1e-10)
            for to_c, count in self.counts[from_c].items():
                top.append((from_c, to_c, count / total, count))
        top.sort(key=lambda x: x[3], reverse=True)
        return {
            "vocabulary_size": self.vocabulary_size,
            "total_transitions": self.total_transitions,
            "top_transitions": [
                {"from": t[0], "to": t[1],
                 "probability": round(t[2], 4),
                 "count": round(t[3], 2)}
                for t in top[:20]
            ],
        }


# ============================================================================
# Chunk - a frequently co-occurring concept sequence
# ============================================================================

@dataclass
class Chunk:
    """
    A chunked concept sequence that flows as a single unit.

    Physical: when concepts frequently co-occur in order,
    the impedance between them drops so low that they fuse
    into a single higher-order attractor.

    "good morning" = not two concepts, but one chunk.
    """
    concepts: Tuple[str, ...]       # Ordered concept sequence
    occurrence_count: int = 0       # How many times observed
    mean_internal_gamma: float = 0.0  # Average Γ_syn within chunk
    last_seen: float = 0.0

    @property
    def label(self) -> str:
        return "+".join(self.concepts)

    @property
    def length(self) -> int:
        return len(self.concepts)

    @property
    def is_mature(self) -> bool:
        """Chunk has been seen enough times with low internal Γ."""
        return (self.occurrence_count >= CHUNK_MIN_OCCURRENCES
                and self.mean_internal_gamma < CHUNK_GAMMA_THRESHOLD)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "concepts": list(self.concepts),
            "occurrence_count": self.occurrence_count,
            "mean_internal_gamma": round(self.mean_internal_gamma, 4),
            "is_mature": self.is_mature,
        }


# ============================================================================
# WernickeEngine - sequence comprehension and prediction
# ============================================================================

class WernickeEngine:
    """
    Wernicke's Area -- the sequence comprehension engine.

    Learns transition probabilities between concepts from hippocampal
    episode sequences.  Computes Γ_syntactic (prediction error) for
    concept sequences.  Forms chunks from frequently co-occurring patterns.

    Integration with AliceBrain:
      - hippocampus.get_concept_sequences() -> wernicke.learn_from_sequences()
      - brain.hear() -> semantic_field.recognize() -> wernicke.observe(concept)
      - wernicke.comprehend(sequence) -> comprehension score
      - wernicke.predict_next(context) -> expected next concept
      - sleep cycle -> wernicke.learn_from_sequences(hippocampus.get_sequences())

    Physical invariants:
      - Γ_syn = 1 - P(next | current) : the N400 of our system
      - Chunks = low-Γ subsequences that fuse into units
      - Understanding = consistently low Γ_syn across a sequence
    """

    def __init__(self):
        self.transitions = TransitionMatrix()
        self.chunks: Dict[str, Chunk] = {}  # key = chunk label

        # Context tracking (rolling window of recent concepts)
        self._context: List[str] = []

        # N400 history (for monitoring comprehension quality)
        self._n400_history: List[float] = []
        self._max_history = 200

        # Statistics
        self.total_observations = 0
        self.total_comprehensions = 0
        self.total_predictions = 0
        self.total_n400_events = 0

    # ------------------------------------------------------------------
    # Online Observation
    # ------------------------------------------------------------------

    def observe(self, concept_label: str) -> Dict[str, Any]:
        """
        Observe a new concept in the stream.

        Called as concepts are recognized by the semantic field.
        Updates transition matrix and context window.
        Returns the Γ_syntactic (surprise) for this transition.

        This is the online learning path:
          hear(sound) -> semantic_field.recognize() -> "cat"
          -> wernicke.observe("cat") -> Γ_syn = 0.2 (expected)

        Returns:
            {
                "gamma_syntactic": float,
                "prediction_was": str or None,
                "is_n400": bool,     # True if Γ_syn > threshold
                "context": List[str],
            }
        """
        self.total_observations += 1
        gamma_syn = 1.0
        prediction_was = None

        if self._context:
            prev = self._context[-1]
            gamma_syn = self.transitions.gamma_syntactic(prev, concept_label)
            # Record transition
            self.transitions.observe(prev, concept_label)

            # What was predicted?
            predictions = self.transitions.predict_next(prev, top_k=1)
            if predictions:
                prediction_was = predictions[0][0]

        # N400 detection
        is_n400 = gamma_syn > N400_THRESHOLD
        if is_n400:
            self.total_n400_events += 1

        # Update N400 history
        self._n400_history.append(gamma_syn)
        if len(self._n400_history) > self._max_history:
            self._n400_history = self._n400_history[-self._max_history:]

        # Update context
        self._context.append(concept_label)
        if len(self._context) > CONTEXT_WINDOW:
            self._context = self._context[-CONTEXT_WINDOW:]

        # Check for chunk formation
        self._update_chunks()

        return {
            "gamma_syntactic": gamma_syn,
            "prediction_was": prediction_was,
            "is_n400": is_n400,
            "context": list(self._context),
        }

    # ------------------------------------------------------------------
    # Batch Learning from Hippocampal Sequences
    # ------------------------------------------------------------------

    def learn_from_sequences(
        self,
        sequences: List[List[str]],
        weight: float = TRANSITION_LEARNING_RATE,
    ) -> Dict[str, Any]:
        """
        Learn transition probabilities from hippocampal episode sequences.

        Called during sleep consolidation or explicitly.

        Args:
            sequences: List of concept-label sequences from hippocampus.
            weight: Learning rate multiplier.

        Returns:
            {
                "sequences_processed": int,
                "transitions_learned": int,
                "chunks_formed": int,
            }
        """
        transitions_learned = 0

        for seq in sequences:
            for i in range(len(seq) - 1):
                self.transitions.observe(seq[i], seq[i + 1], weight)
                transitions_learned += 1

            # Check for chunks in this sequence
            self._scan_for_chunks(seq)

        return {
            "sequences_processed": len(sequences),
            "transitions_learned": transitions_learned,
            "chunks_formed": sum(1 for c in self.chunks.values() if c.is_mature),
        }

    # ------------------------------------------------------------------
    # Comprehension
    # ------------------------------------------------------------------

    def comprehend(self, sequence: List[str]) -> Dict[str, Any]:
        """
        Compute comprehension score for a concept sequence.

        Physics: roll through the sequence, computing Γ_syn at each step.
        Low mean Γ_syn = "this makes sense" (fluent comprehension).
        High mean Γ_syn = "I don't understand" (confusion).

        Returns:
            {
                "comprehension_score": float (0-1, 1=perfect understanding),
                "mean_gamma_syntactic": float,
                "per_transition": List[{from, to, gamma_syn, probability}],
                "surprises": List[{position, from, to, gamma_syn}],
                "is_comprehensible": bool,
            }
        """
        self.total_comprehensions += 1

        if len(sequence) < 2:
            return {
                "comprehension_score": 1.0,
                "mean_gamma_syntactic": 0.0,
                "per_transition": [],
                "surprises": [],
                "is_comprehensible": True,
            }

        per_transition = []
        surprises = []
        gammas = []

        for i in range(len(sequence) - 1):
            from_c = sequence[i]
            to_c = sequence[i + 1]
            g_syn = self.transitions.gamma_syntactic(from_c, to_c)
            prob = self.transitions.probability(from_c, to_c)
            gammas.append(g_syn)

            transition_info = {
                "from": from_c,
                "to": to_c,
                "gamma_syntactic": round(g_syn, 4),
                "probability": round(prob, 4),
            }
            per_transition.append(transition_info)

            if g_syn > SURPRISE_THRESHOLD:
                surprises.append({
                    "position": i,
                    "from": from_c,
                    "to": to_c,
                    "gamma_syntactic": round(g_syn, 4),
                })

        mean_gamma = float(np.mean(gammas))
        comprehension_score = 1.0 - mean_gamma

        return {
            "comprehension_score": round(comprehension_score, 4),
            "mean_gamma_syntactic": round(mean_gamma, 4),
            "per_transition": per_transition,
            "surprises": surprises,
            "is_comprehensible": mean_gamma < COMPREHENSION_GOOD,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next(
        self,
        context: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Predict the most likely next concept given context.

        If context is None, uses the internal rolling context.

        Returns:
            {
                "predictions": List[{concept, probability, gamma_syn}],
                "context_used": str,
                "entropy": float,  # bits of uncertainty
            }
        """
        self.total_predictions += 1

        ctx = context if context is not None else self._context
        if not ctx:
            return {
                "predictions": [],
                "context_used": None,
                "entropy": float('inf'),
            }

        current = ctx[-1]
        predictions = self.transitions.predict_next(current, top_k)

        # Compute entropy (uncertainty)
        all_probs = self.transitions.get_all_transitions(current)
        entropy = 0.0
        for p in all_probs.values():
            if p > 1e-10:
                entropy -= p * math.log2(p)

        return {
            "predictions": [
                {
                    "concept": label,
                    "probability": round(prob, 4),
                    "gamma_syntactic": round(1.0 - prob, 4),
                }
                for label, prob in predictions
            ],
            "context_used": current,
            "entropy": round(entropy, 4),
        }

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _update_chunks(self):
        """Check if recent context forms a chunk."""
        if len(self._context) < 2:
            return

        # Check bigrams and trigrams in the context
        for length in range(2, min(CHUNK_MAX_LENGTH + 1, len(self._context) + 1)):
            subseq = tuple(self._context[-length:])
            label = "+".join(subseq)

            if label in self.chunks:
                chunk = self.chunks[label]
                chunk.occurrence_count += 1
                chunk.last_seen = time.monotonic()
                # Update internal gamma
                gammas = []
                for i in range(len(subseq) - 1):
                    g = self.transitions.gamma_syntactic(subseq[i], subseq[i + 1])
                    gammas.append(g)
                if gammas:
                    chunk.mean_internal_gamma = float(np.mean(gammas))
            else:
                # Check if this bigram/trigram is worth tracking
                # Only create if first transition has been seen before
                if self.transitions.probability(subseq[0], subseq[1]) > 0:
                    self.chunks[label] = Chunk(
                        concepts=subseq,
                        occurrence_count=1,
                        last_seen=time.monotonic(),
                    )

    def _scan_for_chunks(self, sequence: List[str]):
        """Scan a full sequence for potential chunks."""
        for length in range(2, min(CHUNK_MAX_LENGTH + 1, len(sequence))):
            for i in range(len(sequence) - length + 1):
                subseq = tuple(sequence[i:i + length])
                label = "+".join(subseq)

                if label in self.chunks:
                    self.chunks[label].occurrence_count += 1
                    self.chunks[label].last_seen = time.monotonic()
                else:
                    self.chunks[label] = Chunk(
                        concepts=subseq,
                        occurrence_count=1,
                        last_seen=time.monotonic(),
                    )

        # Update internal gammas for all chunks
        for chunk in self.chunks.values():
            gammas = []
            for i in range(len(chunk.concepts) - 1):
                g = self.transitions.gamma_syntactic(
                    chunk.concepts[i], chunk.concepts[i + 1]
                )
                gammas.append(g)
            if gammas:
                chunk.mean_internal_gamma = float(np.mean(gammas))

    def get_mature_chunks(self) -> List[Dict[str, Any]]:
        """Return all mature chunks (frequently co-occurring, low Γ)."""
        return [
            c.to_dict() for c in self.chunks.values()
            if c.is_mature
        ]

    # ------------------------------------------------------------------
    # Integration: Sleep Consolidation with Hippocampus
    # ------------------------------------------------------------------

    def consolidate_from_hippocampus(
        self,
        hippocampus,
    ) -> Dict[str, Any]:
        """
        Pull concept sequences from hippocampus and learn transitions.

        Called during sleep cycle.

        Args:
            hippocampus: HippocampusEngine instance.

        Returns:
            Result from learn_from_sequences().
        """
        sequences = hippocampus.get_concept_sequences()
        return self.learn_from_sequences(sequences)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def tick(self):
        """Per-cognitive-cycle maintenance."""
        self.transitions.decay()

    def reset_context(self):
        """Clear the context window (e.g., at episode boundary)."""
        self._context.clear()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        mature_chunks = self.get_mature_chunks()
        return {
            "vocabulary_size": self.transitions.vocabulary_size,
            "total_observations": self.total_observations,
            "total_transitions": self.transitions.total_transitions,
            "total_comprehensions": self.total_comprehensions,
            "total_predictions": self.total_predictions,
            "total_n400_events": self.total_n400_events,
            "context_window": list(self._context),
            "n_chunks": len(self.chunks),
            "mature_chunks": len(mature_chunks),
            "top_chunks": mature_chunks[:10],
            "mean_recent_gamma": (
                round(float(np.mean(self._n400_history[-20:])), 4)
                if self._n400_history else None
            ),
            "transition_summary": self.transitions.to_dict(),
        }
