# -*- coding: utf-8 -*-
"""
Hippocampus Engine -- Episodic Memory as Temporal Binding Across Membranes
Phase 5.1: The Physics of Experience

Core Philosophy:

  The hippocampus is NOT a "memory storage unit".
  It is a TEMPORAL BINDING ENGINE that stitches snapshots from
  different sensory membranes into coherent episodes.

  Each sensory membrane lives in its own dimensional space:
    - Auditory: 32-dim (cochlear ERB constraint)
    - Visual:   256-dim (retinal FFT resolution)
    - Motor:    variable (proprioceptive channels)

  These membranes CANNOT directly interact (different dimensions).
  The semantic field's attractors are the "wormholes" between membranes.
  But attractors are TIMELESS -- they store what "cat" means, not what
  "that particular cat did at 3pm yesterday".

  The hippocampus adds the MISSING DIMENSION: TIME.

  An episode = a sequence of attractor activations stamped with time:
    t=0.0: see(cat_fingerprint) -> attractor "cat" (visual membrane)
    t=0.2: hear(meow_fingerprint) -> attractor "cat" (auditory membrane)
    t=0.5: feel(fur_fingerprint) -> attractor "soft" (tactile membrane)

  Replay = re-activating this sequence -> re-exciting the attractors
  -> re-experiencing the episode (this IS what "remembering" feels like).

Physical Model:

  1. EpisodicSnapshot = a single moment frozen in time
     {timestamp, modality, fingerprint, attractor_label, gamma, valence}
     This is one "frame" of experience.

  2. Episode = a time-ordered sequence of snapshots
     Like a filmstrip, but each frame can be from a different membrane.
     Physical invariant: timestamps are monotonically increasing.

  3. Encoding = live recording
     As sensory events arrive (see, hear, touch), the hippocampus
     records each snapshot into the current episode.
     Episode boundary = a significant pause (temporal gap > threshold).

  4. Pattern Completion = cue-driven recall
     Given a partial cue (a fingerprint in ANY modality),
     find episodes containing a matching attractor activation.
     This is how "hearing a melody" can bring back a whole visual scene.
     Physics: cue -> attractor match -> retrieve full episode timeline.

  5. Replay = temporal reactivation
     Iterating through an episode's snapshots in order.
     During sleep (N3 + REM), replay strengthens semantic attractors
     = memory consolidation (Tononi's synaptic homeostasis).

  6. Capacity Model = LTP/LTD dynamics
     Recent episodes are "hot" (high activation).
     Old episodes decay (LTD = long-term depression).
     Consolidation transfers episode patterns -> semantic field (LTP).
     This is why you forget the specific episode but remember the concept.

Equations:
  Recency(episode) = exp(-lambda * (t_now - t_last_replay))
  Relevance(episode, cue) = max_i sim(cue, snapshot_i.fingerprint)
  Retrieval_strength = Recency * Relevance
  Consolidation: for each snapshot in episode:
      semantic_field.absorb(label, fingerprint, modality)
      -> attractor mass increases -> concept becomes more permanent
      -> episode itself can then be safely forgotten
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

# --- Capacity ---
MAX_EPISODES = 100               # Short-term episodic buffer
MAX_SNAPSHOTS_PER_EPISODE = 50   # Frames per episode
EPISODE_GAP_THRESHOLD = 2.0      # Seconds of silence -> new episode

# --- Decay ---
RECENCY_LAMBDA = 0.01            # Exponential decay rate
RETRIEVAL_THRESHOLD = 0.3        # Minimum retrieval strength to recall
DECAY_FLOOR = 0.01               # Minimum recency (never fully lost)

# --- Consolidation ---
CONSOLIDATION_BOOST = 0.5        # Extra mass per consolidated snapshot
REPLAY_RECENCY_REFRESH = 0.5     # How much replay refreshes recency

# --- Pattern completion ---
COMPLETION_SIM_THRESHOLD = 0.4   # Minimum similarity for cue matching

# --- Emotional weighting ---
EMOTIONAL_MEMORY_BOOST = 2.0     # High-valence episodes decay slower


# ============================================================================
# Utility - safe cosine similarity (handles different dimensions)
# ============================================================================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Returns 0.0 if dimensions don't match."""
    if a.shape != b.shape:
        return 0.0
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


# ============================================================================
# EpisodicSnapshot - one frozen moment
# ============================================================================

@dataclass
class EpisodicSnapshot:
    """
    A single moment of experience.

    Physical: one "frame" captured from a specific sensory membrane.
    The fingerprint lives in that membrane's dimensional space.
    The attractor_label is the wormhole connecting it to other membranes.
    """
    timestamp: float                    # Absolute time (monotonic)
    modality: str                       # Which membrane: "visual", "auditory", ...
    fingerprint: np.ndarray             # Raw sensory data (membrane-native dims)
    attractor_label: Optional[str]      # Concept recognized (cross-membrane ID)
    gamma: float = 1.0                  # Recognition quality (0=perfect, 1=novel)
    valence: float = 0.0               # Emotional charge (-1 to +1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "modality": self.modality,
            "fingerprint_dim": self.fingerprint.shape[0],
            "attractor_label": self.attractor_label,
            "gamma": round(self.gamma, 4),
            "valence": round(self.valence, 3),
        }


# ============================================================================
# Episode - a coherent experience
# ============================================================================

@dataclass
class Episode:
    """
    A coherent sequence of multi-modal snapshots forming an experience.

    Physical: a filmstrip where each frame can come from a different
    sensory membrane (different dimensions), but they are bound together
    by temporal proximity and attractor co-activation.

    The episode_id is a unique identifier.
    Modalities spanned tells you which membranes were active.
    """
    episode_id: int
    snapshots: List[EpisodicSnapshot] = field(default_factory=list)
    creation_time: float = 0.0
    last_replay_time: float = 0.0
    replay_count: int = 0
    is_open: bool = True           # Still recording?
    emotional_peak: float = 0.0    # Peak |valence| in this episode
    consolidated: bool = False     # Has been transferred to semantic field?

    # ------------------------------------------------------------------
    def add_snapshot(self, snapshot: EpisodicSnapshot):
        """Record a new moment."""
        if len(self.snapshots) >= MAX_SNAPSHOTS_PER_EPISODE:
            return  # Episode full
        self.snapshots.append(snapshot)
        # Track emotional peak
        if abs(snapshot.valence) > abs(self.emotional_peak):
            self.emotional_peak = snapshot.valence

    # ------------------------------------------------------------------
    @property
    def avg_binding_gamma(self) -> float:
        """
        ★ Average binding impedance mismatch — physical quality of episodic memory

        Each snapshot's gamma represents the recognition quality at that instant.
        Higher average gamma = worse cross-modal binding for this episode = faster decay.

        Physical analogy: TV picture and audio out of sync → impedance mismatch → unstable memory fragments

        Empty episode → gamma=0.0 (no signal = no measurable mismatch)
        """
        if not self.snapshots:
            return 0.0  # No signal → cannot measure impedance → assume neutral
        return float(np.mean([s.gamma for s in self.snapshots]))

    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        """Episode duration in seconds."""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[-1].timestamp - self.snapshots[0].timestamp

    @property
    def n_snapshots(self) -> int:
        return len(self.snapshots)

    @property
    def modalities_spanned(self) -> set:
        """Which sensory membranes participated in this episode."""
        return {s.modality for s in self.snapshots}

    @property
    def concepts_mentioned(self) -> set:
        """Which attractor labels were activated."""
        return {s.attractor_label for s in self.snapshots
                if s.attractor_label is not None}

    # ------------------------------------------------------------------
    def recency(self, t_now: float) -> float:
        """
        How "fresh" is this episode?

        Physics: exponential decay modulated by TWO factors:
          1. Emotional boost: high-valence episodes decay slower (flashbulb memory)
          2. ★ Impedance modulation: poorly-bound episodes decay faster

        The impedance effect comes from the same coaxial cable physics:
          λ_eff = λ_base / (1 - Γ_bind²)

        A TV scene where audio and video are out of sync (Γ→1)
        will be forgotten almost immediately, because the signal
        was mostly reflected at the binding interface.

        "The so-called short-term memory decay gradient is a mechanism that cannot be calibrated."
        """
        t_ref = max(self.last_replay_time, self.creation_time)
        dt = max(0.0, t_now - t_ref)

        # Emotional boost: high-valence episodes are more durable
        emotional_factor = 1.0 + abs(self.emotional_peak) * EMOTIONAL_MEMORY_BOOST

        # ★ Impedance factor: poorly-bound episodes decay faster
        # impedance_factor = 1 / (1 - Γ_avg²)
        gamma_sq = min(self.avg_binding_gamma ** 2, 0.99)
        impedance_factor = 1.0 / (1.0 - gamma_sq)

        # Combined effective lambda:
        # - emotional boost DECREASES lambda (slower decay)
        # - impedance mismatch INCREASES lambda (faster decay)
        effective_lambda = RECENCY_LAMBDA * impedance_factor / emotional_factor

        return max(DECAY_FLOOR, math.exp(-effective_lambda * dt))

    # ------------------------------------------------------------------
    def relevance_to_cue(self, cue_fp: np.ndarray, cue_modality: str) -> float:
        """
        How relevant is this episode to a given cue?

        Pattern completion: find the best-matching snapshot.
        Only compares within the same modality (same membrane = same dims).
        """
        best_sim = 0.0
        for snap in self.snapshots:
            if snap.modality != cue_modality:
                continue
            sim = _cosine_sim(cue_fp, snap.fingerprint)
            if sim > best_sim:
                best_sim = sim
        return best_sim

    def relevance_to_concept(self, concept_label: str) -> float:
        """
        How relevant is this episode to a concept label?

        This is cross-membrane: concept labels are the wormholes.
        """
        if concept_label in self.concepts_mentioned:
            # Weighted by how many times this concept appeared
            count = sum(1 for s in self.snapshots
                        if s.attractor_label == concept_label)
            return min(1.0, count / max(1, len(self.snapshots)) * 3.0)
        return 0.0

    # ------------------------------------------------------------------
    def retrieval_strength(self, t_now: float,
                           cue_fp: Optional[np.ndarray] = None,
                           cue_modality: Optional[str] = None,
                           cue_concept: Optional[str] = None) -> float:
        """
        Combined retrieval strength = Recency * Relevance.

        Physics: both temporal proximity and content similarity
        contribute to whether this episode gets recalled.
        """
        rec = self.recency(t_now)

        # Content relevance
        rel = 0.0
        if cue_fp is not None and cue_modality is not None:
            rel = max(rel, self.relevance_to_cue(cue_fp, cue_modality))
        if cue_concept is not None:
            rel = max(rel, self.relevance_to_concept(cue_concept))

        if rel < 1e-6:
            # No content match -> recency-only (for "what just happened?")
            return rec * 0.1

        return rec * rel

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "n_snapshots": self.n_snapshots,
            "duration": round(self.duration, 3),
            "modalities": list(self.modalities_spanned),
            "concepts": list(self.concepts_mentioned),
            "emotional_peak": round(self.emotional_peak, 3),
            "replay_count": self.replay_count,
            "consolidated": self.consolidated,
            "is_open": self.is_open,
        }


# ============================================================================
# HippocampusEngine - temporal binding across membranes
# ============================================================================

class HippocampusEngine:
    """
    Hippocampus -- the temporal binding engine.

    Stitches multi-modal sensory snapshots into episodes (experiences).
    Provides cue-driven recall (pattern completion).
    Supports replay for consolidation (sleep-time transfer to semantic field).

    Integration with AliceBrain:
      - brain.see() / brain.hear() -> hippocampus.record(...)
      - brain.recall(cue) -> hippocampus.recall(cue)
      - sleep cycle -> hippocampus.consolidate(semantic_field)

    Physical invariants:
      - Snapshots within an episode are time-ordered
      - Cross-membrane binding goes through attractor labels (wormholes)
      - Recency * Relevance determines retrieval strength
      - Consolidation = episode -> semantic field mass transfer
    """

    def __init__(self, max_episodes: int = MAX_EPISODES):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes
        self._next_episode_id = 0
        self._current_episode: Optional[Episode] = None
        self._last_record_time: float = -1.0  # sentinel: no prior record

        # Statistics
        self.total_snapshots_recorded = 0
        self.total_episodes_created = 0
        self.total_recalls = 0
        self.total_replays = 0
        self.total_consolidations = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        modality: str,
        fingerprint: np.ndarray,
        attractor_label: Optional[str] = None,
        gamma: float = 1.0,
        valence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Record a sensory moment into the current episode.

        Called by AliceBrain.see() / hear() after semantic field processing.
        Automatically manages episode boundaries based on temporal gaps.

        Returns:
            {
                "episode_id": int,
                "snapshot_index": int,
                "is_new_episode": bool,
            }
        """
        t = timestamp if timestamp is not None else time.monotonic()
        fp = np.asarray(fingerprint, dtype=np.float64)

        # Episode boundary detection
        is_new = False
        if (self._current_episode is None
                or not self._current_episode.is_open
                or (self._last_record_time >= 0 and
                    t - self._last_record_time > EPISODE_GAP_THRESHOLD)):
            is_new = True
            self._start_new_episode(t)

        # Create snapshot
        snapshot = EpisodicSnapshot(
            timestamp=t,
            modality=modality,
            fingerprint=fp.copy(),
            attractor_label=attractor_label,
            gamma=gamma,
            valence=valence,
        )

        self._current_episode.add_snapshot(snapshot)
        self._last_record_time = t
        self.total_snapshots_recorded += 1

        return {
            "episode_id": self._current_episode.episode_id,
            "snapshot_index": self._current_episode.n_snapshots - 1,
            "is_new_episode": is_new,
        }

    def _start_new_episode(self, creation_time: float):
        """Close current episode (if any) and start a new one."""
        if self._current_episode is not None:
            self._current_episode.is_open = False

        episode = Episode(
            episode_id=self._next_episode_id,
            creation_time=creation_time,
            last_replay_time=creation_time,
        )
        self._next_episode_id += 1
        self.episodes.append(episode)
        self._current_episode = episode
        self.total_episodes_created += 1

        # Capacity management: remove oldest, weakest episode
        if len(self.episodes) > self.max_episodes:
            self._evict_weakest()

    def _evict_weakest(self):
        """Remove the episode with lowest retrieval strength."""
        if len(self.episodes) <= 1:
            return
        t_now = time.monotonic()
        # Don't evict the current episode
        candidates = [e for e in self.episodes if e != self._current_episode]
        if not candidates:
            return
        weakest = min(candidates, key=lambda e: e.recency(t_now))
        self.episodes.remove(weakest)

    def end_episode(self):
        """Manually close the current episode."""
        if self._current_episode is not None:
            self._current_episode.is_open = False
            self._current_episode = None

    # ------------------------------------------------------------------
    # Recall (Pattern Completion)
    # ------------------------------------------------------------------

    def recall(
        self,
        cue_fingerprint: Optional[np.ndarray] = None,
        cue_modality: Optional[str] = None,
        cue_concept: Optional[str] = None,
        top_k: int = 3,
        t_now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recall episodes matching a cue.

        Pattern completion: a partial sensory cue or concept label
        retrieves the most relevant episodes.

        Three cue types:
          1. Fingerprint + modality : "I see something that looks like..."
          2. Concept label : "everything related to 'cat'"
          3. Both : combined (highest precision)

        Args:
            t_now: Optional time reference (default: time.monotonic()).
                   Use for deterministic testing.

        Returns:
            List of episode dicts with retrieval_strength, sorted descending.
        """
        self.total_recalls += 1
        if t_now is None:
            t_now = time.monotonic()

        cue_fp = None
        if cue_fingerprint is not None:
            cue_fp = np.asarray(cue_fingerprint, dtype=np.float64)

        scored = []
        for episode in self.episodes:
            strength = episode.retrieval_strength(
                t_now, cue_fp, cue_modality, cue_concept
            )
            if strength >= RETRIEVAL_THRESHOLD * 0.1:  # loose filter
                scored.append((episode, strength))

        # Sort by retrieval strength descending
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for episode, strength in scored[:top_k]:
            ep_dict = episode.to_dict()
            ep_dict["retrieval_strength"] = round(strength, 4)
            ep_dict["snapshots"] = [s.to_dict() for s in episode.snapshots]
            results.append(ep_dict)

        return results

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(self, episode_id: int) -> Optional[List[Dict[str, Any]]]:
        """
        Replay an episode -- re-activate its snapshot sequence.

        Returns the ordered list of snapshots (for re-feeding to semantic field).
        Updates replay count and recency.

        During sleep consolidation, replayed snapshots strengthen
        semantic field attractors.
        """
        self.total_replays += 1
        episode = self._find_episode(episode_id)
        if episode is None:
            return None

        episode.replay_count += 1
        episode.last_replay_time = time.monotonic()

        return [s.to_dict() for s in episode.snapshots]

    # ------------------------------------------------------------------
    # Consolidation (Episode -> Semantic Field)
    # ------------------------------------------------------------------

    def consolidate(
        self,
        semantic_field,
        max_episodes: int = 5,
    ) -> Dict[str, Any]:
        """
        Sleep consolidation: transfer episodic patterns to semantic field.

        Physics:
          During sleep, the hippocampus replays recent episodes.
          Each replay strengthens the semantic field attractors
          (increases mass -> sharper Q -> better discrimination).
          After consolidation, the specific episode can be forgotten
          but the CONCEPT persists permanently.

        This is why you forget what you had for breakfast last Tuesday
        but know perfectly well what "breakfast" means.

        Args:
            semantic_field: SemanticFieldEngine or SemanticField instance.
            max_episodes: How many episodes to consolidate per sleep cycle.

        Returns:
            {
                "episodes_consolidated": int,
                "snapshots_transferred": int,
                "concepts_strengthened": set,
            }
        """
        self.total_consolidations += 1
        t_now = time.monotonic()

        # Select episodes to consolidate (most recent unconsolidated)
        candidates = [
            e for e in self.episodes
            if not e.consolidated and not e.is_open
        ]
        # Prioritize: emotional > recent > old
        candidates.sort(
            key=lambda e: (abs(e.emotional_peak), e.recency(t_now)),
            reverse=True,
        )
        to_consolidate = candidates[:max_episodes]

        snapshots_transferred = 0
        concepts_strengthened = set()

        for episode in to_consolidate:
            # Replay for consolidation
            episode.replay_count += 1
            episode.last_replay_time = t_now

            for snapshot in episode.snapshots:
                if snapshot.attractor_label is not None:
                    # Determine the right absorb interface
                    if hasattr(semantic_field, 'field'):
                        # SemanticFieldEngine
                        semantic_field.field.absorb(
                            snapshot.attractor_label,
                            snapshot.fingerprint,
                            snapshot.modality,
                            snapshot.valence * CONSOLIDATION_BOOST,
                        )
                    else:
                        # SemanticField directly
                        semantic_field.absorb(
                            snapshot.attractor_label,
                            snapshot.fingerprint,
                            snapshot.modality,
                            snapshot.valence * CONSOLIDATION_BOOST,
                        )
                    concepts_strengthened.add(snapshot.attractor_label)
                    snapshots_transferred += 1

            episode.consolidated = True

        return {
            "episodes_consolidated": len(to_consolidate),
            "snapshots_transferred": snapshots_transferred,
            "concepts_strengthened": list(concepts_strengthened),
        }

    # ------------------------------------------------------------------
    # Associative Retrieval (cross-membrane)
    # ------------------------------------------------------------------

    def recall_by_concept(
        self,
        concept_label: str,
        target_modality: Optional[str] = None,
        top_k: int = 3,
        t_now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recall episodes and extract snapshots involving a specific concept.

        If target_modality is set, only return snapshots from that membrane.
        This enables: "What did the cat LOOK like?" (concept=cat, modality=visual)
                      = cross-membrane pattern completion through time.
        """
        results = self.recall(cue_concept=concept_label, top_k=top_k * 2,
                              t_now=t_now)

        if target_modality is None:
            return results[:top_k]

        # Filter snapshots to target modality
        filtered = []
        for ep in results:
            modal_snaps = [
                s for s in ep.get("snapshots", [])
                if s["modality"] == target_modality
            ]
            if modal_snaps:
                ep_copy = dict(ep)
                ep_copy["snapshots"] = modal_snaps
                filtered.append(ep_copy)

        return filtered[:top_k]

    # ------------------------------------------------------------------
    # Sequence Extraction (for Wernicke)
    # ------------------------------------------------------------------

    def get_concept_sequences(
        self,
        min_length: int = 2,
        max_count: int = 50,
    ) -> List[List[str]]:
        """
        Extract ordered concept sequences from all episodes.

        These sequences are the raw material for Wernicke's area:
        transition probabilities between concepts = proto-grammar.

        Returns:
            List of concept-label sequences (e.g., [["cat", "meow", "soft"], ...])
        """
        sequences = []
        for episode in self.episodes:
            seq = []
            for snap in episode.snapshots:
                if snap.attractor_label is not None:
                    # Avoid consecutive duplicates
                    if not seq or seq[-1] != snap.attractor_label:
                        seq.append(snap.attractor_label)
            if len(seq) >= min_length:
                sequences.append(seq)
                if len(sequences) >= max_count:
                    break
        return sequences

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _find_episode(self, episode_id: int) -> Optional[Episode]:
        for ep in self.episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def get_current_episode_id(self) -> Optional[int]:
        if self._current_episode is not None:
            return self._current_episode.episode_id
        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_episodes": len(self.episodes),
            "total_snapshots_recorded": self.total_snapshots_recorded,
            "total_episodes_created": self.total_episodes_created,
            "total_recalls": self.total_recalls,
            "total_replays": self.total_replays,
            "total_consolidations": self.total_consolidations,
            "current_episode_id": self.get_current_episode_id(),
            "recent_episodes": [
                e.to_dict() for e in self.episodes[-5:]
            ],
        }
