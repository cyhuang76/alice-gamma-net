# -*- coding: utf-8 -*-
"""
Narrative Memory Engine -- Autobiographical Memory as Causal-Temporal Weaving
Phase 20.1: The Physics of Self-Story

Core Philosophy:

  The hippocampus records EPISODES -- isolated film clips of experience.
  But human memory is NOT a library of disconnected clips.
  It is a NARRATIVE -- a causally-linked, emotionally-arced autobiography.

  "I remember meeting her at the café (Episode 12),
   then the argument (Episode 17),
   then the reconciliation (Episode 23)."

  This narrative structure is MISSING from the hippocampus alone.
  The hippocampus knows WHAT happened and WHEN.
  The narrative memory knows WHY it happened and WHAT IT MEANS.

Physical Model:

  1. NarrativeArc = a causal chain of episodes
     Each arc has a theme (semantic concept), an emotional trajectory,
     and explicit cause-effect links between episodes.

     Physics: episodes are "charges" in a narrative field.
     Nearby charges (shared concepts + temporal proximity) attract.
     Emotional gradient determines the arc's "direction" (comedy vs tragedy).

  2. Causal Link = impedance-weighted directed edge
     If Episode A and Episode B share concepts AND A precedes B,
     there is a potential causal link.
     Link strength = concept_overlap × temporal_proximity × emotional_coherence.
     Γ_causal = 1 - link_strength (high overlap = low impedance = strong link).

  3. Narrative Coherence = mean(1 - Γ_causal) across the arc's links
     A coherent narrative has low causal impedance throughout.
     An incoherent story has gaps (high Γ) between episodes.

  4. Emotional Trajectory = valence over time
     Comedy: negative → positive (rising arc)
     Tragedy: positive → negative (falling arc)
     Character development: oscillating with increasing amplitude

  5. Autobiographical Index = thematic clustering
     Instead of recall by recency or cue, recall by LIFE THEME:
     "all memories about learning to speak"
     "all memories about pain and recovery"

  6. Episode Summary = compression via dominant concepts + peak emotion
     A 50-snapshot episode compressed to:
     {dominant_concepts, peak_emotion, duration, modalities, gist}

Equations:
  CausalStrength(A→B) = overlap(A,B) × exp(-λ|t_B - t_A|) × emo_coherence(A,B)
  Γ_causal = 1 - CausalStrength
  NarrativeCoherence(arc) = 1 - mean(Γ_causal for all links in arc)
  EmotionalArc(arc) = [episode.emotional_peak for episode in arc.episodes]
  ThematicSimilarity(arc, theme) = |arc.concepts ∩ theme.concepts| / |arc.concepts ∪ theme.concepts|
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# --- Causal linking ---
CAUSAL_TEMPORAL_LAMBDA = 0.005     # Temporal decay for causal link strength
CAUSAL_OVERLAP_WEIGHT = 0.5        # Weight of concept overlap in link strength
CAUSAL_TEMPORAL_WEIGHT = 0.3       # Weight of temporal proximity
CAUSAL_EMOTIONAL_WEIGHT = 0.2      # Weight of emotional coherence
MIN_CAUSAL_STRENGTH = 0.1          # Minimum strength to form a causal link
MAX_CAUSAL_DISTANCE_S = 600.0      # Max time gap (seconds) for causal link

# --- Narrative arcs ---
MAX_ARCS = 50                      # Maximum stored narrative arcs
MAX_EPISODES_PER_ARC = 20          # Maximum episodes in a single arc
ARC_MERGE_THRESHOLD = 0.6          # Jaccard similarity to merge arcs
ARC_COHERENCE_MIN = 0.2            # Minimum coherence to form an arc

# --- Summarization ---
SUMMARY_TOP_CONCEPTS = 5           # Top concepts in a summary
SUMMARY_MIN_SNAPSHOTS = 3          # Minimum snapshots to summarize

# --- Thematic indexing ---
THEME_SIMILARITY_THRESHOLD = 0.3   # Min Jaccard for thematic match
MAX_THEMES = 30                    # Maximum tracked themes

# --- Autobiographical timeline ---
TIMELINE_BIN_SECONDS = 60.0        # Timeline resolution (1 minute bins)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CausalLink:
    """
    A directed causal link between two episodes.

    Physics: impedance-weighted edge in the narrative graph.
    Low Γ_causal = strong causal connection.
    """
    cause_episode_id: int
    effect_episode_id: int
    strength: float                  # 0~1, higher = stronger causal link
    gamma_causal: float              # 1 - strength (impedance)
    shared_concepts: List[str] = field(default_factory=list)
    temporal_gap: float = 0.0        # Seconds between episodes
    emotional_direction: float = 0.0  # Δvalence (positive = escalation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause_episode_id,
            "effect": self.effect_episode_id,
            "strength": round(self.strength, 4),
            "gamma_causal": round(self.gamma_causal, 4),
            "shared_concepts": self.shared_concepts,
            "temporal_gap": round(self.temporal_gap, 2),
            "emotional_direction": round(self.emotional_direction, 3),
        }


@dataclass
class EpisodeSummary:
    """
    Compressed representation of an episode.

    Physics: lossy compression -- dominant attractors + peak emotion.
    """
    episode_id: int
    dominant_concepts: List[str] = field(default_factory=list)
    peak_valence: float = 0.0
    mean_valence: float = 0.0
    duration: float = 0.0
    modalities: List[str] = field(default_factory=list)
    n_snapshots: int = 0
    creation_time: float = 0.0
    gist: str = ""                   # One-line human-readable gist

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "dominant_concepts": self.dominant_concepts,
            "peak_valence": round(self.peak_valence, 3),
            "mean_valence": round(self.mean_valence, 3),
            "duration": round(self.duration, 3),
            "modalities": self.modalities,
            "n_snapshots": self.n_snapshots,
            "gist": self.gist,
        }


@dataclass
class NarrativeArc:
    """
    A causally-linked chain of episodes forming a coherent story.

    Physics: a low-impedance path through the narrative graph.
    Each arc has:
      - A theme (dominant concept family)
      - An emotional trajectory (valence over episodes)
      - Explicit causal links between consecutive episodes
      - A coherence score (mean causal strength)
    """
    arc_id: int
    episode_chain: List[int] = field(default_factory=list)  # Ordered episode IDs
    causal_links: List[CausalLink] = field(default_factory=list)
    theme: str = ""                  # Dominant concept / life theme
    theme_concepts: Set[str] = field(default_factory=set)
    emotional_trajectory: List[float] = field(default_factory=list)
    coherence: float = 0.0          # Mean(1 - Γ_causal) across links
    creation_time: float = 0.0
    last_updated: float = 0.0
    is_complete: bool = False        # Arc has reached narrative closure

    @property
    def length(self) -> int:
        return len(self.episode_chain)

    @property
    def emotional_arc_type(self) -> str:
        """
        Classify the emotional trajectory:
          comedy:   negative → positive
          tragedy:  positive → negative
          growth:   oscillation with increasing amplitude
          stable:   flat
          complex:  other
        """
        if len(self.emotional_trajectory) < 2:
            return "stable"
        first_half = self.emotional_trajectory[:len(self.emotional_trajectory) // 2]
        second_half = self.emotional_trajectory[len(self.emotional_trajectory) // 2:]
        mean_first = np.mean(first_half) if first_half else 0.0
        mean_second = np.mean(second_half) if second_half else 0.0
        delta = mean_second - mean_first

        if abs(delta) < 0.1:
            return "stable"
        if delta > 0.2:
            return "comedy"  # Rising arc
        if delta < -0.2:
            return "tragedy"  # Falling arc

        # Check for growth (oscillation)
        if len(self.emotional_trajectory) >= 4:
            amps = [abs(v) for v in self.emotional_trajectory]
            if amps[-1] > amps[0] * 1.3:
                return "growth"

        return "complex"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arc_id": self.arc_id,
            "theme": self.theme,
            "theme_concepts": list(self.theme_concepts),
            "episode_chain": self.episode_chain,
            "n_episodes": self.length,
            "coherence": round(self.coherence, 4),
            "emotional_arc_type": self.emotional_arc_type,
            "emotional_trajectory": [round(v, 3) for v in self.emotional_trajectory],
            "is_complete": self.is_complete,
            "n_causal_links": len(self.causal_links),
        }


# ============================================================================
# NarrativeMemoryEngine -- Autobiographical Memory Weaver
# ============================================================================

class NarrativeMemoryEngine:
    """
    Narrative Memory -- weaves hippocampal episodes into autobiographical arcs.

    Composition (NOT inheritance) over HippocampusEngine.
    This engine:
      1. Detects causal links between episodes (shared concepts + temporal proximity)
      2. Chains causally-linked episodes into NarrativeArcs
      3. Tracks emotional trajectories within arcs
      4. Provides thematic autobiography retrieval
      5. Compresses episodes into summaries
      6. Maintains a global concept timeline

    Integration with AliceBrain.perceive() pipeline:
      After hippocampus.record() → narrative_memory.on_episode_updated()
      After hippocampus.consolidate() → narrative_memory.consolidate_narratives()
      For recall: narrative_memory.recall_by_theme(), .get_autobiography()
    """

    def __init__(self, hippocampus=None, max_arcs: int = MAX_ARCS):
        # Composition: reference to HippocampusEngine (injected)
        self.hippocampus = hippocampus

        # Narrative arcs
        self.arcs: List[NarrativeArc] = []
        self.max_arcs = max_arcs
        self._next_arc_id = 1

        # Causal link graph: (cause_ep_id, effect_ep_id) → CausalLink
        self.causal_graph: Dict[Tuple[int, int], CausalLink] = {}

        # Episode summaries cache
        self.summaries: Dict[int, EpisodeSummary] = {}

        # Thematic index: theme_label → list of arc_ids
        self.thematic_index: Dict[str, List[int]] = defaultdict(list)

        # Global concept timeline: concept → [(timestamp, episode_id)]
        self.concept_timeline: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

        # Statistics
        self.total_links_created = 0
        self.total_arcs_created = 0
        self.total_weave_calls = 0
        self.total_queries = 0

    # ------------------------------------------------------------------
    # Episode Summary
    # ------------------------------------------------------------------

    def summarize_episode(self, episode) -> EpisodeSummary:
        """
        Compress an episode into its essential information.

        Physics: lossy compression via dominant attractor identification.
        """
        if episode.episode_id in self.summaries:
            return self.summaries[episode.episode_id]

        concepts = []
        valences = []
        modalities = set()

        for snap in episode.snapshots:
            if snap.attractor_label:
                concepts.append(snap.attractor_label)
            valences.append(snap.valence)
            modalities.add(snap.modality)

        # Dominant concepts by frequency
        concept_counts = Counter(concepts)
        dominant = [c for c, _ in concept_counts.most_common(SUMMARY_TOP_CONCEPTS)]

        # Gist = comma-separated dominant concepts + emotional tag
        peak_v = episode.emotional_peak
        emo_tag = "positive" if peak_v > 0.3 else ("negative" if peak_v < -0.3 else "neutral")
        gist = f"{'+'.join(dominant[:3]) if dominant else 'unknown'} ({emo_tag})"

        summary = EpisodeSummary(
            episode_id=episode.episode_id,
            dominant_concepts=dominant,
            peak_valence=peak_v,
            mean_valence=float(np.mean(valences)) if valences else 0.0,
            duration=episode.duration,
            modalities=sorted(modalities),
            n_snapshots=episode.n_snapshots,
            creation_time=episode.creation_time,
            gist=gist,
        )
        self.summaries[episode.episode_id] = summary
        return summary

    # ------------------------------------------------------------------
    # Causal Link Detection
    # ------------------------------------------------------------------

    def _concept_overlap(self, ep_a, ep_b) -> Tuple[float, List[str]]:
        """
        Jaccard overlap coefficient for concepts between two episodes.
        Returns (overlap_score, shared_concept_list).
        """
        concepts_a = ep_a.concepts_mentioned
        concepts_b = ep_b.concepts_mentioned
        if not concepts_a or not concepts_b:
            return 0.0, []
        shared = concepts_a & concepts_b
        union = concepts_a | concepts_b
        return len(shared) / len(union), sorted(shared)

    def _emotional_coherence(self, ep_a, ep_b) -> float:
        """
        Emotional coherence: how smoothly does affect flow from A to B?

        Physics: abrupt valence reversals = high impedance mismatch.
        Smooth transitions (same sign or gradual change) = low impedance.
        """
        va = ep_a.emotional_peak
        vb = ep_b.emotional_peak
        # Coherence = 1 - normalized distance
        return 1.0 - min(1.0, abs(va - vb) / 2.0)

    def detect_causal_link(self, ep_a, ep_b) -> Optional[CausalLink]:
        """
        Detect if there is a causal link from episode A to episode B.

        Conditions:
          1. A was created before B
          2. They share at least one concept
          3. Temporal gap is within MAX_CAUSAL_DISTANCE_S

        Strength = weighted combination of:
          - Concept overlap (Jaccard)
          - Temporal proximity (exponential decay)
          - Emotional coherence (smooth valence transition)
        """
        # Temporal ordering
        t_a = ep_a.creation_time
        t_b = ep_b.creation_time
        if t_a >= t_b:
            return None

        temporal_gap = t_b - t_a
        if temporal_gap > MAX_CAUSAL_DISTANCE_S:
            return None

        # Concept overlap
        overlap, shared = self._concept_overlap(ep_a, ep_b)
        if overlap < 1e-6:
            return None  # No shared concepts = no causal link

        # Temporal proximity
        temporal_score = math.exp(-CAUSAL_TEMPORAL_LAMBDA * temporal_gap)

        # Emotional coherence
        emo_coherence = self._emotional_coherence(ep_a, ep_b)

        # Combined strength
        strength = (
            CAUSAL_OVERLAP_WEIGHT * overlap
            + CAUSAL_TEMPORAL_WEIGHT * temporal_score
            + CAUSAL_EMOTIONAL_WEIGHT * emo_coherence
        )

        if strength < MIN_CAUSAL_STRENGTH:
            return None

        link = CausalLink(
            cause_episode_id=ep_a.episode_id,
            effect_episode_id=ep_b.episode_id,
            strength=strength,
            gamma_causal=1.0 - strength,
            shared_concepts=shared,
            temporal_gap=temporal_gap,
            emotional_direction=ep_b.emotional_peak - ep_a.emotional_peak,
        )
        return link

    def scan_causal_links(self, episodes: Optional[List] = None) -> int:
        """
        Scan all episode pairs for potential causal links.
        Returns number of new links detected.
        """
        if episodes is None:
            if self.hippocampus is None:
                return 0
            episodes = self.hippocampus.episodes

        new_links = 0
        for i, ep_a in enumerate(episodes):
            if ep_a.is_open:
                continue
            for ep_b in episodes[i + 1:]:
                if ep_b.is_open:
                    continue
                key = (ep_a.episode_id, ep_b.episode_id)
                if key in self.causal_graph:
                    continue
                link = self.detect_causal_link(ep_a, ep_b)
                if link is not None:
                    self.causal_graph[key] = link
                    self.total_links_created += 1
                    new_links += 1
        return new_links

    # ------------------------------------------------------------------
    # Narrative Weaving
    # ------------------------------------------------------------------

    def weave(self, episode_ids: Optional[List[int]] = None) -> List[NarrativeArc]:
        """
        Weave episodes into narrative arcs by following causal link chains.

        Algorithm:
          1. Build adjacency list from causal graph
          2. Find connected components (each = potential arc)
          3. For each component, extract longest causal chain
          4. Compute coherence and emotional trajectory
          5. Assign theme from dominant concepts

        If episode_ids is None, weave ALL available episodes.
        Returns newly created arcs.
        """
        self.total_weave_calls += 1

        if self.hippocampus is None:
            return []

        # Step 0: Ensure causal links are up-to-date
        episodes = self.hippocampus.episodes
        self.scan_causal_links(episodes)

        # Step 1: Build adjacency list
        adj: Dict[int, List[int]] = defaultdict(list)
        in_degree: Dict[int, int] = defaultdict(int)
        all_ep_ids = set()

        for (cause_id, effect_id), link in self.causal_graph.items():
            if episode_ids is not None:
                if cause_id not in episode_ids or effect_id not in episode_ids:
                    continue
            adj[cause_id].append(effect_id)
            in_degree[effect_id] += 1
            all_ep_ids.add(cause_id)
            all_ep_ids.add(effect_id)

        if not all_ep_ids:
            return []

        # Step 2: Find connected components (undirected)
        visited = set()
        components = []

        def _bfs(start):
            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in adj.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
                # Also check reverse edges
                for (c, e), _ in self.causal_graph.items():
                    if e == node and c not in visited and c in all_ep_ids:
                        queue.append(c)
            return component

        for ep_id in sorted(all_ep_ids):
            if ep_id not in visited:
                comp = _bfs(ep_id)
                if len(comp) >= 2:
                    components.append(comp)

        # Step 3: For each component, build arc
        new_arcs = []
        ep_by_id = {e.episode_id: e for e in episodes}

        for comp in components:
            # Check if this component is already covered by an existing arc
            existing_covered = False
            for arc in self.arcs:
                if comp.issubset(set(arc.episode_chain)):
                    existing_covered = True
                    break
            if existing_covered:
                continue

            # Sort by creation time → causal chain order
            sorted_ids = sorted(comp, key=lambda eid: ep_by_id[eid].creation_time
                                if eid in ep_by_id else 0)

            # Cap at MAX_EPISODES_PER_ARC
            sorted_ids = sorted_ids[:MAX_EPISODES_PER_ARC]

            # Collect causal links in this chain
            chain_links = []
            for i in range(len(sorted_ids) - 1):
                key = (sorted_ids[i], sorted_ids[i + 1])
                if key in self.causal_graph:
                    chain_links.append(self.causal_graph[key])

            # Coherence = mean link strength
            if chain_links:
                coherence = float(np.mean([l.strength for l in chain_links]))
            else:
                coherence = 0.0

            if coherence < ARC_COHERENCE_MIN:
                continue

            # Emotional trajectory
            emo_traj = []
            for eid in sorted_ids:
                if eid in ep_by_id:
                    emo_traj.append(ep_by_id[eid].emotional_peak)

            # Theme = most frequent concept across all episodes in the arc
            all_concepts = Counter()
            theme_concepts = set()
            for eid in sorted_ids:
                if eid in ep_by_id:
                    for c in ep_by_id[eid].concepts_mentioned:
                        all_concepts[c] += 1
                        theme_concepts.add(c)

            theme = all_concepts.most_common(1)[0][0] if all_concepts else "unknown"

            arc = NarrativeArc(
                arc_id=self._next_arc_id,
                episode_chain=sorted_ids,
                causal_links=chain_links,
                theme=theme,
                theme_concepts=theme_concepts,
                emotional_trajectory=emo_traj,
                coherence=coherence,
                creation_time=time.monotonic(),
                last_updated=time.monotonic(),
            )
            self._next_arc_id += 1
            self.arcs.append(arc)
            self.total_arcs_created += 1
            new_arcs.append(arc)

            # Update thematic index
            self.thematic_index[theme].append(arc.arc_id)

            # Evict oldest arcs if over limit
            if len(self.arcs) > self.max_arcs:
                # Remove least coherent
                self.arcs.sort(key=lambda a: a.coherence, reverse=True)
                removed = self.arcs[self.max_arcs:]
                self.arcs = self.arcs[:self.max_arcs]
                for r in removed:
                    for theme_key, arc_ids in self.thematic_index.items():
                        if r.arc_id in arc_ids:
                            arc_ids.remove(r.arc_id)

        return new_arcs

    # ------------------------------------------------------------------
    # Episode Event Hook
    # ------------------------------------------------------------------

    def on_episode_closed(self, episode) -> Dict[str, Any]:
        """
        Called when hippocampus closes an episode.

        1. Generate summary
        2. Update concept timeline
        3. Scan for new causal links with recent episodes
        4. Attempt to extend or create narrative arcs
        """
        # 1. Summary
        summary = self.summarize_episode(episode)

        # 2. Concept timeline
        for snap in episode.snapshots:
            if snap.attractor_label:
                self.concept_timeline[snap.attractor_label].append(
                    (snap.timestamp, episode.episode_id)
                )

        # 3. Scan causal links (only against recent episodes)
        if self.hippocampus is not None:
            recent = self.hippocampus.episodes[-20:]  # Last 20
            new_links = 0
            for ep in recent:
                if ep.episode_id == episode.episode_id:
                    continue
                if ep.is_open:
                    continue
                # Check both directions
                for (a, b) in [(ep, episode), (episode, ep)]:
                    key = (a.episode_id, b.episode_id)
                    if key not in self.causal_graph:
                        link = self.detect_causal_link(a, b)
                        if link:
                            self.causal_graph[key] = link
                            self.total_links_created += 1
                            new_links += 1
        else:
            new_links = 0

        # 4. Try to extend existing arcs or create new ones
        extended_arcs = []
        for arc in self.arcs:
            if arc.is_complete:
                continue
            last_ep_id = arc.episode_chain[-1]
            key = (last_ep_id, episode.episode_id)
            if key in self.causal_graph:
                link = self.causal_graph[key]
                arc.episode_chain.append(episode.episode_id)
                arc.causal_links.append(link)
                arc.emotional_trajectory.append(episode.emotional_peak)
                # Recompute coherence
                if arc.causal_links:
                    arc.coherence = float(np.mean([l.strength for l in arc.causal_links]))
                arc.theme_concepts |= episode.concepts_mentioned
                arc.last_updated = time.monotonic()
                extended_arcs.append(arc.arc_id)

        return {
            "summary": summary.to_dict(),
            "new_causal_links": new_links,
            "arcs_extended": extended_arcs,
        }

    # ------------------------------------------------------------------
    # Narrative Consolidation
    # ------------------------------------------------------------------

    def consolidate_narratives(self) -> Dict[str, Any]:
        """
        Called during sleep to consolidate narrative structures.

        1. Weave any un-arced episodes
        2. Merge overlapping arcs
        3. Identify completed arcs (no recent extensions)
        """
        # 1. Weave
        new_arcs = self.weave()

        # 2. Merge overlapping arcs (Jaccard > ARC_MERGE_THRESHOLD)
        merged_count = 0
        to_remove = set()
        arc_list = [a for a in self.arcs if a.arc_id not in to_remove]

        for i in range(len(arc_list)):
            if arc_list[i].arc_id in to_remove:
                continue
            for j in range(i + 1, len(arc_list)):
                if arc_list[j].arc_id in to_remove:
                    continue
                set_a = set(arc_list[i].episode_chain)
                set_b = set(arc_list[j].episode_chain)
                union = set_a | set_b
                intersection = set_a & set_b
                if len(union) > 0 and len(intersection) / len(union) > ARC_MERGE_THRESHOLD:
                    # Merge j into i
                    merged_chain = sorted(
                        set_a | set_b,
                        key=lambda eid: eid,
                    )
                    arc_list[i].episode_chain = merged_chain
                    arc_list[i].theme_concepts |= arc_list[j].theme_concepts
                    arc_list[i].last_updated = time.monotonic()
                    to_remove.add(arc_list[j].arc_id)
                    merged_count += 1

        self.arcs = [a for a in self.arcs if a.arc_id not in to_remove]

        # 3. Mark stale arcs as complete
        t_now = time.monotonic()
        completed = 0
        for arc in self.arcs:
            if not arc.is_complete and t_now - arc.last_updated > 300.0:
                arc.is_complete = True
                completed += 1

        return {
            "new_arcs": len(new_arcs),
            "merged_arcs": merged_count,
            "completed_arcs": completed,
            "total_arcs": len(self.arcs),
            "total_causal_links": len(self.causal_graph),
        }

    # ------------------------------------------------------------------
    # Retrieval APIs
    # ------------------------------------------------------------------

    def recall_by_theme(self, theme: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve narrative arcs matching a theme.

        Uses Jaccard similarity between query theme and arc theme concepts.
        """
        self.total_queries += 1

        results = []
        for arc in self.arcs:
            # Direct theme match
            if arc.theme == theme:
                results.append((arc, 1.0))
                continue
            # Concept overlap
            if theme in arc.theme_concepts:
                sim = 1.0 / (1.0 + len(arc.theme_concepts))
                results.append((arc, sim + 0.5))

        results.sort(key=lambda x: x[1], reverse=True)
        return [
            {**arc.to_dict(), "relevance": round(score, 4)}
            for arc, score in results[:top_k]
        ]

    def get_autobiography(
        self,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        theme: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get the full autobiography -- ordered narrative arcs with summaries.
        """
        self.total_queries += 1

        arcs = self.arcs
        if theme:
            arcs = [a for a in arcs if theme in a.theme_concepts or a.theme == theme]

        # Sort by first episode's creation time
        if self.hippocampus:
            ep_by_id = {e.episode_id: e for e in self.hippocampus.episodes}
        else:
            ep_by_id = {}

        def arc_start_time(arc):
            if arc.episode_chain and arc.episode_chain[0] in ep_by_id:
                return ep_by_id[arc.episode_chain[0]].creation_time
            return arc.creation_time

        arcs = sorted(arcs, key=arc_start_time)

        result = []
        for arc in arcs:
            arc_dict = arc.to_dict()
            # Attach summaries for each episode
            arc_dict["episode_summaries"] = []
            for eid in arc.episode_chain:
                if eid in self.summaries:
                    arc_dict["episode_summaries"].append(self.summaries[eid].to_dict())
                elif eid in ep_by_id:
                    s = self.summarize_episode(ep_by_id[eid])
                    arc_dict["episode_summaries"].append(s.to_dict())
            result.append(arc_dict)

        return result

    def get_concept_timeline(
        self,
        concept: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get the timeline of a specific concept across all episodes.
        Returns chronologically ordered [(timestamp, episode_id, summary)].
        """
        self.total_queries += 1
        entries = self.concept_timeline.get(concept, [])
        entries = sorted(entries, key=lambda x: x[0])[:limit]

        result = []
        for ts, ep_id in entries:
            entry = {
                "timestamp": round(ts, 4),
                "episode_id": ep_id,
            }
            if ep_id in self.summaries:
                entry["summary"] = self.summaries[ep_id].gist
            result.append(entry)
        return result

    def query_timeline(
        self,
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """
        Get all episodes within a time range, with summaries and causal links.
        """
        self.total_queries += 1
        if self.hippocampus is None:
            return []

        results = []
        for ep in self.hippocampus.episodes:
            if ep.is_open:
                continue
            if start_time <= ep.creation_time <= end_time:
                summary = self.summarize_episode(ep)
                # Find causal links involving this episode
                links_from = [
                    self.causal_graph.get((ep.episode_id, other_id))
                    for other_id in range(ep.episode_id + 1, ep.episode_id + 100)
                    if (ep.episode_id, other_id) in self.causal_graph
                ]
                links_to = [
                    self.causal_graph.get((other_id, ep.episode_id))
                    for other_id in range(max(0, ep.episode_id - 100), ep.episode_id)
                    if (other_id, ep.episode_id) in self.causal_graph
                ]
                results.append({
                    "summary": summary.to_dict(),
                    "causes": [l.to_dict() for l in links_to if l],
                    "effects": [l.to_dict() for l in links_from if l],
                })

        return results

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """
        Per-cycle maintenance:
          - Prune stale concept timeline entries
          - Update summary cache for new episodes
        """
        # Update summaries for any unsummarized closed episodes
        new_summaries = 0
        if self.hippocampus:
            for ep in self.hippocampus.episodes:
                if not ep.is_open and ep.episode_id not in self.summaries:
                    self.summarize_episode(ep)
                    new_summaries += 1

        return {
            "total_arcs": len(self.arcs),
            "total_links": len(self.causal_graph),
            "new_summaries": new_summaries,
        }

    # ------------------------------------------------------------------
    # State / Stats
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "total_arcs": len(self.arcs),
            "total_causal_links": len(self.causal_graph),
            "total_summaries": len(self.summaries),
            "total_links_created": self.total_links_created,
            "total_arcs_created": self.total_arcs_created,
            "total_weave_calls": self.total_weave_calls,
            "total_queries": self.total_queries,
            "themes": list(self.thematic_index.keys()),
            "concept_timeline_concepts": len(self.concept_timeline),
            "recent_arcs": [a.to_dict() for a in self.arcs[-5:]],
        }

    def get_stats(self) -> Dict[str, Any]:
        return self.get_state()
