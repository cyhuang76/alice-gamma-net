# -*- coding: utf-8 -*-
"""
Recursive Grammar Engine -- Phrase-Structure Syntax as Impedance Hierarchy
Phase 20.2: The Physics of Recursive Language

Core Philosophy:

  Broca's area (Phase 4.3) can produce SINGLE concepts:
    plan("cat") → formants → mouth → waveform

  Wernicke's area (Phase 5.2) can predict NEXT concepts:
    P("sat" | "cat") = 0.4

  But NEITHER can handle RECURSIVE STRUCTURE:
    "The cat [that chased the dog [that bit the man]] sat on the mat."

  This sentence has:
    - Nested relative clauses (center-embedding depth 2)
    - Long-distance dependencies ("the cat ... sat")
    - Phrase structure (NP → Det N, VP → V NP, S → NP VP)

  The Recursive Grammar Engine adds the MISSING LAYER:
    a SYNTAX TREE that organizes concepts into phrases,
    phrases into clauses, and clauses into sentences.

Physical Model:

  1. SyntaxNode = a resonant cavity in the syntactic impedance network
     Each node has a category (S, NP, VP, N, V, Det, PP, etc.)
     Terminal nodes contain concept labels (leaves of the tree).
     Non-terminal nodes contain other nodes (recursive).

  2. PhraseRule = impedance matching condition
     A rule NP → Det + N says: "the impedance of a noun phrase
     is matched when a determiner is followed by a noun."
     Γ_structural = 0 when the rule is perfectly satisfied.
     Γ_structural = 1 when no rule matches the observed sequence.

  3. Merge = the fundamental syntactic operation (Chomsky's Minimalism)
     Merge(α, β) = {α, β} -- combine two units into one phrase.
     Physics: two impedance-matched sub-trees combine into a single
     resonant cavity at the next level.

  4. Center-Embedding = impedance stack depth
     Each embedded clause pushes a pending expectation onto a stack.
     The stack has a finite depth (working memory ≈ 3-4 levels).
     Beyond this depth, comprehension fails (Γ → 1).

  5. Prosody Planning = tree-depth → pitch/volume modulation
     Deeper nodes get lower pitch (subordinate clauses).
     Clause boundaries get pauses (duration increase).
     This is why "The cat that chased the dog that bit the man sat"
     is impossible without prosodic cues but easy with them.

Equations:
  Γ_structural(node) = Σ Γ_child / n_children (propagation)
  Merge_cost(α, β) = Z_α × Z_β — higher cost = harder merge
  Embedding_depth(tree) = max_path_length(root → any terminal)
  Comprehension = 1 / (1 + depth²) — quadratic difficulty increase
  Prosody: pitch(node) = base_pitch × (1 - 0.1 × depth)
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


# ============================================================================
# Physical Constants
# ============================================================================

# --- Impedance ---
GRAMMAR_Z0 = 100.0                 # Base syntactic impedance (Ohm)
RULE_LEARNING_RATE = 0.1           # How fast rules gain confidence
RULE_DECAY_RATE = 0.002            # Per-tick rule confidence decay
MIN_RULE_CONFIDENCE = 0.01         # Minimum rule confidence
MAX_RULE_CONFIDENCE = 5.0          # Maximum rule confidence

# --- Parsing ---
MAX_EMBEDDING_DEPTH = 4            # Maximum center-embedding depth
MAX_PARSE_LENGTH = 30              # Maximum tokens to parse
STRUCTURAL_GAMMA_THRESHOLD = 0.5   # Γ_structural > this = parse failure

# --- Rule capacity ---
MAX_RULES = 100                    # Maximum phrase rules
MAX_CATEGORIES = 20                # Maximum grammatical categories

# --- Prosody ---
PROSODY_PITCH_DECAY = 0.08         # Pitch reduction per tree depth level
PROSODY_PAUSE_BASE = 0.2           # Base pause duration at clause boundaries
PROSODY_CLAUSE_CATEGORIES = {"S", "CP", "REL"}  # Categories that add pauses

# --- Garden-path recovery ---
GARDEN_PATH_REANALYSIS_COST = 0.3  # Γ penalty for reanalysis
MAX_REANALYSIS_ATTEMPTS = 3        # Maximum reparse attempts


# ============================================================================
# Grammatical Categories
# ============================================================================

# Terminal categories (lexical)
TERMINAL_CATEGORIES = {"N", "V", "Det", "Adj", "Adv", "P", "Conj", "Pro"}

# Non-terminal categories (phrasal)
PHRASAL_CATEGORIES = {"S", "NP", "VP", "PP", "AP", "AdvP", "CP", "REL"}

# Special
ROOT_CATEGORY = "S"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SyntaxNode:
    """
    A node in the syntax tree.

    Terminal nodes (leaves): have label, no children.
    Non-terminal nodes: have children, category determines phrase type.

    Physics: each node is a resonant cavity. Γ_structural measures
    how well the children match the expected rule pattern.
    """
    category: str                      # "S", "NP", "VP", "N", "V", "Det", etc.
    label: Optional[str] = None        # Terminal concept label (leaf only)
    children: List['SyntaxNode'] = field(default_factory=list)
    rule_id: Optional[int] = None      # Which PhraseRule produced this node
    gamma_structural: float = 0.0      # Structural impedance mismatch
    z_impedance: float = GRAMMAR_Z0    # Syntactic impedance

    @property
    def is_terminal(self) -> bool:
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        """Maximum depth of the subtree rooted at this node."""
        if self.is_terminal:
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def n_terminals(self) -> int:
        """Number of terminal nodes (words) in the subtree."""
        if self.is_terminal:
            return 1
        return sum(c.n_terminals for c in self.children)

    def get_terminals(self) -> List['SyntaxNode']:
        """Get all terminal nodes in left-to-right order (pre-order leaf traversal)."""
        if self.is_terminal:
            return [self]
        terminals = []
        for child in self.children:
            terminals.extend(child.get_terminals())
        return terminals

    def get_yield(self) -> List[str]:
        """Get the string yield (concept labels) of the tree."""
        return [t.label for t in self.get_terminals() if t.label]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "category": self.category,
            "gamma_structural": round(self.gamma_structural, 4),
            "depth": self.depth,
        }
        if self.label is not None:
            d["label"] = self.label
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.rule_id is not None:
            d["rule_id"] = self.rule_id
        return d

    def to_bracket_string(self) -> str:
        """Bracketed string representation: [S [NP [Det the] [N cat]] [VP [V sat]]]"""
        if self.is_terminal:
            return f"[{self.category} {self.label or '?'}]"
        children_str = " ".join(c.to_bracket_string() for c in self.children)
        return f"[{self.category} {children_str}]"


@dataclass
class PhraseRule:
    """
    A phrase structure rule: LHS → RHS₁ RHS₂ ...

    Example: NP → Det N   (a noun phrase is a determiner followed by a noun)
             S  → NP VP   (a sentence is a noun phrase followed by a verb phrase)
             VP → V NP    (a verb phrase is a verb followed by a noun phrase)

    Physics: the rule defines an impedance matching condition.
    When the rule is satisfied (children match RHS), Γ_structural = 0.
    Confidence grows with successful parses.
    """
    rule_id: int
    lhs: str                           # Left-hand side category ("NP")
    rhs: Tuple[str, ...]              # Right-hand side categories ("Det", "N")
    confidence: float = 0.5            # Rule confidence (affects impedance)
    z_impedance: float = GRAMMAR_Z0    # Rule impedance = Z0 / (1 + confidence)
    usage_count: int = 0
    success_count: int = 0
    creation_time: float = 0.0

    def gamma(self) -> float:
        """Gamma = impedance mismatch relative to Z0."""
        return abs(GRAMMAR_Z0 - self.z_impedance) / (GRAMMAR_Z0 + self.z_impedance)

    def update_impedance(self):
        """Z = Z0 / (1 + confidence)."""
        self.z_impedance = GRAMMAR_Z0 / (1.0 + self.confidence)

    def reinforce(self):
        """Successful parse → increase confidence."""
        self.confidence = min(MAX_RULE_CONFIDENCE,
                              self.confidence + RULE_LEARNING_RATE)
        self.success_count += 1
        self.usage_count += 1
        self.update_impedance()

    def weaken(self):
        """Failed parse → decrease confidence."""
        self.confidence = max(MIN_RULE_CONFIDENCE,
                              self.confidence - RULE_LEARNING_RATE * 0.5)
        self.usage_count += 1
        self.update_impedance()

    def decay(self):
        """Per-tick decay."""
        self.confidence = max(MIN_RULE_CONFIDENCE,
                              self.confidence * (1.0 - RULE_DECAY_RATE))
        self.update_impedance()

    def matches(self, categories: Tuple[str, ...]) -> bool:
        """Does this rule's RHS match the given category sequence?"""
        return self.rhs == categories

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule": f"{self.lhs} → {' '.join(self.rhs)}",
            "confidence": round(self.confidence, 4),
            "gamma": round(self.gamma(), 4),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
        }


@dataclass
class ParseResult:
    """Result of parsing a concept sequence."""
    tree: Optional[SyntaxNode] = None
    success: bool = False
    gamma_structural: float = 1.0      # Overall structural impedance
    comprehension: float = 0.0         # 1 / (1 + depth²)
    embedding_depth: int = 0
    rules_used: List[int] = field(default_factory=list)
    garden_path: bool = False          # Was a garden-path encountered?
    reanalysis_count: int = 0
    parse_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "success": self.success,
            "gamma_structural": round(self.gamma_structural, 4),
            "comprehension": round(self.comprehension, 4),
            "embedding_depth": self.embedding_depth,
            "n_rules_used": len(self.rules_used),
            "garden_path": self.garden_path,
            "reanalysis_count": self.reanalysis_count,
            "parse_time_ms": round(self.parse_time_ms, 3),
        }
        if self.tree:
            d["tree"] = self.tree.to_bracket_string()
        return d


@dataclass
class ProsodyPlan:
    """
    Prosodic plan derived from syntax tree structure.

    Maps each terminal to its prosodic parameters:
      - pitch_factor: relative to base pitch (depth-modulated)
      - volume_factor: relative to base volume
      - pause_after: pause duration after this word (clause boundaries)
    """
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"entries": self.entries}


# ============================================================================
# Lexicon — word category assignments
# ============================================================================

@dataclass
class LexicalEntry:
    """A word's grammatical category assignment."""
    label: str                   # Concept label
    category: str                # Grammatical category (N, V, Det, etc.)
    confidence: float = 0.5      # How confident we are in this assignment

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "category": self.category,
            "confidence": round(self.confidence, 4),
        }


# ============================================================================
# RecursiveGrammarEngine -- Phrase-Structure Parser & Generator
# ============================================================================

class RecursiveGrammarEngine:
    """
    Recursive Grammar -- phrase-structure parsing and generation.

    Composed with BrocaEngine and WernickeEngine:
      - Broca provides articulatory plans for terminal nodes
      - Wernicke provides transition probabilities for prediction
      - This engine provides STRUCTURAL organization

    Core capabilities:
      1. parse(sequence) → SyntaxNode tree
      2. generate(intent) → concept sequence
      3. learn_rule(examples) → discover new PhraseRules
      4. plan_prosody(tree) → prosodic parameters per word
      5. speak_tree(tree, broca, mouth) → sequential speech with prosody

    The grammar starts EMPTY and learns rules from exposure:
      - Wernicke's mature chunks → candidate PhraseRules
      - Successful parses reinforce rules
      - Failed parses weaken or create new rules
    """

    def __init__(self, broca=None, wernicke=None, max_rules: int = MAX_RULES):
        # Composition: references to Broca and Wernicke
        self.broca = broca
        self.wernicke = wernicke

        # Phrase rules library
        self.rules: List[PhraseRule] = []
        self.max_rules = max_rules
        self._next_rule_id = 1

        # Rule index: lhs_category → list of rules
        self.rule_index: Dict[str, List[PhraseRule]] = defaultdict(list)

        # Lexicon: concept_label → list of LexicalEntry (ambiguity allowed)
        self.lexicon: Dict[str, List[LexicalEntry]] = defaultdict(list)

        # Statistics
        self.total_parses = 0
        self.total_generations = 0
        self.total_rules_learned = 0
        self.total_garden_paths = 0
        self.successful_parses = 0

        # Bootstrap basic rules
        self._bootstrap_grammar()

    # ------------------------------------------------------------------
    # Bootstrap Grammar
    # ------------------------------------------------------------------

    def _bootstrap_grammar(self):
        """
        Initialize with universal grammar skeleton.

        These are the minimum rules that ALL human languages share
        (Chomsky's Universal Grammar hypothesis):
          S  → NP VP     (sentence = subject + predicate)
          NP → Det N     (noun phrase = determiner + noun)
          NP → N         (bare noun phrase)
          VP → V         (intransitive verb)
          VP → V NP      (transitive verb + object)
          PP → P NP      (prepositional phrase)
          NP → NP PP     (noun phrase with prepositional modifier)
          VP → V NP PP   (ditransitive verb)
          S  → NP VP PP  (sentence with adjunct)
          CP → Conj S    (complementizer phrase — subordinate clause)
          NP → NP REL    (noun phrase with relative clause)
          REL → Pro VP   (relative clause = pronoun + verb phrase)
        """
        bootstrap_rules = [
            ("S", ("NP", "VP")),
            ("NP", ("Det", "N")),
            ("NP", ("N",)),
            ("NP", ("Adj", "N")),
            ("NP", ("Det", "Adj", "N")),
            ("VP", ("V",)),
            ("VP", ("V", "NP")),
            ("VP", ("V", "NP", "PP")),
            ("PP", ("P", "NP")),
            ("NP", ("NP", "PP")),
            ("S", ("NP", "VP", "PP")),
            ("CP", ("Conj", "S")),
            ("NP", ("NP", "REL")),
            ("REL", ("Pro", "VP")),
        ]

        for lhs, rhs in bootstrap_rules:
            self._add_rule(lhs, rhs, confidence=0.3)

    def _add_rule(self, lhs: str, rhs: Tuple[str, ...],
                  confidence: float = 0.5) -> PhraseRule:
        """Add a new phrase rule."""
        # Check duplicate
        for rule in self.rules:
            if rule.lhs == lhs and rule.rhs == rhs:
                rule.confidence = max(rule.confidence, confidence)
                rule.update_impedance()
                return rule

        rule = PhraseRule(
            rule_id=self._next_rule_id,
            lhs=lhs,
            rhs=rhs,
            confidence=confidence,
            creation_time=time.monotonic(),
        )
        rule.update_impedance()
        self._next_rule_id += 1

        self.rules.append(rule)
        self.rule_index[lhs].append(rule)
        self.total_rules_learned += 1

        # Evict lowest confidence rules if over limit
        if len(self.rules) > self.max_rules:
            self.rules.sort(key=lambda r: r.confidence, reverse=True)
            removed = self.rules[self.max_rules:]
            self.rules = self.rules[:self.max_rules]
            for r in removed:
                if r in self.rule_index.get(r.lhs, []):
                    self.rule_index[r.lhs].remove(r)

        return rule

    # ------------------------------------------------------------------
    # Lexicon Management
    # ------------------------------------------------------------------

    def assign_category(self, label: str, category: str,
                        confidence: float = 0.5) -> LexicalEntry:
        """
        Assign a grammatical category to a concept label.

        A concept can have multiple categories (e.g., "run" is both N and V).
        """
        # Check if already assigned
        for entry in self.lexicon[label]:
            if entry.category == category:
                entry.confidence = max(entry.confidence, confidence)
                return entry

        entry = LexicalEntry(label=label, category=category, confidence=confidence)
        self.lexicon[label].append(entry)
        return entry

    def get_categories(self, label: str) -> List[str]:
        """Get all possible categories for a concept label."""
        return [e.category for e in self.lexicon.get(label, [])]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _make_terminal(self, label: str, category: str) -> SyntaxNode:
        """Create a terminal syntax node."""
        return SyntaxNode(
            category=category,
            label=label,
            z_impedance=GRAMMAR_Z0 / (1.0 + 0.5),  # Default terminal impedance
        )

    def _try_reduce(self, stack: List[SyntaxNode]) -> Optional[Tuple[SyntaxNode, PhraseRule]]:
        """
        Try to reduce the top of the stack using a phrase rule.

        Bottom-up shift-reduce parsing:
        Check if the top N elements match any rule's RHS.
        If so, reduce them to a single node with the rule's LHS.
        """
        # Try rules from longest RHS to shortest
        for length in range(min(4, len(stack)), 0, -1):
            top_cats = tuple(node.category for node in stack[-length:])
            # Find matching rules
            best_rule = None
            best_confidence = -1.0
            for lhs, rules in self.rule_index.items():
                for rule in rules:
                    if rule.rhs == top_cats and rule.confidence > best_confidence:
                        best_rule = rule
                        best_confidence = rule.confidence

            if best_rule is not None:
                # Reduce: pop top N nodes, create parent
                children = stack[-length:]
                # Compute structural gamma
                child_gammas = [c.gamma_structural for c in children]
                avg_child_gamma = float(np.mean(child_gammas)) if child_gammas else 0.0
                rule_gamma = best_rule.gamma()
                structural_gamma = 0.5 * avg_child_gamma + 0.5 * rule_gamma

                parent = SyntaxNode(
                    category=best_rule.lhs,
                    children=list(children),
                    rule_id=best_rule.rule_id,
                    gamma_structural=structural_gamma,
                    z_impedance=best_rule.z_impedance,
                )
                return parent, best_rule

        return None

    def parse(self, concept_sequence: List[str]) -> ParseResult:
        """
        Parse a concept sequence into a syntax tree.

        Algorithm: Shift-Reduce parser
          1. For each concept, create terminal node (shift)
          2. After each shift, try to reduce top of stack (reduce)
          3. Repeat until input exhausted and stack has single S node

        Garden-path recovery:
          If parse fails, try alternative category assignments
          and reparse (limited to MAX_REANALYSIS_ATTEMPTS).
        """
        t0 = time.monotonic()
        self.total_parses += 1

        if not concept_sequence or len(concept_sequence) > MAX_PARSE_LENGTH:
            return ParseResult(
                success=False,
                gamma_structural=1.0,
                parse_time_ms=(time.monotonic() - t0) * 1000,
            )

        # Phase 1: Assign categories to each concept
        token_categories = []
        for label in concept_sequence:
            cats = self.get_categories(label)
            if not cats:
                # Unknown word → default to N (noun, most common open class)
                self.assign_category(label, "N", confidence=0.2)
                cats = ["N"]
            token_categories.append(cats)

        # Phase 2: Try parse with primary categories
        result = self._do_parse(concept_sequence, token_categories, attempt=0)

        # Phase 3: Garden-path recovery (try alternative categories)
        if not result.success and any(len(cats) > 1 for cats in token_categories):
            result.garden_path = True
            self.total_garden_paths += 1
            for attempt in range(1, MAX_REANALYSIS_ATTEMPTS + 1):
                # Rotate ambiguous categories
                alt_cats = []
                for i, cats in enumerate(token_categories):
                    if len(cats) > 1:
                        # Try next category
                        idx = attempt % len(cats)
                        alt_cats.append([cats[idx]] + [c for j, c in enumerate(cats) if j != idx])
                    else:
                        alt_cats.append(cats)
                alt_result = self._do_parse(concept_sequence, alt_cats, attempt=attempt)
                if alt_result.success:
                    alt_result.garden_path = True
                    alt_result.reanalysis_count = attempt
                    alt_result.parse_time_ms = (time.monotonic() - t0) * 1000
                    self.successful_parses += 1
                    return alt_result

            result.reanalysis_count = MAX_REANALYSIS_ATTEMPTS

        if result.success:
            self.successful_parses += 1

        result.parse_time_ms = (time.monotonic() - t0) * 1000
        return result

    def _do_parse(self, sequence: List[str],
                  token_categories: List[List[str]],
                  attempt: int = 0) -> ParseResult:
        """
        Internal shift-reduce parse with given category assignments.
        """
        stack: List[SyntaxNode] = []
        rules_used = []

        for i, label in enumerate(sequence):
            # Shift: create terminal node
            cat = token_categories[i][0]  # Primary category
            terminal = self._make_terminal(label, cat)
            stack.append(terminal)

            # Reduce: repeatedly try to reduce stack top
            max_reductions = 20  # Prevent infinite loops
            for _ in range(max_reductions):
                result = self._try_reduce(stack)
                if result is None:
                    break
                parent, rule = result
                # Pop matched children
                n_children = len(rule.rhs)
                stack = stack[:-n_children]
                stack.append(parent)
                rules_used.append(rule.rule_id)
                rule.reinforce()

        # Final reductions
        for _ in range(20):
            result = self._try_reduce(stack)
            if result is None:
                break
            parent, rule = result
            n_children = len(rule.rhs)
            stack = stack[:-n_children]
            stack.append(parent)
            rules_used.append(rule.rule_id)
            rule.reinforce()

        # Success: single S node on stack
        if len(stack) == 1 and stack[0].category == ROOT_CATEGORY:
            tree = stack[0]
            depth = tree.depth
            comprehension = 1.0 / (1.0 + depth * depth * 0.1)
            return ParseResult(
                tree=tree,
                success=True,
                gamma_structural=tree.gamma_structural,
                comprehension=comprehension,
                embedding_depth=depth,
                rules_used=rules_used,
            )

        # Partial success: wrap remaining stack into an S node
        if len(stack) >= 1:
            # Compute aggregate gamma
            gammas = [n.gamma_structural for n in stack]
            avg_gamma = float(np.mean(gammas))
            # Penalty for unparsed (leftover stack elements)
            leftover_penalty = 0.3 * (len(stack) - 1)
            total_gamma = min(1.0, avg_gamma + leftover_penalty)

            # Construct a partial tree
            if len(stack) > 1:
                wrapper = SyntaxNode(
                    category=ROOT_CATEGORY,
                    children=list(stack),
                    gamma_structural=total_gamma,
                )
                depth = wrapper.depth
            else:
                wrapper = stack[0]
                depth = wrapper.depth

            comprehension = max(0.0, 1.0 / (1.0 + depth * depth * 0.1) - leftover_penalty)

            return ParseResult(
                tree=wrapper,
                success=total_gamma < STRUCTURAL_GAMMA_THRESHOLD,
                gamma_structural=total_gamma,
                comprehension=max(0.0, comprehension),
                embedding_depth=depth,
                rules_used=rules_used,
            )

        return ParseResult(success=False, gamma_structural=1.0)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, intent_category: str = "S",
                 max_depth: int = 3) -> Optional[SyntaxNode]:
        """
        Generate a syntax tree by top-down expansion.

        Starting from intent_category (default "S"), recursively
        expand non-terminals using phrase rules.

        Selection: rules are sampled proportional to confidence.
        """
        self.total_generations += 1
        return self._expand(intent_category, depth=0, max_depth=max_depth)

    def _expand(self, category: str, depth: int,
                max_depth: int) -> Optional[SyntaxNode]:
        """Recursively expand a category into a subtree."""
        # Base case: terminal category or max depth
        if category in TERMINAL_CATEGORIES or depth >= max_depth:
            # Pick a random concept with this category
            candidates = [
                label for label, entries in self.lexicon.items()
                if any(e.category == category for e in entries)
            ]
            if candidates:
                label = candidates[np.random.randint(len(candidates))]
            else:
                label = f"<{category}>"
            return self._make_terminal(label, category)

        # Find applicable rules
        applicable = self.rule_index.get(category, [])
        if not applicable:
            # No rules → treat as terminal
            return self._make_terminal(f"<{category}>", category)

        # Sample rule proportional to confidence
        confidences = np.array([r.confidence for r in applicable])
        probs = confidences / (confidences.sum() + 1e-12)
        idx = np.random.choice(len(applicable), p=probs)
        rule = applicable[idx]

        # Expand each RHS category
        children = []
        for rhs_cat in rule.rhs:
            child = self._expand(rhs_cat, depth + 1, max_depth)
            if child is None:
                return None
            children.append(child)

        node = SyntaxNode(
            category=category,
            children=children,
            rule_id=rule.rule_id,
            gamma_structural=rule.gamma(),
            z_impedance=rule.z_impedance,
        )
        rule.usage_count += 1
        return node

    # ------------------------------------------------------------------
    # Rule Learning from Wernicke Chunks
    # ------------------------------------------------------------------

    def learn_from_chunks(self, chunks: Optional[List] = None) -> Dict[str, Any]:
        """
        Discover phrase rules from Wernicke's mature chunks.

        A mature chunk (e.g., ("the", "cat") with count ≥ 3, gamma < 0.3)
        suggests a phrase rule.

        Algorithm:
          1. For each mature chunk, look up categories of each concept
          2. Form candidate rule: inferred_LHS → categories
          3. If rule doesn't exist, add it
          4. If rule exists, reinforce it
        """
        if chunks is None and self.wernicke is not None:
            chunks = [
                c for c in self.wernicke.chunks.values()
                if c.is_mature
            ]
        if not chunks:
            return {"rules_learned": 0, "rules_reinforced": 0}

        learned = 0
        reinforced = 0

        for chunk in chunks:
            concepts = chunk.concepts if hasattr(chunk, 'concepts') else chunk
            if isinstance(concepts, tuple):
                concepts = list(concepts)

            # Get categories for each concept
            cat_sequences = []
            for label in concepts:
                cats = self.get_categories(str(label))
                if not cats:
                    cats = ["N"]  # Default
                cat_sequences.append(cats)

            # Primary category assignment
            primary_cats = tuple(cs[0] for cs in cat_sequences)

            # Infer LHS category
            lhs = self._infer_lhs(primary_cats)
            if lhs is None:
                continue

            # Check if rule already exists
            existing = False
            for rule in self.rules:
                if rule.lhs == lhs and rule.rhs == primary_cats:
                    rule.reinforce()
                    reinforced += 1
                    existing = True
                    break

            if not existing:
                # Create new rule with moderate confidence
                conf = 0.3 + 0.1 * getattr(chunk, 'occurrence_count', 1)
                self._add_rule(lhs, primary_cats, confidence=min(conf, 2.0))
                learned += 1

        return {"rules_learned": learned, "rules_reinforced": reinforced}

    def _infer_lhs(self, rhs_categories: Tuple[str, ...]) -> Optional[str]:
        """
        Infer the LHS category from the RHS pattern.

        Heuristics:
          (Det, N) → NP
          (N,)     → NP
          (V,)     → VP
          (V, NP)  → VP
          (P, NP)  → PP
          (NP, VP) → S
          (Adj, N) → NP
        """
        if len(rhs_categories) == 0:
            return None

        # Check common patterns
        first = rhs_categories[0]
        last = rhs_categories[-1]

        if first == "Det" and last == "N":
            return "NP"
        if first == "Adj" and last == "N":
            return "NP"
        if len(rhs_categories) == 1 and first == "N":
            return "NP"
        if first == "V":
            return "VP"
        if first == "P":
            return "PP"
        if first == "NP" and last == "VP":
            return "S"
        if first == "NP" and last == "PP":
            return "NP"
        if first == "Conj":
            return "CP"
        if first == "Pro" and last == "VP":
            return "REL"

        # Default: most general category
        return "S" if len(rhs_categories) >= 2 else None

    # ------------------------------------------------------------------
    # Prosody Planning
    # ------------------------------------------------------------------

    def plan_prosody(self, tree: SyntaxNode) -> ProsodyPlan:
        """
        Derive prosodic parameters from syntax tree structure.

        Physics:
          - Deeper nodes → lower pitch (subordinate clauses)
          - Clause boundaries (S, CP, REL) → pauses
          - Higher confidence rules → smoother prosody

        This is why robots sound unnatural:
        they speak without prosodic structure.
        """
        plan = ProsodyPlan()
        self._walk_prosody(tree, depth=0, plan=plan)
        return plan

    def _walk_prosody(self, node: SyntaxNode, depth: int, plan: ProsodyPlan):
        """Recursive tree walk to generate prosody entries."""
        if node.is_terminal:
            pitch_factor = 1.0 - PROSODY_PITCH_DECAY * depth
            pitch_factor = max(0.5, pitch_factor)

            volume_factor = 1.0 - 0.03 * depth
            volume_factor = max(0.6, volume_factor)

            # Pause after clause-ending terminals
            pause_after = 0.0
            # Will be adjusted by parent check below

            plan.entries.append({
                "label": node.label or "?",
                "category": node.category,
                "depth": depth,
                "pitch_factor": round(pitch_factor, 3),
                "volume_factor": round(volume_factor, 3),
                "pause_after": 0.0,  # Updated by parent
            })
            return

        for i, child in enumerate(node.children):
            self._walk_prosody(child, depth + 1, plan)

            # Add pause after last terminal of clause-boundary nodes
            if (node.category in PROSODY_CLAUSE_CATEGORIES
                    and i < len(node.children) - 1
                    and plan.entries):
                plan.entries[-1]["pause_after"] = round(PROSODY_PAUSE_BASE * (1 + depth * 0.1), 3)

    # ------------------------------------------------------------------
    # Speak Tree (Integration with Broca)
    # ------------------------------------------------------------------

    def speak_tree(self, tree: SyntaxNode,
                   broca=None, mouth=None,
                   semantic_field=None) -> Dict[str, Any]:
        """
        Sequentially speak all terminals of a syntax tree.

        Algorithm:
          1. Get prosody plan
          2. For each terminal (left-to-right):
             a. Look up Broca articulatory plan
             b. Modify pitch/volume based on prosody
             c. Execute via Broca/mouth
             d. Add pause if needed
          3. Return aggregate results

        This is how the grammar engine PRODUCES structured speech:
        the tree determines WHAT to say and IN WHAT ORDER,
        Broca determines HOW to say each word.
        """
        if broca is None:
            broca = self.broca
        if broca is None:
            return {"error": "no broca engine", "terminals": tree.get_yield()}

        prosody = self.plan_prosody(tree)
        terminals = tree.get_terminals()

        results = []
        total_duration = 0.0

        for i, (terminal, prosody_entry) in enumerate(zip(terminals, prosody.entries)):
            label = terminal.label
            if label is None:
                continue

            # Get or create articulatory plan
            plan = broca.plan_utterance(label)

            result_entry = {
                "label": label,
                "category": terminal.category,
                "pitch_factor": prosody_entry["pitch_factor"],
                "volume_factor": prosody_entry["volume_factor"],
                "pause_after": prosody_entry["pause_after"],
                "plan_exists": plan is not None,
            }

            if plan and mouth:
                # Modify prosody
                original_pitch = plan.pitch
                original_volume = plan.volume
                plan.pitch *= prosody_entry["pitch_factor"]
                plan.volume *= prosody_entry["volume_factor"]
                plan.pitch = float(np.clip(plan.pitch, 80.0, 400.0))
                plan.volume = float(np.clip(plan.volume, 0.1, 1.0))

                exec_result = broca.execute_plan(
                    plan, mouth, ram_temperature=0.0
                )
                result_entry["executed"] = True
                result_entry["gamma_loop"] = exec_result.get("gamma_loop", 0.5)

                # Restore original values
                plan.pitch = original_pitch
                plan.volume = original_volume

                total_duration += plan.duration_steps + prosody_entry["pause_after"]
            else:
                result_entry["executed"] = False

            results.append(result_entry)

        return {
            "terminals_spoken": len(results),
            "total_duration": round(total_duration, 3),
            "prosody": prosody.to_dict(),
            "results": results,
            "yield": tree.get_yield(),
        }

    # ------------------------------------------------------------------
    # Structural Prediction (Integration with Wernicke)
    # ------------------------------------------------------------------

    def predict_structural(self, context_categories: List[str]) -> Dict[str, Any]:
        """
        Predict the next expected category based on current parse state.

        Uses the grammar rules to determine what categories could legally
        follow the current context.

        This is how grammar constrains prediction:
        After "Det", only "N" or "Adj" are expected (not "V").
        """
        if not context_categories:
            return {"expected": list(TERMINAL_CATEGORIES), "confidence": 0.0}

        last_cat = context_categories[-1]

        # Find all rules where last_cat appears in RHS
        expected = Counter()
        for rule in self.rules:
            for i, rhs_cat in enumerate(rule.rhs):
                if rhs_cat == last_cat and i < len(rule.rhs) - 1:
                    next_cat = rule.rhs[i + 1]
                    expected[next_cat] += rule.confidence

        if not expected:
            return {"expected": list(TERMINAL_CATEGORIES), "confidence": 0.0}

        # Normalize
        total = sum(expected.values())
        result = {
            cat: round(score / total, 4)
            for cat, score in expected.most_common(5)
        }
        return {
            "expected": result,
            "confidence": round(max(expected.values()) / total, 4),
        }

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> Dict[str, Any]:
        """
        Per-cycle maintenance:
          - Decay rule confidences (use-it-or-lose-it)
          - Learn from Wernicke chunks if available
        """
        for rule in self.rules:
            rule.decay()

        chunk_result = None
        if self.wernicke:
            chunk_result = self.learn_from_chunks()

        return {
            "n_rules": len(self.rules),
            "lexicon_size": len(self.lexicon),
            "chunk_learning": chunk_result,
        }

    # ------------------------------------------------------------------
    # State / Stats
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "n_rules": len(self.rules),
            "lexicon_size": len(self.lexicon),
            "total_parses": self.total_parses,
            "successful_parses": self.successful_parses,
            "parse_success_rate": round(
                self.successful_parses / max(1, self.total_parses), 3
            ),
            "total_generations": self.total_generations,
            "total_rules_learned": self.total_rules_learned,
            "total_garden_paths": self.total_garden_paths,
            "top_rules": [
                r.to_dict() for r in sorted(self.rules, key=lambda r: r.confidence, reverse=True)[:10]
            ],
        }

    def get_stats(self) -> Dict[str, Any]:
        return self.get_state()
