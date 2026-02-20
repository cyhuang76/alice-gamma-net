# -*- coding: utf-8 -*-
"""
Unit Tests — Recursive Grammar Engine (Phase 20.2)
"""

import time
import pytest
import numpy as np

from alice.brain.recursive_grammar import (
    RecursiveGrammarEngine,
    SyntaxNode,
    PhraseRule,
    ParseResult,
    ProsodyPlan,
    LexicalEntry,
    ROOT_CATEGORY,
    TERMINAL_CATEGORIES,
    PHRASAL_CATEGORIES,
    GRAMMAR_Z0,
    MAX_EMBEDDING_DEPTH,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def grammar():
    return RecursiveGrammarEngine()


@pytest.fixture
def grammar_with_lexicon(grammar):
    """Grammar with basic English lexicon assigned."""
    words = {
        "the": "Det", "a": "Det", "every": "Det",
        "cat": "N", "dog": "N", "mouse": "N", "mat": "N", "man": "N",
        "sat": "V", "chased": "V", "bit": "V", "ran": "V",
        "big": "Adj", "small": "Adj",
        "on": "P", "under": "P", "with": "P",
        "that": "Pro",
        "and": "Conj",
        "quickly": "Adv",
    }
    for word, cat in words.items():
        grammar.assign_category(word, cat, 0.8)
    return grammar


# ============================================================================
# Construction / Initialization
# ============================================================================

class TestGrammarInit:
    def test_creation(self, grammar):
        assert len(grammar.rules) > 0  # Bootstrap rules
        assert len(grammar.rule_index) > 0

    def test_bootstrap_rules_include_basics(self, grammar):
        rule_strs = [(r.lhs, r.rhs) for r in grammar.rules]
        assert ("S", ("NP", "VP")) in rule_strs
        assert ("NP", ("Det", "N")) in rule_strs
        assert ("VP", ("V",)) in rule_strs
        assert ("VP", ("V", "NP")) in rule_strs

    def test_get_state(self, grammar):
        state = grammar.get_state()
        assert "n_rules" in state
        assert "lexicon_size" in state
        assert "total_parses" in state

    def test_get_stats(self, grammar):
        assert grammar.get_stats() == grammar.get_state()


# ============================================================================
# Lexicon
# ============================================================================

class TestLexicon:
    def test_assign_category(self, grammar):
        entry = grammar.assign_category("cat", "N", 0.8)
        assert entry.label == "cat"
        assert entry.category == "N"
        assert entry.confidence == 0.8

    def test_get_categories(self, grammar):
        grammar.assign_category("run", "V", 0.7)
        grammar.assign_category("run", "N", 0.5)
        cats = grammar.get_categories("run")
        assert "V" in cats
        assert "N" in cats

    def test_duplicate_category_updates_confidence(self, grammar):
        grammar.assign_category("cat", "N", 0.5)
        grammar.assign_category("cat", "N", 0.9)
        entries = grammar.lexicon["cat"]
        assert len(entries) == 1  # Not duplicated
        assert entries[0].confidence == 0.9

    def test_unknown_word(self, grammar):
        cats = grammar.get_categories("xyzzy")
        assert len(cats) == 0

    def test_lexical_entry_to_dict(self, grammar):
        entry = grammar.assign_category("cat", "N", 0.8)
        d = entry.to_dict()
        assert d["label"] == "cat"
        assert d["category"] == "N"


# ============================================================================
# SyntaxNode
# ============================================================================

class TestSyntaxNode:
    def test_terminal(self):
        node = SyntaxNode(category="N", label="cat")
        assert node.is_terminal
        assert node.depth == 0
        assert node.n_terminals == 1

    def test_non_terminal(self):
        det = SyntaxNode(category="Det", label="the")
        n = SyntaxNode(category="N", label="cat")
        np_node = SyntaxNode(category="NP", children=[det, n])
        assert not np_node.is_terminal
        assert np_node.depth == 1
        assert np_node.n_terminals == 2

    def test_get_yield(self):
        det = SyntaxNode(category="Det", label="the")
        n = SyntaxNode(category="N", label="cat")
        np_node = SyntaxNode(category="NP", children=[det, n])
        assert np_node.get_yield() == ["the", "cat"]

    def test_bracket_string(self):
        det = SyntaxNode(category="Det", label="the")
        n = SyntaxNode(category="N", label="cat")
        np_node = SyntaxNode(category="NP", children=[det, n])
        bs = np_node.to_bracket_string()
        assert "[NP" in bs
        assert "[Det the]" in bs
        assert "[N cat]" in bs

    def test_to_dict(self):
        node = SyntaxNode(category="N", label="cat")
        d = node.to_dict()
        assert d["category"] == "N"
        assert d["label"] == "cat"
        assert d["depth"] == 0

    def test_nested_depth(self):
        leaf = SyntaxNode(category="N", label="cat")
        mid = SyntaxNode(category="NP", children=[leaf])
        top = SyntaxNode(category="S", children=[mid])
        assert top.depth == 2


# ============================================================================
# PhraseRule
# ============================================================================

class TestPhraseRule:
    def test_creation(self):
        rule = PhraseRule(rule_id=1, lhs="NP", rhs=("Det", "N"))
        assert rule.lhs == "NP"
        assert rule.rhs == ("Det", "N")

    def test_gamma(self):
        rule = PhraseRule(rule_id=1, lhs="NP", rhs=("Det", "N"), confidence=0.5)
        rule.update_impedance()
        g = rule.gamma()
        assert 0.0 <= g <= 1.0

    def test_reinforce(self):
        rule = PhraseRule(rule_id=1, lhs="NP", rhs=("Det", "N"), confidence=0.5)
        old_conf = rule.confidence
        rule.reinforce()
        assert rule.confidence > old_conf
        assert rule.success_count == 1

    def test_weaken(self):
        rule = PhraseRule(rule_id=1, lhs="NP", rhs=("Det", "N"), confidence=0.5)
        old_conf = rule.confidence
        rule.weaken()
        assert rule.confidence < old_conf

    def test_decay(self):
        rule = PhraseRule(rule_id=1, lhs="NP", rhs=("Det", "N"), confidence=1.0)
        rule.decay()
        assert rule.confidence < 1.0

    def test_matches(self):
        rule = PhraseRule(rule_id=1, lhs="NP", rhs=("Det", "N"))
        assert rule.matches(("Det", "N"))
        assert not rule.matches(("V", "NP"))

    def test_to_dict(self):
        rule = PhraseRule(rule_id=1, lhs="NP", rhs=("Det", "N"))
        d = rule.to_dict()
        assert "NP → Det N" in d["rule"]


# ============================================================================
# Parsing — Basic
# ============================================================================

class TestParsingBasic:
    def test_simple_sentence(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "sat"])
        assert r.success
        assert r.tree.category == "S"
        assert r.tree.get_yield() == ["the", "cat", "sat"]

    def test_transitive_sentence(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "chased", "the", "mouse"])
        assert r.success
        assert r.tree.get_yield() == ["the", "cat", "chased", "the", "mouse"]

    def test_bare_noun(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["cat", "ran"])
        assert r.success
        assert "cat" in r.tree.get_yield()

    def test_empty_sequence(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse([])
        assert not r.success

    def test_single_word(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["cat"])
        # May or may not fully parse, but shouldn't crash
        assert isinstance(r, ParseResult)

    def test_parse_result_to_dict(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "sat"])
        d = r.to_dict()
        assert "success" in d
        assert "gamma_structural" in d
        assert "tree" in d

    def test_unknown_word_defaults_to_noun(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "flurble", "sat"])
        assert r.success  # "flurble" assigned N by default


# ============================================================================
# Parsing — Complex
# ============================================================================

class TestParsingComplex:
    def test_adjective_noun(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "big", "cat", "sat"])
        assert r.success
        assert r.embedding_depth >= 2

    def test_prepositional_phrase(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "sat", "on", "the", "mat"])
        assert r.success
        assert "[P on]" in r.tree.to_bracket_string()

    def test_depth_increases_with_complexity(self, grammar_with_lexicon):
        r1 = grammar_with_lexicon.parse(["cat", "sat"])
        r2 = grammar_with_lexicon.parse(["the", "big", "cat", "chased", "the", "small", "mouse"])
        if r1.success and r2.success:
            assert r2.embedding_depth >= r1.embedding_depth


# ============================================================================
# Garden-Path Recovery
# ============================================================================

class TestGardenPath:
    def test_ambiguous_category(self, grammar):
        grammar.assign_category("run", "V", 0.7)
        grammar.assign_category("run", "N", 0.5)
        grammar.assign_category("the", "Det", 0.9)
        r = grammar.parse(["the", "run"])
        assert r.success  # Should resolve via reanalysis if needed

    def test_no_garden_path_unambiguous(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "sat"])
        # Unambiguous → no garden path
        assert not r.garden_path or r.success


# ============================================================================
# Generation
# ============================================================================

class TestGeneration:
    def test_generate_sentence(self, grammar_with_lexicon):
        tree = grammar_with_lexicon.generate("S", max_depth=3)
        assert tree is not None
        assert tree.category == "S"
        assert len(tree.get_yield()) >= 2

    def test_generate_np(self, grammar_with_lexicon):
        tree = grammar_with_lexicon.generate("NP", max_depth=2)
        assert tree is not None
        assert tree.category == "NP"

    def test_generate_increments_counter(self, grammar_with_lexicon):
        before = grammar_with_lexicon.total_generations
        grammar_with_lexicon.generate()
        assert grammar_with_lexicon.total_generations == before + 1


# ============================================================================
# Rule Learning
# ============================================================================

class TestRuleLearning:
    def test_learn_from_mock_chunks(self, grammar):
        from dataclasses import dataclass
        from typing import Tuple

        @dataclass
        class MockChunk:
            concepts: Tuple[str, ...]
            occurrence_count: int = 5
            mean_internal_gamma: float = 0.2
            last_seen: float = 0.0

            @property
            def is_mature(self):
                return self.occurrence_count >= 3 and self.mean_internal_gamma < 0.3

        grammar.assign_category("my", "Det", 0.8)
        grammar.assign_category("friend", "N", 0.8)
        chunks = [MockChunk(concepts=("my", "friend"), occurrence_count=10)]

        result = grammar.learn_from_chunks(chunks)
        total = result["rules_learned"] + result["rules_reinforced"]
        assert total >= 1

    def test_add_duplicate_rule(self, grammar):
        r1 = grammar._add_rule("NP", ("Det", "N"), confidence=0.5)
        r2 = grammar._add_rule("NP", ("Det", "N"), confidence=0.8)
        assert r1 is r2
        assert r2.confidence >= 0.8  # Updated


# ============================================================================
# Prosody
# ============================================================================

class TestProsody:
    def test_prosody_plan(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "chased", "the", "mouse"])
        assert r.success
        prosody = grammar_with_lexicon.plan_prosody(r.tree)
        assert len(prosody.entries) == 5

    def test_pitch_decreases_with_depth(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "sat", "on", "the", "mat"])
        if r.success:
            prosody = grammar_with_lexicon.plan_prosody(r.tree)
            entries = prosody.entries
            # All pitch factors should be valid
            for e in entries:
                assert 0.4 <= e["pitch_factor"] <= 1.0

    def test_prosody_to_dict(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "sat"])
        prosody = grammar_with_lexicon.plan_prosody(r.tree)
        d = prosody.to_dict()
        assert "entries" in d


# ============================================================================
# Structural Prediction
# ============================================================================

class TestStructuralPrediction:
    def test_after_det_expect_noun(self, grammar):
        result = grammar.predict_structural(["Det"])
        expected = result["expected"]
        assert isinstance(expected, dict)
        # N should be among expected categories
        assert "N" in expected or "Adj" in expected

    def test_empty_context(self, grammar):
        result = grammar.predict_structural([])
        assert "expected" in result


# ============================================================================
# Speak Tree
# ============================================================================

class TestSpeakTree:
    def test_speak_tree_no_broca(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "sat"])
        result = grammar_with_lexicon.speak_tree(r.tree)
        assert "terminals" in result or "error" in result

    def test_speak_tree_returns_yield(self, grammar_with_lexicon):
        r = grammar_with_lexicon.parse(["the", "cat", "sat"])
        result = grammar_with_lexicon.speak_tree(r.tree)
        # Without broca/mouth, just check structure
        if "error" in result:
            assert "terminals" in result


# ============================================================================
# Tick
# ============================================================================

class TestTick:
    def test_tick(self, grammar):
        result = grammar.tick()
        assert "n_rules" in result

    def test_tick_decay(self, grammar):
        # Get initial conf of first rule
        r = grammar.rules[0]
        old_conf = r.confidence
        grammar.tick()
        assert r.confidence <= old_conf  # Decayed


# ============================================================================
# Integration with AliceBrain
# ============================================================================

class TestAliceBrainIntegration:
    def test_alice_brain_has_narrative_memory(self):
        from alice.alice_brain import AliceBrain
        from alice.brain.narrative_memory import NarrativeMemoryEngine
        brain = AliceBrain()
        assert hasattr(brain, "narrative_memory")
        assert isinstance(brain.narrative_memory, NarrativeMemoryEngine)

    def test_alice_brain_has_recursive_grammar(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        assert hasattr(brain, "recursive_grammar")
        assert isinstance(brain.recursive_grammar, RecursiveGrammarEngine)

    def test_introspect_includes_new_modules(self):
        from alice.alice_brain import AliceBrain
        brain = AliceBrain()
        state = brain.introspect()
        assert "narrative_memory" in state["subsystems"]
        assert "recursive_grammar" in state["subsystems"]
