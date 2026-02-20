# -*- coding: utf-8 -*-
"""Alice Smart System — Brain Module"""

from alice.brain.fusion_brain import (
    BrainRegionType,
    Neuron,
    BrainRegion,
    NeuralMessage,
    ProtocolNeuralAdapter,
    FusionBrain,
)

from alice.brain.perception import (
    PhysicalTuner,
    ConceptMemory,
    PerceptionPipeline,
    PerceptionResult,
    BAND_INFO_LAYER,
)

from alice.brain.pruning import (
    SynapticConnection,
    CorticalSpecialization,
    CorticalRegion as PrunableCorticalRegion,
    NeuralPruningEngine,
    MODALITY_SIGNAL_PROFILE,
)

from alice.brain.sleep_physics import (
    SleepPhysicsEngine,
    ImpedanceDebtTracker,
    SynapticEntropyTracker,
    SlowWaveOscillator,
    REMDreamDiagnostic,
    SleepQualityReport,
)

from alice.brain.auditory_grounding import (
    AuditoryGroundingEngine,
    CrossModalHebbianNetwork,
    CrossModalSynapse,
)

from alice.brain.semantic_field import (
    SemanticFieldEngine,
    SemanticField,
    SemanticAttractor,
    gamma_semantic,
)

from alice.brain.broca import (
    BrocaEngine,
    ArticulatoryPlan,
    extract_formants,
)

from alice.brain.hippocampus import (
    HippocampusEngine,
    Episode,
    EpisodicSnapshot,
)

from alice.brain.wernicke import (
    WernickeEngine,
    TransitionMatrix,
    Chunk,
)

from alice.brain.thalamus import (
    ThalamusEngine,
    SensoryChannel,
    ThalamicGateResult,
)

from alice.brain.amygdala import (
    AmygdalaEngine,
    FearMemory,
    EmotionalState,
    AmygdalaResponse,
)

from alice.brain.emotion_granularity import (
    EmotionGranularityEngine,
    EmotionVector,
    GranularEmotionState,
)

from alice.brain.prefrontal import (
    PrefrontalCortexEngine,
    Goal,
    ActionProposal,
    PlanStep,
    GoNoGoDecision,
    TaskSwitchResult,
)

from alice.brain.basal_ganglia import (
    BasalGangliaEngine,
    ActionChannel,
    SelectionResult,
    HabitSnapshot,
)

from alice.brain.curiosity_drive import (
    CuriosityDriveEngine,
    NoveltyEvent,
    SpontaneousAction,
    SelfOtherJudgment,
    InternalGoal,
    SpontaneousActionType,
)

from alice.brain.predictive_engine import (
    PredictiveEngine,
    ForwardModel,
    WorldState,
    SimulationPath,
    PredictionResult,
)

from alice.brain.metacognition import (
    MetacognitionEngine,
    MetacognitionResult,
    CognitiveSnapshot,
    CounterfactualResult,
)

from alice.brain.social_resonance import (
    SocialResonanceEngine,
    Belief,
    SocialAgentModel,
    SocialCouplingResult,
    SallyAnneResult,
    SocialHomeostasisState,
)

from alice.brain.narrative_memory import (
    NarrativeMemoryEngine,
    NarrativeArc,
    CausalLink,
    EpisodeSummary,
)

from alice.brain.recursive_grammar import (
    RecursiveGrammarEngine,
    SyntaxNode,
    PhraseRule,
    ParseResult,
    ProsodyPlan,
    LexicalEntry,
)
from alice.brain.semantic_pressure import (
    SemanticPressureEngine,
    InnerMonologueEvent,
)
from alice.brain.homeostatic_drive import (
    HomeostaticDriveEngine,
    HomeostaticState,
    HomeostaticDriveSignal,
)
from alice.brain.physics_reward import (
    PhysicsRewardEngine,
    RewardChannel,
    RewardExperience,
)
from alice.brain.pinch_fatigue import (
    PinchFatigueEngine,
    ConductorState,
    PinchEvent,
    AgingSignal,
)
from alice.brain.phantom_limb import (
    PhantomLimbEngine,
    PhantomLimbState,
    PhantomPainEvent,
    AmputationRecord,
)

from alice.brain.clinical_neurology import (
    ClinicalNeurologyEngine,
    StrokeModel,
    StrokeEvent,
    ALSModel,
    ALSState,
    DementiaModel,
    DementiaState,
    AlzheimersModel,
    AlzheimersState,
    CerebralPalsyModel,
    CerebralPalsyState,
)

from alice.brain.pharmacology import (
    PharmacologyEngine,
    DrugProfile,
    MSModel,
    MSState,
    MSLesion,
    ParkinsonModel,
    PDState,
    EpilepsyModel,
    EpilepsyState,
    DepressionModel,
    DepressionState,
    ClinicalPharmacologyEngine,
)

__all__ = [
    # Fusion Brain
    "BrainRegionType",
    "Neuron",
    "BrainRegion",
    "NeuralMessage",
    "ProtocolNeuralAdapter",
    "FusionBrain",
    # Perception Pipeline (physics-driven)
    "PhysicalTuner",
    "ConceptMemory",
    "PerceptionPipeline",
    "PerceptionResult",
    "BAND_INFO_LAYER",
    # Neural Pruning (§3.5.2 large-scale Γ apoptosis)
    "SynapticConnection",
    "CorticalSpecialization",
    "PrunableCorticalRegion",
    "NeuralPruningEngine",
    "MODALITY_SIGNAL_PROFILE",
    # Sleep Physics Engine (offline impedance reorganization)
    "SleepPhysicsEngine",
    "ImpedanceDebtTracker",
    "SynapticEntropyTracker",
    "SlowWaveOscillator",
    "REMDreamDiagnostic",
    "SleepQualityReport",
    # Auditory Grounding Engine (Phase 4.1 language physicalization)
    "AuditoryGroundingEngine",
    "CrossModalHebbianNetwork",
    "CrossModalSynapse",
    # Semantic Field (Phase 4.2)
    "SemanticFieldEngine",
    "SemanticField",
    "SemanticAttractor",
    "gamma_semantic",
    # Broca's Area (Phase 4.3)
    "BrocaEngine",
    "ArticulatoryPlan",
    "extract_formants",
    # Hippocampus (Phase 5.1)
    "HippocampusEngine",
    "Episode",
    "EpisodicSnapshot",
    # Wernicke's Area (Phase 5.2)
    "WernickeEngine",
    "TransitionMatrix",
    "Chunk",
    # Thalamus (Phase 5.3)
    "ThalamusEngine",
    "SensoryChannel",
    "ThalamicGateResult",
    # Amygdala (Phase 5.4)
    "AmygdalaEngine",
    "FearMemory",
    "EmotionalState",
    "AmygdalaResponse",
    # Emotion Granularity Engine (Phase 36 — Plutchik 8D emotion vector)
    "EmotionGranularityEngine",
    "EmotionVector",
    "GranularEmotionState",
    # Prefrontal Cortex (Phase 6.1)
    "PrefrontalCortexEngine",
    "Goal",
    "ActionProposal",
    "PlanStep",
    "GoNoGoDecision",
    "TaskSwitchResult",
    # Basal Ganglia (Phase 6.2)
    "BasalGangliaEngine",
    "ActionChannel",
    "SelectionResult",
    "HabitSnapshot",
    # Curiosity Drive Engine (Phase 9 — free will & self-awareness)
    "CuriosityDriveEngine",
    "NoveltyEvent",
    "SpontaneousAction",
    "SelfOtherJudgment",
    "InternalGoal",
    "SpontaneousActionType",
    # Predictive Processing Engine (Phase 17 — eye of time)
    "PredictiveEngine",
    "ForwardModel",
    "WorldState",
    "SimulationPath",
    "PredictionResult",
    # Metacognition Engine (Phase 18 — inner auditor)
    "MetacognitionEngine",
    "MetacognitionResult",
    "CognitiveSnapshot",
    "CounterfactualResult",
    # Social Resonance Engine (Phase 19 — social physics field)
    "SocialResonanceEngine",
    "Belief",
    "SocialAgentModel",
    "SocialCouplingResult",
    "SallyAnneResult",
    "SocialHomeostasisState",
    # Narrative Memory Engine (Phase 20.1 — autobiographical memory weaving)
    "NarrativeMemoryEngine",
    "NarrativeArc",
    "CausalLink",
    "EpisodeSummary",
    # Recursive Grammar Engine (Phase 20.2 — phrase structure recursion)
    "RecursiveGrammarEngine",
    "SyntaxNode",
    "PhraseRule",
    "ParseResult",
    "ProsodyPlan",
    "LexicalEntry",
    # Semantic Pressure Engine (Phase 21 — language thermodynamics integration)
    "SemanticPressureEngine",
    "InnerMonologueEvent",
    # Hypothalamic Homeostatic Drive (Phase 22.1 — hunger/thirst)
    "HomeostaticDriveEngine",
    "HomeostaticState",
    "HomeostaticDriveSignal",
    # Physics Reward Engine (Phase 22.2 — impedance matching reward)
    "PhysicsRewardEngine",
    "RewardChannel",
    "RewardExperience",
    # Pinch Fatigue Engine (Phase 23 — Pollock-Barraclough neural aging)
    "PinchFatigueEngine",
    "ConductorState",
    "PinchEvent",
    "AgingSignal",
    # Phantom Limb Engine (Phase 24 — Γ=1.0 open-circuit physics)
    "PhantomLimbEngine",
    "PhantomLimbState",
    "PhantomPainEvent",
    "AmputationRecord",
    # Clinical Neurology Engine (Phase 25 — unified physics of five neurological diseases)
    "ClinicalNeurologyEngine",
    "StrokeModel",
    "StrokeEvent",
    "ALSModel",
    "ALSState",
    "DementiaModel",
    "DementiaState",
    "AlzheimersModel",
    "AlzheimersState",
    "CerebralPalsyModel",
    "CerebralPalsyState",
    # Computational Pharmacology Engine (Phase 26 — unified α_drug + four neurological diseases)
    "PharmacologyEngine",
    "DrugProfile",
    "MSModel",
    "MSState",
    "MSLesion",
    "ParkinsonModel",
    "PDState",
    "EpilepsyModel",
    "EpilepsyState",
    "DepressionModel",
    "DepressionState",
    "ClinicalPharmacologyEngine",
]
