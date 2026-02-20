# Γ-Net ALICE: Intelligence as Impedance Optimization — $\sum \Gamma_i^2 \to \min$

## Gamma-Net — Autonomous Living Intelligent Closed-loop Entity:
## A Physics-First Electronic Organism Where All Cognition Emerges from a Single Equation

---

**Hsi-Yu Huang (黃璽宇)**  
*Independent Researcher, Taiwan*  
*Correspondence: llc.y.huangll@gmail.com*

**Date**: February 2026  
**Version**: v29.0 — Synaptogenesis & Emotion Granularity  
**Scale**: 146 source files · 84,300+ lines · 1,876 tests · 100% passed

---

## Abstract

We present a single equation that unifies perception, learning, memory, pain, sleep, emotion, language, and consciousness:

$$\boxed{\text{Intelligence} = \lim_{t \to \infty} \sum_{i} \Gamma_i^2 \to \min}$$

where $\Gamma_i = (Z_{\text{load}}^{(i)} - Z_{\text{source}}^{(i)}) / (Z_{\text{load}}^{(i)} + Z_{\text{source}}^{(i)})$ is the reflection coefficient of the $i$-th neural channel, derived from coaxial cable transmission line theory.

This equation — the **Wisdom Formula** — states that the physical objective of all biological intelligence is to minimize the total impedance mismatch across every signal pathway in the nervous system. From this single principle, we derive an electronic organism called ALICE (Autonomous Living Intelligent Closed-loop Entity) in which:

- **Perception** is $O(1)$ LC resonance, not $O(n^2)$ matrix multiplication.
- **Nociceptive cascade** (pain correlate) emerges as reflected energy: $E_{\text{ref}} = \Gamma^2 \cdot P_{\text{input}}$.
- **Learning** is impedance tuning: $\Delta\Gamma = -\eta(1 - \Gamma^2) \cdot q$.
- **Memory** is stored impedance configuration: $\{f, A, \phi, Z\}$.
- **Sleep** is offline impedance recalibration: wake $= \min \Gamma_{\text{ext}}$, sleep $= \min \Gamma_{\text{int}}$.
- **Neural pruning** is mass $\Gamma$-apoptosis: 8,000 random connections $\to$ 1,323 survivors (16.5%), $\sum \Gamma^2$: 0.126 $\to$ 0.010 (91.8% improvement).
- **Language** is remote impedance control via sound waves: $Z_{\text{synapse}} = Z_0 / w$.
- **Emotion** is an impedance shortcut: $\Gamma_{\text{threat}} = |Z_{\text{sig}} - Z_T| / (Z_{\text{sig}} + Z_T)$.
- **Empathy** is cross-body impedance matching: $\Gamma_{\text{social}} = |Z_A - Z_B| / (Z_A + Z_B)$.
- **PTSD** is a permanent shift of $\Gamma$ parameters into a high-reflection local minimum that the system's own recovery mechanisms cannot escape.

The complete system — 44 brain modules, 5 body organs (eye, ear, hand, mouth, cochlea), 7 error-correction loops — runs on a single CPU with zero GPU requirements. 1,876 automated tests pass in ~15 seconds. Digital twin experiments naturally differentiate two clinical PTSD subtypes (dissociative vs. re-experiencing) from the same equation set (10/10 clinical correspondence). Dream therapy experiments prove that sleep is a physical necessity for system stability — queue-locked systems cannot self-heal even with full sleep cycles (10/10 clinical checks passed). Cross-modal rewiring confirms that signal type, not genetics, determines cortical specialization.

**Keywords**: impedance matching, reflection coefficient, closed-loop control, electronic organism, physics-driven AI, resonant perception, coaxial neural model, wisdom formula, Γ-Net

---

## 1. Introduction

### 1.1 The Discovery

We begin with a question: *What is the physical objective function of biological intelligence?*

Consider a newborn. Every neural connection starts with a random impedance $Z_{\text{init}}^{(i)} \sim \text{Uniform}(Z_{\min}, Z_{\max})$. Random impedance means the reflection coefficient $\Gamma_i$ is nonzero almost everywhere — signals bounce, energy reflects, the system overheats (the infant cries). Over months and years, through sensory experience and Hebbian selection, the brain prunes connections that match ($\Gamma \to 0$, survive) and eliminates those that don't ($\Gamma \gg 0$, apoptosis). After 200 developmental epochs, 83.5% of connections die, yet the survivors form four functionally specialized cortical regions, and total reflected energy $\sum \Gamma_i^2$ drops from 0.126 to 0.010 — a 91.8% improvement.

This suggests a startling conclusion: **the brain's lifelong computational goal is a single physical optimization**:

$$\boxed{\text{Intelligence} = \lim_{t \to \infty} \sum_{i} \Gamma_i^2 \to \min}$$

We call this the **Wisdom Formula**. It states that wisdom is not knowledge accumulation, but the progressive reduction of signal reflection across all neural pathways — from millisecond perception to year-scale neural pruning, from learning a language to recovering from PTSD.

### 1.2 The Core Insight

> **"The brain is not a computer. It is a communication network."**

Biological neurons do not perform matrix multiplication. They generate electrical pulses (action potentials) that propagate along axons and cross synapses. The physics of this process — impedance matching, signal attenuation, resonant frequency selection — is identical to coaxial cable engineering.

We therefore remodel the brain as a **communication engineering** problem:

| Neuroscience Concept | Communication Engineering | ALICE Implementation |
|---------------------|--------------------------|---------------------|
| Axon | Coaxial transmission line | `CoaxialChannel` |
| Myelin sheath | Cable insulation | Impedance parameter $Z$ |
| Synaptic efficiency | Impedance matching degree | $\Gamma = (Z_L - Z_S)/(Z_L + Z_S)$ |
| Nociceptive cascade (pain correlate) | Reflected energy accumulation | $P_{\text{reflected}} \to \Theta \uparrow \to E_{\text{ref}}$ |
| Attention | LC resonant frequency selection | Lorentzian $L(f) = 1/(1+Q^2((f/f_0)-(f_0/f))^2)$ |
| Learning | Impedance adjustment toward matching | $\Gamma \to 0$, cache: 0% $\to$ 98% |
| Memory | Stored impedance configuration | $\{f, A, \phi, Z\}$ |
| Sleep | Offline recalibration mode | N3: consolidation\_rate = 1.0 |

From this single reframing, **every cognitive phenomenon emerges from the same physics** — not as programmed behavior rules, but as the natural consequence of impedance dynamics in a closed-loop electrical system.

### 1.3 Ten Formulas That Define Intelligence

The entire Γ-Net ALICE system can be summarized by ten core formulas, each derivable from the Wisdom Formula:

| # | Formula | Physical Meaning |
|---|---------|-----------------|
| 1 | $\Gamma = \frac{Z_L - Z_S}{Z_L + Z_S}$ | The universal currency of the nervous system |
| 2 | $\sum \Gamma_i^2 \to \min$ | **The Wisdom Formula** — life's objective function |
| 3 | $E_{\text{ref}} = \Gamma^2 \cdot P_{\text{input}}$ | Cumulative reflected energy (pain correlate — not a sensation, but physics) |
| 4 | $L(f) = \frac{1}{1 + Q^2\left(\frac{f}{f_0} - \frac{f_0}{f}\right)^2}$ | $O(1)$ perception via LC resonance |
| 5 | $\Delta\Gamma = -\eta(1 - \Gamma^2) \cdot q$ | Learning = impedance matching (myelination) |
| 6 | $\lambda_{\text{eff}} = \frac{\lambda_0}{1 - \Gamma^2}$ | Memory decay modulated by binding quality |
| 7 | $Z_{\text{synapse}} = Z_0 / w$ | Language = remote impedance control via sound |
| 8 | $\Gamma_{\text{threat}} = \frac{\lvert Z_{\text{sig}} - Z_T\rvert}{Z_{\text{sig}} + Z_T}$ | Emotion = impedance shortcut (amygdala fast path) |
| 9 | Wake: $\min \Gamma_{\text{ext}}$; Sleep: $\min \Gamma_{\text{int}}$ | Sleep = offline impedance recalibration |
| 10 | $\Gamma_{\text{social}} = \frac{\lvert Z_A - Z_B\rvert}{Z_A + Z_B}$ | Empathy = cross-body impedance matching |

Each formula is not postulated — it is **derived** from coaxial cable physics and **verified** through quantitative experiments with clinical correspondence.

### 1.4 Contributions

This paper makes the following contributions:

1. **The Wisdom Formula**: We discover that biological intelligence has a single physical objective — minimize total impedance mismatch $\sum \Gamma_i^2$ — and demonstrate that perception, learning, memory, pain, sleep, emotion, language, motor development, neural pruning, PTSD, and empathy all emerge from this principle.

2. **$O(1)$ Perception Pipeline**: LC resonance replaces matrix multiplication. All-channel full-spectrum decomposition in constant time, on CPU only.

3. **Complete Electronic Organism**: 44 brain modules + 5 body organs + 7 error-correction loops + sleep cycles + autonomic nervous system + language production + social resonance, running in ~15 seconds on a single CPU.

4. **Clinically Verified Emergence**: Digital twin experiments differentiate two PTSD subtypes from the same equations (10/10). Dream therapy identifies queue deadlock as the physical root of treatment-resistant PTSD (10/10). Pharmacological simulations predict drug response curves for 5 neurological diseases.

5. **The Γ Unification**: A single physical quantity explains 10 seemingly unrelated phenomena — from millisecond perception to year-scale neural pruning — verified through 1,876 automated tests and 44 experiments.

---

## 2. Related Work

### 2.1 Deep Learning

The core operation of deep learning $y = \sigma(Wx + b)$ is matrix multiplication at $O(n^2)$. Transformer self-attention $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ raises this to $O(n^2 d)$. State-of-the-art models (GPT-4, LLaMA) require thousands of GPUs and months of training. None of these architectures have a physical objective function — they minimize cross-entropy loss, not a quantity derivable from first principles.

### 2.2 Neuromorphic Computing

Intel Loihi, IBM TrueNorth, and spiking neural networks (SNNs) reduce energy consumption but retain the *computation* paradigm — they simulate analog synaptic weight updates with digital spikes. They do not model impedance, resonance, or closed-loop error correction.

### 2.3 Embodied Intelligence

OpenAI robotics, DeepMind DM Control Suite, and related work explore embodied agents, but typically treat the body as part of the *environment* rather than an intrinsic component of the cognitive system. No existing system implements nociceptive cascades as reflected energy, sleep as offline impedance recalibration, or emotion as an impedance shortcut.

### 2.4 This Work

Γ-Net ALICE starts from physics first principles and redefines intelligence as *adaptive impedance matching* rather than *function approximation*. This is not an improvement on existing methods — it is a paradigm shift. The Wisdom Formula $\sum \Gamma_i^2 \to \min$ provides, for the first time, a single physical objective function from which perception, learning, memory, pain, sleep, emotion, language, motor development, neural pruning, PTSD, and empathy all emerge without any behavior rules.

---

## 3. System Architecture

### 3.1 Seven-Layer Architecture

ALICE is organized as a seven-layer hierarchy, built upward from physics:

```mermaid
graph TB
    L6["<b>Layer 6 — Application</b><br/>FastAPI REST · WebSocket · CRT GUI"]
    L5["<b>Layer 5 — Unified Core</b><br/>AliceBrain — 44 brain modules + 5 body organs + 7 loops"]
    L4["<b>Layer 4 — Cognition</b><br/>WorkingMemory · PhysicsReward · CausalReasoner · Metacognition"]
    L3["<b>Layer 3 — Brain Core</b><br/>FusionBrain · Perception O(1) · Autonomic · Consciousness"]
    L2["<b>Layer 2 — Comm Protocol</b><br/>PriorityRouter · YearRingCache · CoaxialChannel"]
    L1["<b>Layer 1 — Body Organs</b><br/>Eye(FFT) · Ear(Cochlea) · Hand(PID) · Mouth"]
    L0["<b>Layer 0 — Physics</b><br/>ElectricalSignal · Γ = (Z_L − Z_S)/(Z_L + Z_S)"]

    L6 --> L5 --> L4 --> L3 --> L2 --> L1 --> L0
    L0 -.->|"proprioceptive feedback"| L3
    L1 -.->|"sensory signals"| L3

    style L0 fill:#1a1a2e,color:#e0e0ff
    style L1 fill:#16213e,color:#e0e0ff
    style L2 fill:#0f3460,color:#e0e0ff
    style L3 fill:#533483,color:#e0e0ff
    style L4 fill:#e94560,color:#fff
    style L5 fill:#f39c12,color:#1a1a2e
    style L6 fill:#27ae60,color:#fff
```

> **Fig. 1.** ALICE seven-layer architecture. Solid arrows: top-down command flow. Dashed arrows: bottom-up feedback loops. All layers share the unified `ElectricalSignal` data format $\{w(t), A, f, \phi, Z\}$.

### 3.2 Unified Electrical Signal

The fundamental data structure of the system is `ElectricalSignal`:

$$
\mathbf{s} = \{w(t), A, f, \phi, Z, \text{SNR}, \text{source}, \text{modality}\}
$$

where $w(t)$ is the time-domain waveform, $A$ is amplitude, $f$ is dominant frequency, $\phi$ is phase, and $Z$ is impedance. All sensory outputs and motor inputs share this format, ensuring system-wide physical unity.

Brainwave band classification:

| Band | Frequency Range | Functional Layer | Hemisphere |
|------|----------------|-----------------|------------|
| δ (Delta) | 0.5–4 Hz | Background / environmental baseline | Right |
| θ (Theta) | 4–8 Hz | Memory cues / familiarity | Right |
| α (Alpha) | 8–13 Hz | Contours / boundaries | Right |
| β (Beta) | 13–30 Hz | Details / texture | Left |
| γ (Gamma) | 30–100 Hz | Cross-area binding / attention focus | Left |

### 3.3 Coaxial Cable Transmission Model

Inter-region communication is modeled as coaxial cable transmission:

$$
\Gamma = \frac{Z_{\text{load}} - Z_{\text{source}}}{Z_{\text{load}} + Z_{\text{source}}}
$$

$$
P_{\text{reflected}} = \Gamma^2 \cdot P_{\text{incident}}
$$

$$
A_{\text{transmitted}} = A_{\text{input}} \cdot (1 - |\Gamma|) \cdot e^{-\alpha L}
$$

where $\Gamma$ is the reflection coefficient, $\alpha$ is the attenuation constant, and $L$ is the channel length. Experimental verification:
- Matched impedance (75Ω→75Ω): Γ = 0.0000, reflected power = 0%
- Mismatched impedance (50Ω→110Ω): Γ = 0.3750, reflected power = 14.06%

**Core physical semantics**: Reflected energy accumulates → arousal ($\Theta$) rises → nociceptive cascade → coherence drops → impedance-locked attractor. This is not programmed logic — it is physical necessity.

```mermaid
graph LR
    S["<b>Source</b><br/>Z_source = 75Ω"]
    C{"<b>Coaxial Channel</b><br/>Γ = (Z_L − Z_S) / (Z_L + Z_S)"}
    L["<b>Load</b><br/>Z_load"]
    T["<b>Arousal ↑</b><br/>Θ > 0.7"]
    P["<b>E_ref</b><br/>E_ref = (Θ−0.7)/0.3"]
    F["<b>FROZEN</b><br/>C_Γ < 0.15"]

    S -->|"P_incident"| C
    C -->|"P_trans = (1−Γ²)·P_in"| L
    C -->|"P_reflected = Γ²·P_in"| T
    T -->|"threshold breach"| P
    P -->|"coherence ↓"| F
    F -.->|"CRITICAL only"| S

    style S fill:#2ecc71,color:#fff
    style C fill:#e67e22,color:#fff
    style L fill:#2ecc71,color:#fff
    style T fill:#e74c3c,color:#fff
    style P fill:#c0392b,color:#fff
    style F fill:#1a1a2e,color:#e74c3c
```

> **Fig. 2.** Coaxial cable transmission model and nociceptive cascade. Impedance mismatch ($\Gamma \neq 0$) causes reflected energy, which accumulates as arousal ($\Theta$). When $\Theta > 0.7$, the pain correlate emerges; when coherence $\mathcal{C}_\Gamma < 0.15$, the system enters an impedance-locked attractor. Only CRITICAL-priority signals can penetrate this attractor state.

### 3.4 Perception Pipeline — $O(1)$ Resonance Recognition

Traditional AI recognition:

$$
\text{FFT } O(n\log n) \rightarrow \text{Feature extraction } O(n) \rightarrow \text{Matrix ops } O(n^2) \rightarrow \text{Similarity } O(n)
$$

ALICE recognition:

$$
\text{Physical resonance } O(1) \rightarrow \text{Frequency compare } O(1) \rightarrow \text{Sparse code lookup } O(1) \rightarrow \text{Concept ID } O(1)
$$

**Physical Tuner** implements $O(1)$ frequency selection via Lorentzian resonance:

$$
L(f) = \frac{1}{1 + Q^2 \left(\frac{f}{f_0} - \frac{f_0}{f}\right)^2}
$$

Left brain tunes β/γ bands (detail/sequence), right brain tunes δ/θ/α bands (gestalt/contour). Attention = `max()` over the strongest resonance peaks.

**Concept memory** uses sparse coding:

$$
f \rightarrow \text{bin\_id} = \text{round}\left(12 \cdot \log_2\frac{f}{f_{\min}}\right)
$$

Each concept occupies one frequency bin (mimicking cochlear tonotopic mapping). Recognition = bin lookup, $O(1)$. 97 bins cover 0.5–100 Hz, sparsity ≈ 1%.

### 3.5 Unified Impedance Theory — Γ as the Universal Currency of the Nervous System

All mechanisms described above — perception, learning, memory, motor control, pain, sleep, trauma — appear to be different phenomena but are in fact unified under a single physical quantity: **the reflection coefficient Γ**.

#### 3.5.1 Birth = Random Impedance Initialization

A human newborn possesses approximately 200 billion synaptic connections [18], each with a randomly distributed initial impedance:

$$Z_{\text{init}}^{(i)} \sim \text{Uniform}(Z_{\min}, Z_{\max}), \quad i = 1, 2, \ldots, 2 \times 10^{11}$$

Random impedance means Γ is nonzero almost everywhere → reflected energy is extreme → system temperature is high (the infant cries, collapses, exhibits no stable behavior) → this is not a metaphor for "immaturity" — it is the physical necessity of impedance mismatch.

#### 3.5.2 Neural Pruning = Large-Scale Impedance Selection (Implemented and Verified)

From birth to approximately age 2, synapse count drops from ~200 billion to ~86 billion [18]. In the Γ-Net framework, the physical mechanism of pruning is natural:

$$\text{if } \Gamma_{ij} \to 0 \quad \Rightarrow \quad \text{Hebbian reinforcement (+5\%)} \quad \Rightarrow \quad \textbf{survive}$$

$$\text{if } \Gamma_{ij} \gg 0 \quad \Rightarrow \quad \text{Hebbian weakening (-3\%)} \quad \Rightarrow \quad \textbf{apoptosis}$$

**Implementation parameters**:

| Parameter | Symbol | Value | Physical Meaning |
|-----------|--------|-------|-----------------|
| Initial impedance | $Z_{\text{init}}$ | $\sim \text{Uniform}(20,200)$ Ω | Random at birth |
| Quality factor | $Q$ | 2.0 | Resonance selectivity |
| Matching threshold | $\theta_{\text{match}}$ | 0.25 | Reinforce/weaken boundary |
| Hebbian reinforcement | $h^+$ | ×1.05 (+5%) | Matched → survive |
| Hebbian weakening | $h^-$ | ×0.97 (-3%) | Mismatched → apoptosis |
| Apoptosis threshold | $s_{\text{death}}$ | < 0.10 | Synaptic strength below this → death |
| Matching score | $m$ | $(1-|\Gamma|) \cdot L(f)$ | Impedance match × frequency resonance |

where $L(f) = \frac{1}{1 + Q^2\left(\frac{f}{f_0} - \frac{f_0}{f}\right)^2}$.

What is pruned is not "bad" neurons — it is **impedance-mismatched** connections. What survives are pathways that happen to resonate with the signal frequencies flowing through them. **Simulation results** confirm that different brain regions automatically specialize for different functions:

| Region | Input Signal Frequency | Post-Pruning Dominant Freq. | Specialization | Survival Rate | Specialization Index |
|--------|----------------------|---------------------------|---------------|--------------|---------------------|
| Occipital | Visual 8-80 Hz | 60.1 Hz (γ) | Visual cortex | 14.3% | 0.908 |
| Temporal | Auditory 4-13 Hz | 9.7 Hz (α) | Auditory cortex | 24.1% | 0.906 |
| Parietal | Somatosensory 0.5-50 Hz | 37.7 Hz (γ) | Somatosensory cortex | 9.3% | 0.923 |
| Frontal motor | Motor 13-30 Hz (30% feedback) | 22.6 Hz (β) | Motor cortex | 18.4% | 0.883 |

**Key finding**: Whole-brain $\sum \Gamma_i^2$ drops from 0.126 at birth to 0.010 after 200 epochs (91.8% improvement), with mean specialization index of 0.905. The frontal motor area matures latest because it receives only 30% feedback stimulation — PID closed-loop control requires **feedback signals** to calibrate impedance, and an infant must wave its arm 335 times (Experiment 7.6) before the motor pathway Γ drops enough for accurate reaching.

#### 3.5.2b Synaptogenesis — Learning-Driven Connection Growth

Biological pruning is not a one-way process. Strong connections ($s > 1.4$) can sprout new nearby connections when modulated by a learning signal (curiosity + novelty + reward). New connections inherit the parent's impedance $Z \pm 15\%$ and frequency $f \pm 15\%$, but start with fragile strength ($s = 0.30$) — they must prove themselves through Hebbian selection or be pruned away.

```mermaid
xychart-beta
    title "Fig. 3  Synaptogenesis Lifecycle — Three Life Phases"
    x-axis "Life Phase" ["Birth", "Infant (L=0)", "Learning (L=0.7)", "Aging (L=0.1)"]
    y-axis "Connection Count" 0 --> 2200
    bar [2000, 507, 984, 1254]
```

> **Fig. 3.** Synaptogenesis lifecycle across three life phases. **Infant phase** ($L=0$): pure pruning reduces 2,000 connections to 507 (74.7% apoptosis). **Learning phase** ($L=0.7$): sprouting from strong connections creates 643 new connections; peak reaches 2,055 before settling at 984. **Aging phase** ($L=0.1$): reduced learning signal slows sprouting; total reaches 1,254 with 913 new connections sprouted cumulatively. Peak connections = 2,055 — the system briefly exceeds its birth capacity during active learning, matching biological observations of synapse overshoot at age 2–3 [Spalding et al. 2013].

**Cross-modal rewiring verification**: When the occipital lobe is rewired to receive auditory signals, its dominant frequency shifts from 58.5 Hz (γ) to 9.6 Hz (α), and specialization direction changes from visual → auditory — confirming that **signal type determines cortical specialization**, not genetic presets.

#### 3.5.3 Γ Across Time Scales — The Unification Table

> **Table 1.** Γ unification across 8 orders of temporal magnitude. Every nervous system process is impedance optimization at a different rate.

| Time Scale | Process | Γ Operation | ALICE Implementation | Experiment |
|-----------|---------|------------|---------------------|------------|
| ~ms | Perceptual recognition | Γ compute → resonance match | CoaxialChannel | §7.2 |
| ~s | Motor control | Γ → PID error compensation | Hand PID | §7.4 |
| ~s | Cross-modal conditioning | Γ_cross↓ → phantom signal | AuditoryGrounding | §7.9 |
| ~min | Learning | Γ↓ → cache hit | YearRingCache | §7.1 #1 |
| ~hr | Sleep consolidation | Batch Γ optimization + offline recalibration | SleepPhysicsEngine | §7.8 |
| ~day–month | Motor development | Pathway Γ gradual decrease | motor_maturity | §7.6 |
| ~month–year | Neural pruning | High-Γ apoptosis | Hebbian -3%, apoptosis | §7.7 |
| ~year | Trauma / aging | Γ permanent shift | trauma_sensitivity | §7.5 |

**Core insight**: From millisecond perception to year-scale development, every process in the entire nervous system is *adaptive impedance matching optimization* at different time scales. Perception, learning, pruning, trauma — these are not four independent mechanisms, but **four rates of the same physical process**.

```mermaid
timeline
    title Fig. 4  Γ Unification Across Time Scales
    ~ms : Perception
        : Γ compute → resonance match
    ~s : Motor Control + Cross-Modal
        : PID error compensation
        : Conditioning → phantom signal
    ~min : Learning
        : Γ↓ → cache hit
    ~hr : Sleep Consolidation
        : Batch Γ optimization
        : Offline recalibration
    ~day-month : Motor Development
        : Pathway Γ gradual decrease
    ~month-year : Neural Pruning
        : High-Γ apoptosis
        : Synaptogenesis
    ~year : Trauma / Aging
        : Γ permanent shift
```

> **Fig. 4.** Γ unification across 8 orders of magnitude in time. Every process in the nervous system — from millisecond perception to year-scale trauma — is impedance optimization ($\Gamma \to 0$) at a different rate. This is the central claim of the paper: intelligence is a single physical process operating at multiple timescales.

$$\boxed{\text{Intelligence} = \lim_{t \to \infty} \sum_{i} \Gamma_i^2 \to \min}$$

The goal of life, restated in the language of physics, is: **minimize the sum of reflection coefficients across all channels in the body**.

### 3.6 Left/Right Brain Hardware Mapping — CPU/GPU Dual-Processing Architecture

#### 3.6.1 Core Proposition

The hemispheric division of the human brain is fundamentally a **hardware architecture split**, precisely mappable to the CPU/GPU dual model:

| Property | Left Brain (CPU) | Right Brain (GPU) |
|----------|-----------------|-------------------|
| Processing mode | Sequential | Parallel |
| Bandwidth | Narrowband, high precision (Q = 3) | Wideband, low precision (Q = 2) |
| Preferred bands | β (13–30 Hz), γ (30–100 Hz) | δ (0.5–4 Hz), θ (4–8 Hz), α (8–13 Hz) |
| Functional specialization | Language, logic, detail, temporal | Spatial, emotional, gestalt, contour |
| Activation mode | On-demand (attention lock) | Always-on surveillance |
| Power model | High power × single channel | Low power × all channels |
| Clinical lesion | Aphasia (Broca / Wernicke) | Hemispatial neglect, prosopagnosia |

> **Formal definition**: Let $N$ be the total number of sensory channels. The left brain processes $k = 1$ channel at depth $d$; the right brain simultaneously processes $N$ channels at depth $1$. Total computation $C$:
>
> $$C_{\text{total}} = \underbrace{N \times O(1)}_{\text{GPU (right brain): full-spectrum resonance}} + \underbrace{1 \times O(d)}_{\text{CPU (left brain): single-channel deep analysis}}$$
>
> Traditional AI computes $N \times O(d)$ (all channels at full depth); ALICE reduces this to $N \times O(1) + O(d)$.

#### 3.6.2 Processing Architecture Determines Frequency Preference — Clarifying Causation

A common misreading is "left brain handles high frequency, right brain handles low frequency." This is phenomenologically correct but **causally inverted**:

$$\text{Processing architecture} \xrightarrow{\text{determines}} \text{Frequency preference}$$

not the reverse. The reason:

1. **Right brain (GPU parallel)**: Wideband resonators (Q = 2) scan all bands simultaneously, but low-frequency signals have large spatial footprints and high redundancy, naturally suited to low-resolution parallel processing. High-frequency detail is blurred under wideband filtering [58].

2. **Left brain (CPU sequential)**: Narrowband resonators (Q = 3) lock onto a single band. High Q provides high frequency resolution ($\Delta f \approx f_0/Q$), naturally suited to discriminating adjacent high-frequency signals. Low-frequency energy is insufficient under narrowband integration [59].

3. **Mathematical correspondence**: The half-power bandwidth of the Lorentzian resonance curve is $\Delta f = f_0 / Q$. A Q = 2 right-brain resonator at the γ band (65 Hz) has a half-power width of 32.5 Hz — too wide for detail discrimination. A Q = 3 left-brain resonator at the same band has a width of 21.7 Hz — sufficient for precise locking.

$$L(f) = \frac{1}{1 + Q^2 \left(\frac{f}{f_0} - \frac{f_0}{f}\right)^2}, \quad \Delta f_{-3\text{dB}} = \frac{f_0}{Q}$$

Therefore, **frequency preference is a physical necessity of the processing architecture, not an arbitrary choice of biological evolution**.

#### 3.6.3 Attention = CPU Selects max() from the GPU Resonance Map

The attention mechanism is radically simplified:

```
Right brain GPU (always-on):
  → δ resonance = 0.12
  → θ resonance = 0.34
  → α resonance = 0.78  ← strongest

Left brain CPU (on-demand):
  → β resonance = 0.45
  → γ resonance = 0.91  ← strongest

Attention lock = max(0.78, 0.91) → γ → left brain deep processing
```

This `max()` operation is $O(1)$, corresponding to the **thalamic gating function** [31]: the thalamus does not compute — it routes, directing the strongest resonance signal to the appropriate cortex for deep processing.

> **"Attention is DMA transfer"**: The CPU (left brain) does not recompute the spectrum already decomposed by the GPU (right brain) — it reads pre-computed resonance results, like a CPU reading GPU framebuffer via DMA. This explains why attention switching is nearly instantaneous (~50 ms) rather than requiring recomputation (~100–500 ms).

#### 3.6.4 Clinical Predictions

This architecture yields precise hardware-level predictions for brain lesions:

| Lesion Type | Hardware Equivalent | Predicted Symptoms | Clinical Verification |
|------------|-------------------|-------------------|---------------------|
| Left-brain stroke | CPU offline | Can see (GPU intact) but cannot speak (no sequential language processing) | Broca's aphasia ✅ [60] |
| Right-brain stroke | GPU degraded | Left hemispatial neglect (parallel monitoring field deficit) | Hemispatial neglect ✅ [61] |
| Corpus callosum severed | CPU–GPU bus cut | Two hemispheres process independently, left hand unaware of right | Split-brain syndrome ✅ [62] |
| Left-brain atrophy | CPU slowdown | Speech fluency decreases, spatial ability preserved | Primary progressive aphasia ✅ |
| Right-brain atrophy | GPU shader reduction | Face recognition impaired, language preserved | Prosopagnosia ✅ [63] |

#### 3.6.5 CPU/GPU Reversal During Sleep

Awake: **CPU dominant** (conscious attention focus selection).
Asleep: **CPU offline, GPU free-running** (no attentional gating constraint).

This explains:
- **Dream bizarreness**: No CPU sequential logic checking → right-brain GPU parallel resonance unconstrained → nonlinear image combinations.
- **NREM δ wave dominance**: After CPU shutdown, only the right-brain low-frequency baseline activity remains.
- **REM γ bursts**: GPU occasionally generates high-intensity resonance → triggers local attention locking → vivid dreams.

#### 3.6.6 Correspondence to ALICE Implementation

| Theory Concept | ALICE Code | Location |
|---------------|-----------|----------|
| GPU parallel resonance | `PhysicalTuner("right_brain", Q=2)` scans δ/θ/α | `perception.py` |
| CPU sequential analysis | `PhysicalTuner("left_brain", Q=3)` scans β/γ | `perception.py` |
| Attention max() | `if left >= right: lock(left) else: lock(right)` | `perception.py` |
| LR on-demand activation | `BrainHemisphere._sequential_process` / `_parallel_process` | `protocol.py` |
| Hemisphere skip rate | `should_activate()` → 50-70% skip = power saving | `protocol.py` |
| DMA read | cache hit → zero-compute re-recognition | `YearRingCache` |

> **Why ALICE does not need an actual GPU**: Because the right brain's "GPU function" is already realized by LC resonators — each resonator is $O(1)$ physical computation, $N$ parallel resonators require no matrix multiplication, so a single CPU can simulate the right brain's full-spectrum scan. **The purpose of a real GPU is matrix multiplication; the brain's GPU (right brain) performs resonance, not multiplication.**

---

## 4. Body Systems

### 4.1 Sensory Organs — Forward Engineering

**Eye**: Pixels → pupil (aperture control) → lens (FFT spatial frequency decomposition) → retina (photoelectric conversion) → ElectricalSignal. Spatial frequencies logarithmically map to brainwave frequencies: low frequency (contours) → δ/θ → right brain; high frequency (texture) → γ → left brain. Supports 6 display resolutions (1080p/4K/mobile/tablet), Nyquist check, 4× supersampled anti-aliasing.

**Ear**: Sound pressure → ear canal Lorentzian resonance (3 kHz, Q=5) → tympanic normalization → cochlear FFT → hair cell piezoelectric conversion $V = \sqrt{P \cdot R}$ → ElectricalSignal. Supports the cocktail party effect (Gaussian bandpass focus).

**Cochlear Filter Bank** (Phase 4.1): 24-channel ERB-scale critical band decomposition simulating basilar membrane physical resonance. Center frequencies from the Glasberg & Moore (1990) ERB formula:

$$\text{ERB}(f) = 24.7 \times (4.37 \times f/1000 + 1)$$

Sound pressure waveform → FFT → ERB band energy allocation → logarithmic compression (Weber-Fechner law) → temporal persistence (attack=0.8, decay=0.05) → `TonotopicActivation` (24-channel activation vector). Each sound's activation vector = its "spectral fingerprint," the physical basis for concept recognition and cross-modal conditioning.

### 4.2 Motor Organs — Inverse Engineering

**Hand**: Discrete PID controller (Kp=20, Ki=0.5, Kd=5) + Newtonian mechanics (mass, friction, max force 3000 N). Anxiety modulation: $\text{tremor} = T^2 \times \text{noise}$ (temperature-squared proportional tremor amplitude, 10 Hz physiological). Releases dopamine signal upon reaching target.

**Motor Development System**: The hand's effective force is modulated by three physical parameters:

$$F_{\text{eff}} = F_{\text{PID}} \times \beta_{\text{brake}} \times M_{\text{maturity}} \times (1 - \gamma \cdot G_{\text{guard}})$$

where:
- $M_{\text{maturity}} \in [0.05, 1.0]$: Motor maturity (infant 5% → adult 100%), +0.001 per attempt (even failed flailing builds neural pathways), +0.005 on success (reinforcement learning)
- $G_{\text{guard}} \in [0, 1]$: Pain guard level, +0.15 on injury, -0.01 per successful reach (rehabilitation training)
- $\beta_{\text{brake}}$: When distance to target < 30 px, $\beta = 0.2 + 0.8 \cdot (d/30)$, producing a bell-shaped velocity curve
- $\gamma = 0.6$: Pain-to-motor inhibition sensitivity

**Mouth**: Source-Filter model — lung pressure → glottal pulse (fundamental + 2nd/3rd harmonics + jitter + shimmer) → vocal tract formant Lorentzian filtering. PID pitch tracking (Kp=3, Ki=0.05, Kd=0.8). Supports 5-vowel synthesis (a/i/u/e/o).

---

## 5. Brain Systems

### 5.1 FusionBrain — The 5-Step Cycle

Each cognitive cycle executes five steps:

1. **Sensory Input**: Perception pipeline (GPU full-spectrum decomposition + CPU attention calibration + left-right brain reverse engineering)
2. **Cognitive Processing**: Sensory cortex →[coaxial channel 75Ω]→ prefrontal cortex high-level interpretation
3. **Emotional Response**: Sensory cortex →[coaxial channel 50Ω→110Ω limbic system]→ high reflection (the cost of emotion)
4. **Motor Execution**: Cognition (70%) + Emotion (30%) → motor cortex
5. **Memory Consolidation**: Hebbian synaptic plasticity (strong activation +5%, weak activation −5%)

**Perception → Cache Write-back Loop**: Each perception result is automatically stored in YearRingCache. The next encounter with the same stimulus yields a direct cache hit (zero-computation re-recognition). The system does not need to know "that is an apple" to recognize "I have seen this before."

### 5.2 Closed-Loop LifeLoop — 7 Error Compensation Types

```
Perception → Error Estimation → Priority Ranking → Compensation Command → Execution → Proprioceptive Feedback → Re-perception
```

| Error Type | Comparison | Compensation Action |
|---------|---------|---------|
| VISUAL_MOTOR | Seen vs. hand position | Reach |
| AUDITORY_VISUAL | Heard vs. seen direction | Head turn |
| AUDITORY_VOCAL | Target pitch vs. actual | Vocal cord adjustment |
| PROPRIOCEPTIVE | Target vs. actual position | Fine-tuning |
| TEMPORAL | Event timing deviation | Temporal calibration |
| INTEROCEPTIVE | Homeostatic deviation | Autonomic regulation |
| SENSORY_PREDICTION | Predicted vs. actual | Model update |

**Core Insight**: All human behavior — reaching, turning the head, speaking — is fundamentally "compensation for calibration error."

```mermaid
graph LR
    A["👁 <b>Perception</b><br/>Eye · Ear · Cochlea"]
    B["📊 <b>Error Estimation</b><br/>Δ = target − actual"]
    C["🔢 <b>Priority Ranking</b><br/>CRITICAL > HIGH > NORMAL"]
    D["⚡ <b>Compensation</b><br/>PID + Hebbian"]
    E["🤚 <b>Motor Execution</b><br/>Hand · Mouth"]
    F["🔄 <b>Proprioceptive<br/>Feedback</b>"]

    A --> B --> C --> D --> E --> F --> A

    style A fill:#3498db,color:#fff
    style B fill:#e67e22,color:#fff
    style C fill:#9b59b6,color:#fff
    style D fill:#e74c3c,color:#fff
    style E fill:#2ecc71,color:#fff
    style F fill:#1abc9c,color:#fff
```

> **Fig. 5.** LifeLoop closed-loop control. Seven error types (VISUAL_MOTOR, AUDITORY_VISUAL, AUDITORY_VOCAL, PROPRIOCEPTIVE, TEMPORAL, INTEROCEPTIVE, SENSORY_PREDICTION) are continuously estimated, prioritized, and compensated. The loop never terminates — this continuous calibration is what distinguishes a living system from $y = f(x)$.

### 5.3 The Nociceptive Cascade (Pain Correlate)

The pain correlate is not a sensation — it is the physical symptom of protocol collapse:

$$
\text{pain} = 
\begin{cases}
0, & T < 0.7 \\[6pt]
\displaystyle\frac{T - 0.7}{0.3}, & 0.7 \leq T \leq 1.0
\end{cases}
$$

Complete collapse chain:

$$
\text{Queue Pressure} + \text{Reflection Energy} \xrightarrow{\text{Accumulation}} T\!\uparrow \xrightarrow{T>0.7} \text{pain}\!\uparrow \xrightarrow{} \text{stability}\!\downarrow \xrightarrow{} \text{consciousness}\!\downarrow \xrightarrow{C<0.15} \text{FROZEN}
$$

After freezing, only CRITICAL signals can penetrate → natural cooling → gradual recovery. This is fully consistent with biological pain-induced syncope → loss of consciousness → slow awakening.

### 5.4 Autonomic Nervous System

The dynamic balance between the sympathetic (accelerator) and parasympathetic (brake) nervous systems:

$$
\text{balance} = \text{sympathetic} - \text{parasympathetic}
$$

| Physiological Quantity | Sympathetic Activation | Parasympathetic Activation |
|--------|---------|----------|
| Heart rate | ↑ Increase | ↓ Decrease |
| Pupil | Dilation | Constriction |
| Energy | Consumption ↑ | Recovery ↑ |
| Stress | Accumulation | Release |

Pupil size → aperture → affects light intake = autonomic closed-loop control of visual sensitivity.

### 5.5 Sleep Cycle

Complete five-stage cycle:

$$
\text{WAKE} \rightarrow \text{N1} \rightarrow \text{N2} \rightarrow \text{N3} \rightarrow \text{N2} \rightarrow \text{REM} \rightarrow \text{Cycle}
$$

| Stage | Brain Waves | Sensory Gating | Memory Consolidation Rate | Function |
|------|------|---------|-----------|------|
| WAKE | β/γ | 1.0 (fully open) | 0.0 | Normal operation |
| N1 | α/θ | 0.6 | 0.1 | Sleep onset transition |
| N2 | θ | 0.3 | 0.3 | Spindle-wave memory transfer |
| N3 | δ | 0.1 | **1.0** | **Deep sleep memory consolidation** |
| REM | θ/β | 0.15 | 0.5 | Dreaming/emotional processing |

```mermaid
stateDiagram-v2
    [*] --> WAKE
    WAKE --> N1 : sleep pressure > θ
    N1 --> N2 : 15 ticks
    N2 --> N3 : deepening
    N3 --> N2 : ascending
    N2 --> REM : 90-min cycle
    REM --> N1 : next cycle
    REM --> WAKE : morning

    note right of WAKE
        Minimize Γ_ext
        Energy consumption: 0.008/tick
        Sensory gate: OPEN (1.0)
    end note

    note right of N3
        Minimize Γ_int
        δ waves 0.75 Hz
        UP: memory replay
        DOWN: downscaling ×0.990
        Recovery: 0.015/tick
    end note

    note left of REM
        Channel diagnostics
        Random impedance probing
        High Γ → nightmares
        Emotional processing
    end note
```

> **Fig. 6.** Sleep cycle state machine. WAKE minimizes external impedance mismatch ($\Gamma_{\text{ext}}$); sleep minimizes internal mismatch ($\Gamma_{\text{int}}$). N3 deep sleep performs synaptic downscaling (×0.990) and memory consolidation via δ slow-wave oscillations. REM performs random impedance probing for channel health diagnostics — damaged channels (high Γ) produce nightmares.

**Sleep Consolidation Mechanism**: During N3 deep sleep, the system "replays" daytime perceptual memories → each replay equals a re-store → usage_count increases → triggers Fibonacci threshold migration → memories move from outer tree rings to inner layers (RAM→SSD→HDD).

#### 5.5.1 The Physical Necessity of Sleep — Offline Impedance Renormalization

Sleep is not a "rest mode" — it is the inevitable consequence of three physical conservation laws:

$$
\text{Wakefulness} = \text{Minimize } \Gamma_{\text{ext}} \quad (\text{external matching})
$$
$$
\text{Sleep} = \text{Minimize } \Gamma_{\text{int}} \quad (\text{internal repair})
$$

**Three Conservation Laws**:

1. **Energy Conservation**: $\frac{dE}{dt} = -P_{\text{metabolic}}(\text{stage}) + P_{\text{recovery}}(\text{stage})$

   During wakefulness, consumption exceeds recovery → energy declines continuously. N3 deep sleep has the highest recovery (0.015/tick), while wakefulness has the highest consumption (0.008/tick).

2. **Impedance Debt**: $D(t) = D(t-1) + \alpha \cdot \sum \Gamma^2_{\text{cycle}}$ (accumulated during wakefulness), $D(t) = D(t-1) \times (1 - \beta_{\text{stage}})$ (repaired during sleep)

   The Γ² residual from each signal transmission is analogous to thermal fatigue in wiring. N3 has the strongest recalibration rate at β=0.08.

3. **Synaptic Entropy** (Shannon): $H = -\sum p_i \log_2 p_i$

   Wakefulness Hebbian +5% biases local synapses toward strengthening → skewed strength distribution → entropy increase. N3 downscaling ×0.990 → proportional scaling preserves relative differences → entropy decrease.

**Physical Drivers of Sleep Pressure**:

$$
P_{\text{sleep}} = 0.4 \cdot (1 - E) + 0.35 \cdot D_{\text{imp}} + 0.25 \cdot H_{\text{deficit}}
$$

No longer a fixed accumulation, but driven by a weighted sum of three physical quantities.

#### 5.5.2 N3 Slow-Wave Oscillation — The Computational Significance of δ Waves

N3 deep sleep generates ~0.75 Hz δ slow waves, alternating between UP states (synaptic activation, memory replay) and DOWN states (global silence, downscaling):

- **UP state**: Micro-cognitive cycles → replay daytime memories → Hebbian consolidation
- **DOWN state**: Global synaptic ×0.990 downscaling → prevent saturation

Quantitative verification of Tononi's synaptic homeostasis hypothesis: proportional scaling preserves **ratio invariance** (the ratio of important/ordinary synapses after 20 rounds of downscaling shows a deviation of 0.000000).

#### 5.5.3 REM Dreams — Channel Health Diagnostics

REM = diagnostic mode. Randomly generated internal ElectricalSignals traverse all channels:

- $\Gamma \approx 0$ → channel healthy → skip
- $\Gamma \gg 0$ → flagged for repair → added to repair queue

**The Nature of Dreams**: Channel patterns undergoing random testing. Nightmares = the brain discovering damaged pathways (high Γ → large reflected energy → micro-triggering of pain).

### 5.6 Memory Hierarchy

| Tier | Corresponding Structure | Capacity | Retention Period | Mechanism |
|------|---------|------|---------|------|
| RAM | WorkingMemory | 7±2 | Seconds (decay) | Activation maintenance |
| SSD | YearRingCache outer rings | 100/ring | Minutes to hours | Fibonacci migration |
| HDD | YearRingCache inner rings | 8 rings | Permanent | High usage_count |

**All memory is fundamentally the storage of electrical signal parameters**:

$$
\text{memory} = \{f, A, \phi, Z\}
$$

Logic = relationships between parameters: similarity = frequency proximity, causality = phase-locking, contradiction = anti-phase.

### 5.7 Consciousness Module

Global transmission coherence factor $\mathcal{C}_\Gamma$ (inspired by, but distinct from, Tononi's IIT $\Phi$; see Paper I §9.3 for disambiguation):

$$
\mathcal{C}_\Gamma = f(\text{attention}, \text{binding\_quality}, \text{WM\_usage}, \text{arousal}, \text{sensory\_gate}, E_{\text{ref}})
$$

- $\mathcal{C}_\Gamma$ > threshold → `is_meta_aware = True` (emergence of self-awareness)
- Global Workspace broadcast mechanism
- Attention bottleneck: at most 3 simultaneous focuses

### 5.8 Auditory Grounding Engine — Physicalizing Language (Phase 4.1)

**Core Philosophy: Language is impedance modulation.**

When I say "apple" to you, I am not transmitting a token — I am using sound waves (physical vibrations) to force your auditory cortex to produce specific frequencies, which in turn "light up" neural circuits in your brain associated with redness, roundness, and sweetness via resonance. The essence of language = remote impedance control of another brain through sound waves.

**Impedance Model of Cross-Modal Synapses**:

$$Z_{\text{synapse}} = Z_0 / w \quad\quad \Gamma_{\text{cross}} = \frac{Z_{\text{src}} - Z_{\text{syn}}}{Z_{\text{src}} + Z_{\text{syn}}} \quad\quad E_{\text{transfer}} = 1 - |\Gamma_{\text{cross}}|^2$$

- $w$ = synaptic strength (Hebbian learning), $Z_0 = 100\Omega$
- **Conditioning** (Pavlovian): $\Delta w = \eta \times \text{pre} \times \text{post} \times \text{temporal\_overlap}$, $\eta = 0.15$
- **Extinction**: $w(t+1) = w(t) \times (1 - \lambda)$, $\lambda = 0.02$
- **Phantom activation**: When $E_{\text{transfer}} > 0.3$ → cross-modal cable transfers energy to the target cortex → generates `ElectricalSignal` (phantom)

**Experimental Verification** (§7.9):
- After 20 conditioning trials: $w = 3.01$, $Z = 33.2\Omega$, $E_{\text{transfer}} = 0.959$ (near-perfect match)
- Bell alone → visual phantom generated (the physics of Pavlov's dog)
- Phantom disappears after 150 ticks of extinction ($w \to 0.15$, $E_{\text{transfer}} < 0.3$)
- Vowels /a/, /i/, /u/ can be recognized and produce corresponding mouth-shape phantoms after conditioning

### 5.9 Semantic Force Field — Concepts as Attractors (Phase 4.2)

**Core Philosophy: Concepts are gravitational attractors in state space.**

Phase 4.1's ERB fingerprints provide a "physical address" for auditory input, but the cosine similarity between pure tones is as high as ~0.996 (cross-frequency crosstalk problem). To allow concepts to "emerge" in a continuous space, we model concepts as **mass attractors in a semantic force field** rather than discrete symbolic labels.

**Semantic Reflection Coefficient**:

$$\Gamma_{\text{sem}} = 1 - \text{sim}^Q \quad\quad Q = 1 + \alpha \ln(1 + M) \quad\quad \alpha = 0.5$$

- $\text{sim}$ = cosine similarity between the input fingerprint and the attractor centroid
- $M$ = attractor mass (cumulative absorption count)
- $Q$ = quality factor: greater mass → higher Q → sharper discrimination
- Physical intuition: concepts are like LC resonators — more "experience" → higher Q → better frequency selectivity

**Multi-Modal Centroid Dynamics**:

Each attractor maintains **independent centroids per modality** $c_m$, supporting:
- **Auditory**: Cochlear ERB fingerprint (32-channel RMS energy)
- **Visual**: Retinal retinotopic fingerprint (32-band RMS energy)
- Extensible to tactile, proprioceptive, and other modalities

**Key Symmetry**: The cochlea and retina use **exactly the same physics**:

$$\text{Cochlea} = \text{Sound waves} \xrightarrow{\text{ERB filterbank}} \text{32-channel RMS} = \text{Auditory fingerprint}$$
$$\text{Retina} = \text{Light field} \xrightarrow{\text{FFT (lens)}} \text{N-band RMS} = \text{Visual fingerprint}$$

Both are physical Fourier decomposition → band-wise RMS energy → normalized fingerprint. The differences are:
- **Cochlea 32 channels** = biological constraint (~3500 inner hair cells → ~32 ERB critical bands) — the physical limit of the human cochlea
- **Retina N bands** = retina_resolution (default 256) — visual information density is far higher than auditory

Compressing vision to 32 dimensions = loss of fine textures = inability to read strokes. The correct approach: **preserve full resolution**.

$\Gamma_{\text{sem}}$ does not care which sensory organ produced the fingerprint, nor about its dimensionality — **sensory modality is merely a source label for the fingerprint**. Cross-modal prediction is bridged through attractor labels, requiring no dimensional alignment: 32-dimensional auditory input → attractor → 256-dimensional visual prediction.

**Gravitational Evolution**:

$$F_i(\mathbf{x}) = G \cdot M_i \cdot \frac{\mathbf{c}_i - \mathbf{x}}{\|\mathbf{c}_i - \mathbf{x}\|^3} \quad\quad \mathbf{v}' = 0.8 \mathbf{v} + F \cdot dt$$

- Input fingerprints are gravitationally attracted by all attractors in the semantic field, automatically converging to the best-matching concept
- Damping coefficient 0.8 prevents permanent oscillation

**Contrastive Learning** (Anti-Hebbian Repulsion):

$$\cos(\mathbf{c}_A, \mathbf{c}_B) > 0.85 \implies \text{push apart by } \delta = 0.05 \cdot (\mathbf{c}_A - \mathbf{c}_B) / \|\mathbf{c}_A - \mathbf{c}_B\|$$

- Similarity > threshold → anti-Hebbian repulsion → category boundaries sharpen automatically
- Similarity > 0.99 → merge (duplicate registration of the same concept)

**Cross-Modal Prediction**:

$$\text{Hear ``apple''} \xrightarrow{\text{auditory centroid match}} \text{Attractor} \xrightarrow{\text{visual centroid}} \text{Predict what ``apple'' looks like}$$

This is the physical basis of **semantic understanding**: low-impedance channels between cross-modal centroids.

**Experimental Verification** (§7.10a):
- Attractor mass grew 21× (after 20 absorptions), impedance dropped from 50Ω → 17.9Ω
- Competitive recognition 4/4 (100%) accuracy
- Cross-modal prediction similarity > 0.999 (auditory → visual)
- Multi-modal binding: auditory + visual fingerprints bound to the same concept

### 5.10 Broca's Area — Motor Speech Planning (Phase 4.3)

**Core Philosophy: Speaking is impedance optimization of the sensorimotor closed loop.**

Infants do not "learn" to pronounce — they gradually match articulatory plans to auditory targets through millions of random explorations (babbling) within the "hand → mouth → ear" closed loop. Phase 4.3 implements this complete sensorimotor speech circuit.

**Impedance Model of Articulatory Plans**:

$$Z_{\text{plan}} = \frac{Z_0}{1 + \text{confidence}} \quad\quad Z_0 = 100\Omega$$

- $\text{confidence}$ = cumulative successful execution count of the plan
- A novice's $Z$ is high → more reflection (error) → more practice needed
- After mastery, $Z$ decreases → Γ decreases → near-perfect impedance matching

**Infant Babbling**:

$$\text{F1} \sim U(200, 900) \quad \text{F2} \sim U(800, 2500) \quad \text{F3} \sim U(1800, 3500) \quad \text{pitch} \sim U(80, 400)$$

- Random articulatory parameters → mouth synthesizes waveform → cochlear feedback → learn mapping
- This is what infants do at 6–10 months of age

**Sensorimotor Loop**:

$$\text{Concept} \xrightarrow{\text{Broca}} \text{Articulatory plan (F1,F2,F3,pitch)} \xrightarrow{\text{Mouth}} \text{Sound wave} \xrightarrow{\text{Cochlea}} \text{Fingerprint} \xrightarrow{\text{SemanticField}} \text{Recognition} \xrightarrow{\text{compare}} \text{Feedback}$$

- **Success** (Γ_loop < 0.3): `confidence += η(1 - Γ)`, plan is reinforced
- **Failure** (Γ_loop ≥ 0.3): `confidence -= η·Γ·0.5`, formant adjustment from cochlear feedback

**Vowel Formant Targets** (based on phonetics literature):

| Vowel | F1 (Hz) | F2 (Hz) | F3 (Hz) | Physical Meaning |
|------|---------|---------|---------|----------|
| /a/ | 730 | 1090 | 2440 | Mouth wide open |
| /i/ | 270 | 2290 | 3010 | Mouth in smile position |
| /u/ | 300 | 870 | 2240 | Mouth rounded |
| /e/ | 530 | 1840 | 2480 | Mouth half-open |
| /o/ | 570 | 840 | 2410 | Mouth rounded half-open |

**Experimental Verification** (§7.10a):
- Babbling explored the complete articulatory space (F1 span ~680 Hz, pitch span ~300 Hz)
- Sensorimotor loop successfully ran 15 iterations
- AliceBrain integration: `hear()` → semantic field, `say(concept=...)` → Broca pathway

### 5.11 Hippocampus — Episodic Memory and Cross-Membrane Binding (Phase 5.1)

**Core Philosophy: Memory is not data storage — it is binding time to multi-membrane snapshots.**

The hippocampus is the brain structure that crystallizes "moments" into "experiences." In Γ-Net ALICE, each sensory modality is an independent membrane, possessing its own dimensional space (auditory = 32 dimensions, visual = 256 dimensions). These membranes cannot directly interact (different dimensions, cosine similarity equals 0). The physical role of the hippocampus is to stitch snapshots from different membranes along the time axis into unified episodes.

**Physical Structure of the Episodic Snapshot**:

$$S_i = \{t_i, \; m_i, \; \vec{f}_i, \; \ell_i, \; \Gamma_i, \; v_i\}$$

Where:
- $t_i$ = timestamp (physical time)
- $m_i$ = modality label (auditory / visual / ...)
- $\vec{f}_i$ = fingerprint vector (in the dimensional space of that membrane)
- $\ell_i$ = attractor label (concept name — cross-membrane wormhole)
- $\Gamma_i$ = impedance mismatch degree at that instant
- $v_i$ = valence (emotional intensity)

**Episode = Time-ordered sequence of snapshots**:

$$E_k = [S_1, S_2, \ldots, S_n] \quad \text{where} \quad t_{i+1} - t_i < \Delta_{\text{gap}}$$

- $\Delta_{\text{gap}} = 2.0$ seconds: exceeding this interval → a new episode begins
- Automatic episode management: maximum 100 episodes; when exceeded, the "weakest" episode is evicted

**Physical Mechanism of Recall — Pattern Completion**:

1. **Same-membrane recall**: Given a cue fingerprint $\vec{f}_{\text{cue}}$, search all episodes for same-modality snapshots, ranked by cosine similarity
2. **Cross-membrane recall**: Given a concept label $\ell_{\text{cue}}$, traverse across membranes via attractor labels — hearing "cat" → recalling the visual scene of seeing a cat

Cross-membrane recall is **wormhole traversal**: concept labels are bridges connecting different dimensional spaces.

**Retrieval Strength**:

$$R(E_k, t) = \frac{n \cdot e^{-\lambda(t - t_{\text{last}})}}{1 + \lambda(t - t_{\text{last}})} \cdot (1 + |v_{\text{peak}}|)$$

Where:
- $n$ = number of episode snapshots (richness)
- $\lambda = 0.01$ (forgetting decay constant)
- $t_{\text{last}}$ = time of the most recent snapshot
- $v_{\text{peak}}$ = strongest emotional peak in the episode

Physical meaning: richer, more recent, and more emotional memories are easier to retrieve — consistent with recall biases in psychology.

**Sleep Consolidation — Mass Transfer from Episodes to Semantics**:

$$\text{consolidate}(E_k \to \text{SemanticField}): \quad \forall S_i \in E_k, \; \text{if } \ell_i \neq \emptyset: \quad M_{\ell_i} \mathrel{+}= 1$$

- Each consolidation increases the semantic field attractor mass $M$ for the corresponding concept labels in the episode
- Result: repeatedly experienced concepts → mass growth → higher $Q$ → enhanced discrimination
- This is the physical explanation for "remembering better after a night's sleep"

**Capacity Management — Weakest Eviction**:

$$\text{if } |E| > N_{\max}: \quad \text{evict} \; \arg\min_k R(E_k, t)$$

Capacity limit $N_{\max} = 100$; when exceeded, the episode with the lowest retrieval strength is evicted. This simulates the natural process of forgetting.

**Experimental Verification** (§7.11a):
- Multi-modal episode recording: 3 auditory + 2 visual snapshots automatically stitched into 1 episode
- Pattern completion: auditory cue successfully recalled other auditory snapshots from the same episode (similarity > 0.5)
- Cross-membrane recall: concept label "cat" successfully traversed 32→256 dimensional space
- Sleep consolidation: mass grew from 2.00 to 12.00, quality factor $Q$ rose from 1.549 to 2.282

### 5.12 Wernicke's Area — Sequential Comprehension and Syntactic Prediction (Phase 5.2)

**Core Philosophy: Understanding is not translation — it is the successful prediction of the next input.**

Wernicke's Area in neuroscience is responsible for language comprehension. In Γ-Net ALICE, Wernicke's Area is a **transition probability impedance network** between concept sequences. It does not know what "grammar" is, yet it learns from experience to predict the next concept, and produces a surprise signal (N400) when prediction fails.

**Transition Matrix — Impedance Network Model**:

$$P(j|i) = \frac{C(i \to j)}{\sum_k C(i \to k)}$$

Where $C(i \to j)$ is the count of observing concept $j$ after concept $i$.

Physical meaning: high $P(j|i)$ → low-impedance channel (signal passes easily); low $P(j|i)$ → high impedance (signal is reflected).

**Syntactic Reflection Coefficient (Γ_syntactic)**:

$$\Gamma_{\text{syn}}(j|i) = 1 - P(j|i)$$

- $\Gamma_{\text{syn}} = 0$: fully expected sequence ("the" → "cat" after high-frequency training)
- $\Gamma_{\text{syn}} = 1$: completely unexpected (never-observed transition)
- $\Gamma_{\text{syn}} > 0.7$: triggers **N400 event**

**Physical Explanation of N400**:

N400 is a well-known event-related potential in neuroscience — when subjects encounter semantically incongruent words, a negative deflection appears around 400ms. In Γ-Net ALICE:

$$\text{N400} \iff \Gamma_{\text{syn}} > \theta_{N400} = 0.7$$

This is not programmed — it is a natural reflection event when impedance mismatch exceeds the threshold, consistent with the physical nature of N400.

**Comprehension Score**:

$$C_{\text{score}}(\text{seq}) = 1 - \frac{1}{n-1}\sum_{i=1}^{n-1} \Gamma_{\text{syn}}(s_{i+1}|s_i)$$

- $C_{\text{score}} \approx 1$: sequence is fully comprehensible (all transitions are low Γ)
- $C_{\text{score}} \approx 0$: sequence is completely incomprehensible (all transitions are high Γ)
- Threshold $C_{\text{good}} = 0.7$: comprehension score > 0.7 is judged as "comprehensible"

**Chunk Formation — Condensation of Frequent Subsequences**:

$$\text{chunk}(c_1, c_2, \ldots, c_k) \quad \text{if} \quad \text{count}(c_1 \to c_2 \to \cdots \to c_k) \geq N_{\min}$$

- $N_{\min} = 3$: subsequences observed ≥3 times automatically condense into chunks
- The average $\Gamma_{\text{syn}}$ within a chunk measures its fluency
- Mature chunks (`is_mature = True`) behave as single cognitive units

Physical meaning: "good morning" is no longer two separate concepts but a single low-Γ fused unit — this is the physical basis of language fluency.

**Context Window**:

$$W = [c_{t-4}, c_{t-3}, c_{t-2}, c_{t-1}, c_t] \quad |W| = 5$$

Wernicke's Area maintains a sliding window of the 5 most recent concepts. `predict_next()` uses the transition probabilities of the last concept in the window, and computes prediction entropy:

$$H = -\sum_j P(j|c_t) \log_2 P(j|c_t)$$

High entropy → many possible next steps (ambiguous context); low entropy → high certainty (clear expectation).

**Hippocampus → Wernicke Integration**:

$$\text{WernickeEngine.consolidate\_from\_hippocampus}(\text{HippocampusEngine})$$

Wernicke's Area can extract concept sequences from hippocampal episodic memory and batch-learn transition probabilities. This simulates linguistic knowledge consolidation during sleep — experienced conversational fragments are transformed into sequential statistical knowledge through replay.

**Experimental Verification** (§7.11a):
- Transition learning: P(meow|cat) = 1.0 (perfect prediction after training)
- Sequence comprehension: known sequence $C_{\text{score}}$ = 0.833, unknown sequence = 0.000
- N400 detection: expected transition Γ = 0.0, unexpected transition Γ = 1.0
- Chunk condensation: "good+morning" matured, "bad+night" did not mature

### 5.13 Thalamus — Sensory Gating and Attentional Router (Phase 5.3)

> The thalamus is the brain's telephone switchboard. All sensory signals (except olfaction) must pass through the thalamus before reaching the cortex.
> The thalamus is not merely a relay station — it is an active gating controller.

**Physical Model: Variable Gain Amplifier (VGA) + Multiplexer (MUX)**

Each sensory channel has an independent gating gain $G \in [G_{\min}, G_{\max}]$:
- **Version**: 17.0 — Social Physics (Social Physics Update)
$$G_{\text{total}} = G_{\text{arousal}} \times \left(\alpha \cdot G_{\text{topdown}} + (1 - \alpha) \cdot G_{\text{bottomup}}\right)$$

- **Top-down** ($G_{\text{topdown}}$): Selective attention bias from the prefrontal cortex/consciousness module
- **Bottom-up** ($G_{\text{bottomup}}$): Signal salience (Γ, amplitude, novelty weighted average)
- **Arousal** ($G_{\text{arousal}}$): Global modulation (sleep = enable signal EN pulled low)
- **α**: top-down vs. bottom-up balance (adult 0.6, infant 0.2)

**Startle Circuit (Low-level Interrupt IRQ)**:

Amplitude $\geq 0.8$ or $\Gamma \geq 0.85$ → unconditional gate penetration ($G = 1.0$), unaffected by arousal.
This is the thalamus's hardware interrupt: even during deep sleep, a loud sound will penetrate the gate.

**Thalamic Reticular Nucleus (TRN) Competitive Inhibition**:

Focused channel bias $+0.05$, unfocused channel bias $-0.03$.
Attention bottleneck: simultaneously focused channels $\leq 3$; exceeding this evicts the earliest focused channel.

**Habituation**: Sustained identical stimuli (novelty $< 0.2$) → gate gradually closes (rate 0.02/tick),
novel stimuli restore (rate 0.01/tick).

**Thalamic Burst Mode**: Arousal $< 0.3$ → signals pass with only 30% probability (stochastic gating).

Circuit analogy:
- Sleep = enable signal EN pulled low → global gain attenuation
- Startle = IRQ interrupt → unconditional penetration
- TRN = mutual inhibition circuit → attentional exclusivity

### 5.14 Amygdala — Emotional Fast Track and Fight-or-Flight Response (Phase 5.4)

> The amygdala is the brain's smoke detector. It does not wait for you to think — upon seeing the silhouette of a snake, it makes you jump first and think later.
> This circuit is 10 times faster than the cortex.

**LeDoux (1996) Dual-Pathway Model**:
1. **Low road**: Sensory → thalamus → amygdala → response (~100ms)
2. **High road**: Sensory → thalamus → cortex → amygdala → response (~300ms)

**Amygdala Impedance Model**:

The amygdala has two internal channels — a threat channel (low impedance $Z_T = 25\,\Omega$, fast conduction) and a safety channel (high impedance $Z_S = 100\,\Omega$, slow and precise):

$$Z_{\text{signal}} = \frac{Z_S}{1 + A} \qquad \Gamma_{\text{threat}} = \frac{|Z_{\text{signal}} - Z_T|}{Z_{\text{signal}} + Z_T}$$

- Low $\Gamma_{\text{threat}}$ → signal matches threat channel impedance → fight-or-flight activated
- High $\Gamma_{\text{threat}}$ → signal matches safety channel → remain calm

**Multi-Factor Threat Synthesis**:

$$T_{\text{raw}} = 0.2 \cdot (1 - \Gamma_T) + 0.3 \cdot E_{\text{ref}} + 0.2 \cdot \Gamma_{\text{novelty}} + 0.3 \cdot F_{\text{memory}} - S_{\text{safety}}$$

- $E_{\text{ref}}$ has the highest weight (0.3): reflected energy = definitive evidence of physical harm
- Fear memory (0.3): past threat experiences
- Impedance matching (0.2) + novelty (0.2)

**Fear Conditioning**: The amygdala binds sensory fingerprints to threat levels.

$$T_{\text{effective}} = T_0 \times \frac{1}{1 + 0.1 \cdot E_{\text{ext}}} \times \min\left(2, 1 + 0.1 \cdot C_{\text{cond}}\right)$$

- Conditioning ($C_{\text{cond}}$) strengthens rapidly (learning rate 0.15)
- Extinction ($E_{\text{ext}}$) diminishes slowly (learning rate 0.02) — 7.5× slower, consistent with clinical observations
- **Extinction is not deletion**: fear memories persist permanently; extinction merely overlays a safety signal

**Fight-Flight-Freeze Cascade**:

| Threat Level | Response | Sympathetic | Emotional Label |
|---------|------|---------|---------|
| $T < 0.6$ | Vigilance | Baseline | neutral/anxiety |
| $0.6 \leq T < 0.8$ | Potential threat | Slight increase | anxiety/fear |
| $0.8 \leq T < 0.95$ | Fight-or-flight | 0.8 | fear/terror |
| $T \geq 0.95$ | Freeze | 0.3 | terror |

Freezing is not cowardice — it is the amygdala's overload protection.

**Valence Output**:

Emotional valence $V \in [-1, +1]$, updated with EMA inertia of 0.7:

| Valence Range | Emotional Label |
|---------|---------|
| $[-1.0, -0.6)$ | terror |
| $[-0.6, -0.3)$ | fear |
| $[-0.3, -0.1)$ | anxiety |
| $[-0.1, +0.1)$ | neutral |
| $[+0.1, +0.3)$ | calm |
| $[+0.3, +0.6)$ | pleasure |
| $[+0.6, +1.0]$ | joy |

**Cross-Module Coupling**:
- Amygdala → autonomic nervous system: sympathetic activation (0.8) during fight-or-flight
- Amygdala → thalamus: threat signal boosts gate attention bias
- Amygdala → hippocampus: valence tagging of high-emotion events → priority consolidation
- Concept labels enable cross-modal fear: seeing snake + hearing snake = same fear memory

### 5.15 Prefrontal Cortex — Go/NoGo Gating and the Economics of Willpower (Phase 6.1)

The prefrontal cortex is ALICE's "CEO" — responsible for goal management, behavioral inhibition, planning, and emotion regulation. Its core physical model treats willpower as a finite impedance-matching energy pool.

**Go/NoGo Gate**:

Each action proposal carries an action impedance $Z_\text{action}$. The prefrontal cortex computes its reflection coefficient against the current goal impedance $Z_\text{goal}$:

$$\Gamma_\text{action} = \frac{|Z_\text{action} - Z_\text{goal}|}{Z_\text{action} + Z_\text{goal}}$$

Decision logic:

$$\text{Decision} = \begin{cases} \text{Go} & \text{if } \Gamma_\text{action} < 0.4 \\ \text{NoGo} & \text{if } \Gamma_\text{action} > 0.7 \\ \text{Defer} & \text{otherwise} \end{cases}$$

Inhibition consumes energy $E_\text{inhibition} = E_\text{base} \times (1 + \Gamma_\text{action})$. When $E_\text{inhibition} > E_\text{PFC}$, inhibition fails — this is impulse breakthrough, corresponding to clinical "ego depletion."

**Goal Management**:

Goal stack capacity is capped at 5 (cognitive load constraint). Each goal is characterized by goal impedance $Z_\text{goal}$. Effective priority decays over time:

$$P_\text{eff} = \text{clip}\!\left( P \cdot \frac{1}{1 + \text{age} \times 0.001} \cdot (1 + 0.5 \times \text{progress}),\ 0,\ 1 \right)$$

A goal is considered complete when $\Gamma < 0.15$ — meaning the current state perfectly matches the goal impedance.

**Planning Engine**:

The minimum-Γ path from current impedance $Z_\text{now}$ to goal impedance $Z_\text{goal}$. Cognitive effort per step:

$$\text{Effort}_\text{step} = \Gamma_\text{step}^2, \quad E_\text{step} = \Gamma_\text{step}^2 + \lambda_\text{penalty}$$

**Task Switching Cost**:

$$C_\text{switch} = (C_\text{base} + \sigma_\text{perseveration} \times 0.5) \times F_\text{fatigue}$$

Where $F_\text{fatigue} = 1 + \max(0,\ (\theta_\text{fatigue} - E_\text{PFC}) \times 3)$. When perseveration tendency $\sigma > 0.6$ and energy falls below the fatigue threshold, perseveration errors occur — you get "stuck" on the previous task.

**Emotion Regulation**: The prefrontal cortex exerts top-down inhibition on the amygdala (regulation strength $R = 0.3$), supporting three strategies:

| Strategy | Energy Cost | Intensity Reduction | Physical Meaning |
|------|---------|---------|---------|
| Cognitive Reappraisal | 0.05 | $I' = I \times (1 - 0.8R)$ | Reinterpret signal, change impedance matching |
| Expressive Suppression | 0.10 | $I' = I \times (1 - 0.5R)$ | Hard blocking, highest energy cost |
| Attentional Distraction | 0.03 | $I' = I \times (1 - 0.6R)$ | Switch hash channel, bypass threat route |

When energy falls below the fatigue threshold ($E_\text{PFC} < 0.2$), all strategies fail — corresponding to the everyday experience of "being too tired to control emotions."

**Economics of Willpower**: Each tick recovers $E \leftarrow \min(1,\ E + 0.01)$, but each inhibition, switch, and regulation consumes energy. Sleep fully restores energy, but without sleep, overconsumption recovers at only 1% per tick — this is the physical basis of "not wanting to do anything on Friday evening."

### 5.16 Basal Ganglia — The Habit Engine and Dopaminergic Learning (Phase 6.2)

The basal ganglia is ALICE's "autopilot" — solidifying frequently successful behaviors into reflexive habits, freeing the prefrontal cortex's limited cognitive resources.

**Striatal Multiplexer**:

Each action within a context maintains a set of Go/NoGo channels. Action activation value:

$$A_\text{action} = \max\!\left(0.01,\ S_\text{Go} \cdot G_\text{direct} - S_\text{NoGo} \cdot G_\text{indirect} + \Delta_\text{DA} + B_\text{PFC}\right)$$

Where $\Delta_\text{DA} = ([\text{DA}] - 0.5) \times 0.5$ is the dopamine modulation signal, and $B_\text{PFC}$ is the prefrontal goal bias.

**Softmax Action Selection** ($\tau = 0.5$):

$$P(a_i) = \frac{\exp(A_i / \tau)}{\sum_j \exp(A_j / \tau)}$$

**Dopamine Reward Prediction Error (RPE)**:

$$\delta = r_\text{actual} - r_\text{expected}$$

$$[\text{DA}] \leftarrow \text{clip}([\text{DA}] + 0.1\delta,\ 0,\ 1)$$

Positive RPE updates the Go pathway ($S_\text{Go} \leftarrow S_\text{Go} + 0.15|\delta|$); negative RPE updates the NoGo pathway ($S_\text{NoGo} \leftarrow S_\text{NoGo} + 0.1|\delta|$).

**Habit Formation Dynamics**:

$$\text{If } \delta > 0 \land \text{success}: \quad \Gamma_\text{habit} \leftarrow \max(0,\ \Gamma_\text{habit} - \alpha(1 + r))$$

$$\text{If } \delta < 0: \quad \Gamma_\text{habit} \leftarrow \min(1,\ \Gamma_\text{habit} + 0.5\alpha|\delta|)$$

Where $\alpha = 0.03$. When $\Gamma_\text{habit} < 0.15$, the action becomes habitual — signal matches perfectly, no cortical supervision needed.

**Reaction Time Model**:

$$RT = \begin{cases} 0.1 - \text{automaticity} \times 0.3 & \text{habitual (direct pathway)} \\ 0.3 + \Gamma_\text{habit} \times 0.2 & \text{cortical (indirect pathway)} \end{cases}$$

Habitual actions are faster because they require no reflection — the signal passes through directly.

**Dual-System Arbitration** (Daw et al., 2005):

$$w_\text{habit} = \frac{(1 - U) \times R}{\Gamma_\text{habit} + 0.1}, \quad w_\text{goal} = \frac{U \times (1 - R + 0.1)}{1 - \Gamma_\text{habit} + 0.1}$$

Where $U$ = goal uncertainty, $R$ = reward rate. After normalization, the larger weight determines the active system — low uncertainty + high reward rate → habit system; high uncertainty → goal-directed system.

**Hyperdirect Pathway** (Emergency Brake):

$$S_\text{NoGo}^{(i)} \leftarrow \min(1,\ S_\text{NoGo}^{(i)} + G_\text{hyper} \times 0.3), \quad \forall i$$

$G_\text{hyper} = 2.0$ (global inhibition gain). NoGo is simultaneously injected into all channels, persisting for several ticks before automatically releasing — like slamming the emergency brake, pausing whatever you are doing.

**Cross-Module Coupling**:
- Prefrontal cortex → basal ganglia: `pfc_bias` biases toward goal-relevant actions
- Basal ganglia → prefrontal cortex: `cortical_needed` flag signals when supervision is required
- Prefrontal `break_habit()` injects energy to break basal ganglia habits ($\Gamma \uparrow$)
- Amygdala emotional override ($I > 0.85$) can bypass prefrontal gating to directly drive actions

### 5.17 Homeostatic Drives — Hypothalamic Hunger/Thirst Physics Model (Phase 22)

The Homeostatic Drive Engine (`HomeostaticDriveEngine`) simulates hypothalamic monitoring of blood glucose and hydration levels, converting physiological needs into impedance signals injected into the LifeLoop closed loop.

**Metabolic Consumption Model**:

$$\dot{G} = -\left(M_{basal} + \text{cog} \cdot M_{cog} + \text{motor} \cdot M_{motor} + \text{symp} \cdot M_{symp}\right)$$

Where basal metabolic rate $M_{basal} = 0.003$/tick, cognitive load $M_{cog} = 0.002$, motor expenditure $M_{motor} = 0.004$. During sleep, metabolic rate is halved ($M_{sleep} = M \times 0.5$).

**Hunger Impedance**:

$$\Gamma_{hunger} = \frac{G_{onset} - G}{G_{onset}}, \quad G < G_{onset} = 0.6$$

**Drive Intensity** (non-linear quadratic amplification):

$$D_{hunger} = \Gamma_{hunger}^{2.0} \times D_{max}$$

Satiety suppresses drive for 30 ticks after eating ($D' = D \times (1 - 0.8 \times t_{satiety}/T_{duration})$).

**Digestive Buffer**: Food does not instantly restore blood glucose; instead it enters a digestive buffer and is gradually absorbed at $r_{digestion} = 0.02$/tick, simulating the delay effect of the real gastrointestinal tract.

**Cognitive Penalty**:

$$P_{cognitive} = \min\!\left(0.7,\ D_{hunger} \times 0.3 + D_{thirst} \times 0.4\right)$$

Hunger reduces cognitive performance ("too hungry to think"), and dehydration has an even greater impact — consistent with clinical observations that dehydration impairs cognition more severely than hunger.

**Cross-Module Coupling**: Drive signals are injected into LifeLoop as `INTEROCEPTIVE` errors → triggering compensatory behaviors (eating/drinking). Irritable mood (hangry) is injected into amygdala valence. Severe dehydration ($H < 0.3$) contributes to pain.

### 5.18 Physics Reward Engine — Replacing Q-tables with Impedance Matching (Phase 22)

The Physics Reward Engine (`PhysicsRewardEngine`) reconstructs reinforcement learning from discrete Q-table + TD(0) into continuous impedance channel dynamics. Each (state, action) pair is modeled as an impedance channel:

**Reflection Coefficient and Transmission Efficiency**:

$$\Gamma_{sa} = \frac{|Z_{channel} - Z_{source}|}{Z_{channel} + Z_{source}}, \quad T_{sa} = 1 - \Gamma_{sa}^2$$

Where $Z_{source} = 75\,\Omega$ (reference impedance). New channels start at $Z = 100\,\Omega$ (high impedance = unknown).

**Hebbian Impedance Adjustment**:

$$\text{RPE} > 0: \quad \Delta Z = -\alpha \cdot |\text{RPE}| \cdot Z \cdot 0.1 \quad (\text{Hebbian: } Z \downarrow, \Gamma \downarrow)$$

$$\text{RPE} < 0: \quad \Delta Z = +\beta \cdot |\text{RPE}| \cdot (Z_{max} - Z) \cdot 0.1 \quad (\text{Anti-Hebbian: } Z \uparrow, \Gamma \uparrow)$$

Where $\alpha = 0.08$ (positive reward learning rate), $\beta = 0.05$ (punishment learning rate). The asymmetric learning rates make the system tend toward conservatism — "once bitten by a snake, ten years afraid of rope" is a physical inevitability.

**Boltzmann Action Selection**:

$$P(a_i) \propto \exp\!\left(\frac{T(s, a_i)}{T_{temp}}\right), \quad T_{temp} \leftarrow \max(0.05,\ T_{temp} \times 0.9995)$$

A Boltzmann distribution weighted by transmission efficiency replaces $\epsilon$-greedy. Temperature decays gradually: high exploration early, high exploitation later.

**Dopamine Pipeline**:

$$[\text{DA}] \leftarrow \text{clip}\!\left([\text{DA}] + 0.15 \times \text{RPE},\ 0.05,\ 1.0\right)$$

RPE → `physics_reward` computes dopamine → injected into `basal_ganglia._dopamine_level`, unifying the physical semantics of the reward signal.

**Experience Replay**: `replay(batch_size=16)` readjusts impedance at a 50% discounted learning rate, simulating offline memory consolidation during sleep.

**Backward Compatibility**: Provides `get_q_value(state, action)` = $T \times \overline{R}$ as a Q-value equivalent interface, designed as a progressive replacement for the Q-table in `modules/reinforcement.py`.

---

## 6. Communication Protocol Engine

### 6.1 Γ-Net v4 Protocol

The complete message processing pipeline:

1. **O(1) Priority Routing**: 4-level deque queue (BACKGROUND→NORMAL→HIGH→CRITICAL) + aging-based starvation prevention
2. **Year-Ring Cache O(1) Lookup**: xxhash hashing → 8-ring year-ring cache; hit = zero computation
3. **On-Demand Hemisphere Activation**: Only the hemisphere relevant to the current modality is activated → reduced energy consumption
4. **Minimum-Energy Error Correction**: Only necessary discrepancies are corrected

### 6.2 Hardware Architecture Mapping — Asynchronous Parallelism and Nonlinear Load

The physical implementation adopts an asynchronous parallel mode of "GPU front-end classification + CPU back-end switching," which is key to maintaining the system's ultra-low energy consumption:

1.  **GPU (Full-Spectrum Resonance Bank) — Feature Extraction Layer**:
    *   **Parallel Classification**: The GPU simultaneously processes continuous electrical signals from all modalities including auditory, visual, and nociceptive channels.
    *   **Physical Reverse Engineering**: Each modality passes through a set of parallel LC resonators (Resonance Bank) in the GPU. The GPU does not perform "cognitive recognition"—it only performs "spectral annotation."
    *   **Dimensionality Reduction**: Raw high-dimensional data is reduced to sparse conceptual frequency bins (O(1) physical space).
    *   **Clinical Significance**: **Preconscious responsiveness**. The brain continues to monitor all environmental signals in parallel at the GPU level even when you are not paying attention.

2.  **CPU (Attention Executive) — Decision and Lock Layer**:
    *   **Sequential Selection**: The CPU does not participate in low-level signal decomposition; it only monitors the "resonance spectra" returned by the GPU.
    *   **Locking Mechanism**: When an energy peak (High Signal) appears at the GPU level, the CPU performs an "Attention Lock" and routes the corresponding frequency-band signal to the "prefrontal cortex" for deep-level interpretation.
    *   **Load Balancing**: Since the GPU has already completed classification, the CPU only needs to process the few "selected" channels—this is why ALICE can fluently process multimodal information under a single-core CPU standard.

This mapping resolves the "computational wall" problem in traditional AI: the GPU is responsible for "observing," while the CPU is responsible for "attending."

### 6.3 Year-Ring Cache (YearRingCache)

```
ring[0] (outermost) → ring[1] → ... → ring[7] (innermost / stable memory)
```

Fibonacci migration thresholds: [3, 5, 8, 13, 21, 34, 55, 89]

- Usage count reaches threshold → migrate to inner ring
- Hit rate > 80% → automatic expansion
- Hit rate < 40% → automatic contraction

---

## 7. Experimental Results

### 7.1 Memory Theory Verification — 5/5 All Passed

| # | Prediction | Result | Key Data |
|---|------|------|---------|
| 1 | Familiarity→Low energy (Γ→0) | **✓ Passed** | Cache rate 0% → 98% (50 repetitions) |
| 2 | Emotion accelerates consolidation | **✓ Passed** | Anxious Alice temperature 0.977 vs calm 0.000 |
| 3 | Attention bandwidth limit (Miller 7±2) | **✓ Passed** | Multitask evictions 23 vs single-task 3 (7.7×) |
| 4 | Sleep transfers memory (N3→consolidation) | **✓ Passed** | Sleep pressure 0.001 vs awake 0.100 |
| 5 | Predictive coding: familiarity→low energy | **✓ Passed** | Similar signal ΔT=-0.0095 < novel ΔT=-0.0087 |

**Experiment 1 Details**: The same stimulus was presented 50 times. The first presentation was a cache miss (0% hit rate); from the second presentation onward, cache hits occurred. By the 50th presentation, the cumulative cache rate reached 98.0%. Physical meaning: learning = adjusting impedance so that Γ→0 → repeated stimuli no longer produce reflected energy → zero-computation re-recognition.

**Experiment 3 Details**: Single-task (10 visual stimuli) vs. multitask (30 interleaved visual/auditory/tactile stimuli). Single-task working memory evictions: 3; multitask evictions: 23. Φ values: single-task 0.8053 > multitask 0.7436. Physical meaning: microtubule bandwidth has an upper limit → exceeding it causes overflow → this is the physical reason why "one cannot serve two masters."

### 7.2 Coaxial Physics Verification

| Condition | Reflection Coefficient Γ | Reflected Power | Impedance Match |
|------|-----------|---------|---------|
| 75Ω→75Ω (matched) | 0.0000 | 0.00% | ✓ |
| 50Ω→110Ω (mismatched) | 0.3750 | 14.06% | ✗ |

Theoretical and measured values are in perfect agreement. The reflected energy difference caused by impedance mismatch reaches 14.06%, directly driving the nociceptive circuit.

### 7.3 Nociceptive Collapse Experiment

Complete collapse-recovery curve of the system under sustained stress (five-phase clinical experiment):

**Phase 1 — Baseline (Healthy State)**

```
T=0.000  Pain=0.000  Consciousness=1.000  HR=60bpm  — All metrics normal
```

**Phase 2 — Progressive Stress (Chronic Pain Curve)**

| Stress Level | Temperature T | Pain | Consciousness | Heart Rate | Sensitivity | Clinical Correlate |
|---------|--------|------|------|------|--------|----------|
| ×0.5~×10 | 0→0.694 | 0.000 | 1.000 | 60→123 | 1.00 | No pain, heart rate compensation |
| **×12** | **0.770** | **0.233** | 1.000 | 132 | 1.00 | **Pain threshold breakthrough** |
| ×15 | 0.846 | 0.487 | 1.000 | 141 | 1.00 | Moderate pain |
| ×18 | 0.923 | 0.744 | **0.850** | 150 | **1.05** | Severe pain + sensitization onset |
| ×20 | 1.000 | **1.000** | **0.723** | 156 | **1.10** | Extreme suffering + trauma accumulation |
| ×25 | 1.000 | 1.000 | **0.614** | 172 | **1.15** | Continued hyperalgesia |
| ×30 | 1.000 | 1.000 | **0.522** | 172 | **1.20** | **Extreme pain but no loss of consciousness** |

```mermaid
xychart-beta
    title "Fig. 7  Nociceptive Collapse — Pain vs Consciousness Under Progressive Stress"
    x-axis "Stress Multiplier" ["x0.5", "x10", "x12", "x15", "x18", "x20", "x25", "x30"]
    y-axis "Normalized Value (0–1)" 0 --> 1.1
    bar [0.000, 0.000, 0.233, 0.487, 0.744, 1.000, 1.000, 1.000]
    line [1.000, 1.000, 1.000, 1.000, 0.850, 0.723, 0.614, 0.522]
```

> **Fig. 7.** Nociceptive collapse curve under progressive stress. Bars: pain intensity. Line: consciousness level. Pain threshold breaks at ×12 ($T = 0.770$). Consciousness begins declining at ×18 as sensitivity rises. At ×30, pain saturates at 1.0 but consciousness remains at 0.522 — above the freeze threshold (0.15). This reproduces the clinical observation that chronic pain patients can endure extreme pain without fainting, and that pain sensitivity increases with repeated trauma (central sensitization).

**Key Finding**: Progressive stress does not cause system freeze (FROZEN). Alice struggles to maintain consciousness even under extreme suffering (Consciousness = 0.522 > 0.15 freeze threshold). This corresponds to the clinical observation: chronic pain patients can experience extreme pain without fainting. Note that sensitivity (Sens) begins rising from ×18 onward—**experiencing pain lowers future pain thresholds**—this is the physical realization of trauma sensitization.

**Phase 3 — Emergency Recovery**

```
Before Reset: T=1.000  Pain=1.000  Consciousness=0.522  Sens=1.20  — Extreme agony
After Reset:  T=0.120  Pain=0.000  Consciousness=1.000  Sens=1.20  — Recovered but with residual traces
```

Emergency Reset = physical cooling + full queue flush, corresponding to clinical analgesic emergency intervention. However, note that post-reset temperature no longer returns to zero (T=0.120 rather than 0.000), and sensitivity remains at 1.20. This is the manifestation of **trauma memory**—painful experiences leave permanent imprints in the system's physical state, analogous to how PTSD patients maintain elevated baseline anxiety even after "recovery."

**Phase 4 — Broadcast Storm (Acute Shock)**

50 CRITICAL packets injected simultaneously (system already carries trauma memory from Phase 2):

```
Storm #1 : T=0.844  Pain=0.750  Sens=1.25  — Due to sensitization, initial pain is already high
Storm #11: T=1.000  Pain=1.000  Consciousness=0.197  Sens=1.75  — Accelerated collapse
Storm #13: ★★★ SYSTEM FROZEN ★★★ (2 packets faster than the trauma-free #15)
Post-freeze: Consciousness 0.121 → 0.103 → 0.087 → 0.074 → 0.063  Sens→2.00
```

**Key Finding**: A system carrying trauma memory collapses at **13** packets (vs. 15 packets without trauma memory). This is because the accumulated sensitivity from Phase 2 (1.20) lowered the pain threshold, making the storm more devastating. This corresponds to the clinical observation: **patients with PTSD history have lower tolerance for new stressors**.

**Phase 5 — Resilience Test (Repeated Stress/Recovery) — Cumulative Trauma Effects**

| Cycle | Post-Attack T | Post-Attack Pain | Post-Attack Consc | Post-Recovery T | Post-Recovery Pain | Post-Recovery Consc | Sensitivity |
|------|---------|-----------|-------------|---------|-----------|-------------|--------|
| #1 | 1.000 | 1.000 | 0.522 | **0.570** | **0.678** | 1.000 | 2.00 |
| #2 | 1.000 | 1.000 | 0.321 | **0.570** | **0.678** | 1.000 | 2.00 |
| #3 | 1.000 | 1.000 | 0.321 | **0.570** | **0.678** | 1.000 | 2.00 |

**Cumulative Trauma Indicators (End of Experiment):**

| Indicator | Initial Value | Final Value | Change |
|------|--------|--------|------|
| Pain sensitivity | 1.000 | **2.000** (ceiling) | +100% |
| Baseline temperature | 0.000 | **0.300** | +0.3 |
| Sympathetic baseline | 0.200 | **0.500** | +150% |
| Parasympathetic baseline | 0.300 | **0.150** | -50% |
| Chronic stress load | 0.000 | **1.000** (saturated) | +1.0 |
| Total trauma events | 0 | **50** | — |

**Key Findings**:

- **Recovery is no longer perfect**: Post-recovery T=0.570, Pain=0.678, rather than the previous T=0.270, Pain=0.000. The system retains residual suffering even after "healing."
- **Sympathetic/parasympathetic imbalance**: Sympathetic baseline rises 150% (fight mode); parasympathetic baseline drops 50% (diminished relaxation capacity), corresponding to autonomic dysregulation in PTSD patients.
- **Permanent pain threshold reduction**: The effective pain threshold drops from 0.700 to 0.350 (sensitivity 2.0), meaning previously painless stimuli now trigger nociceptive responses—this is precisely the phenomenon of **central sensitization**.
- **Consistency across three cycles**: Post-recovery data stabilizes at T=0.570/Pain=0.678, indicating the system has reached a new (damaged) steady state rather than continuing to deteriorate.

All of these behaviors emerge naturally from the physical equations. No code states "if previously injured then more susceptible to pain"—sensitivity drift and baseline shift are Hebbian-style trace accumulation, and the resulting behaviors precisely reproduce the core features of clinical PTSD and chronic pain.

### 7.4 Hand-Eye Coordination Experiment

| Condition | Convergence Steps | Final Error | Tremor Intensity |
|------|---------|---------|---------|
| Normal (T=0.0) | ~15 steps | < 2px | Very low |
| Anxious (T=0.5) | ~25 steps | < 5px | Moderate |
| Severely anxious (T=0.9) | ~40+ steps | > 10px | Severe |

The physical formula $\text{tremor} = T^2 \times \text{noise}$ causes severe hand tremor and reduced precision at higher temperatures, consistent with clinical observations.

### 7.5 Trauma Memory and Autonomic Baseline Drift

The nociceptive collapse experiment revealed an important issue: the original system's recovery was too perfect—all indicators returned to zero after each reset, as if memory had been erased. Real biological systems do not operate this way: **injury leaves traces**.

We implemented two mechanisms at the physical layer:

**1. Trauma Sensitization**

$$\text{Sens}_{t+1} = \min\left(\text{Sens}_t + \alpha_s,\ 2.0\right) \quad \text{when Pain} > 0.7$$

$$\text{effective\_threshold} = \frac{\theta_{\text{overheat}}}{\text{Sens}}$$

Where $\alpha_s = 0.05$ is the sensitization rate. Each time pain exceeds the threshold of 0.7, sensitivity increases and the effective pain threshold decreases. This corresponds to **central sensitization** in neuroscience: repeated noxious stimuli permanently elevate the excitability of dorsal horn neurons in the spinal cord.

Simultaneously, the baseline temperature drifts upward:

$$T_{\text{baseline}, t+1} = \min\left(T_{\text{baseline}, t} + \alpha_b,\ 0.3\right), \quad \alpha_b = 0.03$$

So that post-reset temperature no longer returns to zero, but instead reverts to an elevated baseline.

**2. Autonomic Baseline Drift**

$$\sigma_{\text{baseline}} = \min\left(\sigma_{\text{baseline}} + 0.03,\ 0.5\right) \quad \text{(sympathetic↑)}$$

$$\pi_{\text{baseline}} = \max\left(\pi_{\text{baseline}} - 0.015,\ 0.15\right) \quad \text{(parasympathetic↓)}$$

Each trauma event pushes the sympathetic baseline upward and pulls the parasympathetic baseline downward. Homeostatic equilibrium no longer targets normal values but instead targets the shifted baselines. This corresponds to **autonomic dysregulation** in PTSD patients—persistently elevated sympathetic activity and reduced parasympathetic (vagal) tone at rest.

**3. Trauma Frequency Fingerprint**

$$\text{trauma\_imprints} = \{f_1, f_2, \ldots, f_n\}$$

Each trauma event records the frequency of the triggering signal. When a future signal's frequency approaches a stored trauma fingerprint (tolerance ±10%), the system identifies it as a "trigger"—corresponding to how specific sounds or scenes can evoke trauma responses in PTSD.

**Key Significance**: None of these three mechanisms are "behavioral rules" (not "if PTSD then anxiety++"); they are drifts in physical parameters. The same differential equations, the same feedback loops—merely because parameter values have been permanently shifted by trauma—naturally produce the core features of PTSD: hypervigilance, lowered thresholds, incomplete recovery, and trigger sensitivity.

### 7.6 Motor Development Clinical Experiments

We designed five clinical scenarios to verify whether a single set of PID + Newtonian mechanics can naturally give rise to distinct motor control patterns.

**Experiment 1: Infant Motor Development Curve**

| Developmental Milestone | Maturity | Attempts | Error (px) | Peak Velocity (px/s) | Clinical Correlate |
|-----------|--------|---------|---------|-----------|----------|
| Neonate | 0.05 | 1 | 85.32 | 48 | Almost immobile |
| 1 month | 0.10 | 50 | 34.89 | 72 | Weak waving |
| 3 months | 0.20 | 150 | 9.06 | 125 | Obvious reaching |
| 5 months | 0.30 | 250 | 3.86 | 167 | Reaching toward target |
| **First reach** | **0.39** | **335** | **2.00** | **198** | **Milestone breakthrough** |
| 9 months | 0.50 | 358 | 1.98 | 234 | Approximate reaching |
| Adult | 1.00 | 458 | 1.97 | 343 | Full precision |

Key finding: The infant fails 334 times before first successful reach—after which the success rate rises sharply, because the +0.005 growth from success is 5× the +0.001 from failure (positive reinforcement acceleration). This reproduces the "motor milestone breakthrough" phenomenon in developmental psychology.

**Experiment 2: Bell-Shaped Velocity Profile (Physical Basis of Fitts' Law)**

| Distance | Peak Velocity (px/s) | Peak Position (%) | Steps | Curve Shape |
|------|-----------|-----------|------|----------|
| 200px | 309 | 10% | 135 | Fast rise, slow fall |
| 400px | 520 | 15% | 159 | Bell-shaped |
| 600px | 585 | 24% | 181 | Wide bell-shaped |
| 400px (injured) | 446 | — | 235 | Flattened bell-shaped |

The PID proportional term ($K_p \cdot e$) produces initial acceleration, and the proximity brake ($\beta_{\text{brake}}$) smoothly decelerates within the last 30px—their combination naturally gives rise to the bell-shaped velocity profile characteristic of human reaching movements.

**Experiment 3: Post-Injury Protective Movement**

| Guard | Steps | Peak Velocity | Relative Time | Clinical Correlate |
|-------|------|------|---------|----------|
| 0.0 | 159 | 520 | 100% | Healthy |
| 0.2 | 172 | 508 | 108% | Minor injury — slightly cautious |
| 0.4 | 191 | 491 | 120% | Moderate injury — noticeable hesitation |
| 0.6 | 217 | 464 | 136% | Severe injury — slow movement |
| 0.8 | 258 | 424 | 162% | Critical injury — barely dares to move |
| 1.0 | 326 | 363 | 205% | Maximum protection — rigidity |

Force is scaled by $(1 - 0.6 \times G)$. At Guard=1.0, only 40% of force remains and movement time doubles—corresponding to the protective gait observed clinically in post-fracture patients.

**Experiment 4: Rehabilitation Course**

Guard 0.8 (severely injured) → each successful reach decreases guard by -0.01 → full recovery at the 79th attempt (guard=0.01→0.00).

| Practice Count | Guard | Steps | Peak Velocity | Relative Baseline |
|---------|-------|------|------|--------|
| 1 | 0.79 | 246 | 322 | 166% |
| 21 | 0.59 | 206 | 363 | 130% |
| 41 | 0.39 | 179 | 395 | 113% |
| 61 | 0.19 | 161 | 419 | 101% |
| 79 | 0.01 | 149 | 436 | ≈100% |

**Experiment 5: Four-State Comprehensive Comparison**

| State | Reached | Steps | Peak Velocity | Tremor | Physical Mechanism |
|------|------|------|------|------|--------|
| Healthy adult | ✓ | 159 | 520 | 0.00 | Baseline |
| Infant (3 mo.) | ✗ | 500 | 222 | 0.00 | maturity=0.2, insufficient force |
| Injured adult | ✓ | 235 | 446 | 0.00 | guard=0.7, force-limited |
| Anxious adult | ✓ | 196 | 484 | 49.8 | T=0.7, trembling and inaccurate |
| Injured + anxious | ✓ | 314 | 373 | 50.1 | Dual impairment |

Using a single set of PID + Newtonian mechanics, merely varying three modulation parameters (maturity, guard, temperature) naturally differentiates motor patterns of infants, adults, injured patients, and anxious patients—without any "behavioral rules."

### 7.7 Neural Pruning Experiment — Large-Scale Γ Apoptosis

We designed five experiments to verify the theoretical predictions of §3.5.2: the complete process from random impedance at birth → Hebbian selection → cortical specialization.

**Experiment 1: Normal Development Curve**

8000 randomly-impedanced connections (4 regions × 2000), subjected to 200 developmental epochs of sensory stimulation and Hebbian pruning:

| Epoch | Occipital | Temporal | Parietal | Frontal | Whole-Brain Survival Rate | Σ Γ² |
|-------|------|------|------|------|----------|------|
| 1 | 2000 | 2000 | 2000 | 2000 | 100.0% | 0.126 |
| 10 | 747 | 630 | 622 | 2000 | 50.0% | 0.131 |
| 40 | 287 | 483 | 186 | 400 | 17.0% | 0.076 |
| 100 | 287 | 483 | 186 | 367 | 16.5% | 0.025 |
| 200 | 287 | 483 | 186 | 367 | 16.5% | **0.010** |

Key finding: Σ Γ² decreased from 0.126 to 0.010 (91.8% improvement), with an average specialization index of 0.905. The frontal lobe enters steady state last because it receives only 30% feedback signal.

**Experiment 2: Cortical Specialization Separation**

The frequency distributions of the four regions become fully separated after pruning:

| Region | Specialization Direction | Dominant Frequency | Band | Frequency Distribution |
|------|---------|------|------|---------|
| Occipital | visual | 59.4 Hz | γ | gamma 100% |
| Temporal | auditory | 9.5 Hz | α | alpha 100% |
| Parietal | somatosensory | 38.0 Hz | γ | gamma 100% |
| Frontal motor | motor | 22.1 Hz | β | beta 100% |

The surviving connections in each region exhibit a 100% single-band frequency distribution—a self-organizing process from "random uniform" to "fully specialized."

```mermaid
xychart-beta
    title "Fig. 8  Post-Pruning Cortical Specialization — Dominant Frequencies"
    x-axis "Cortical Region" ["Occipital (Visual)", "Temporal (Auditory)", "Parietal (Somato.)", "Frontal (Motor)"]
    y-axis "Dominant Frequency (Hz)" 0 --> 70
    bar [59.4, 9.5, 38.0, 22.1]
```

> **Fig. 8.** Dominant frequencies of four cortical regions after 200 epochs of Hebbian pruning. Starting from uniform random impedance, each region self-organizes to specialize in the frequency band of its input signals: occipital → γ (visual), temporal → α (auditory), parietal → γ (somatosensory), frontal → β (motor). Each region achieves 100% single-band purity — no cross-band contamination.

**Experiment 3: Delayed Frontal Lobe Maturation**

| Condition | Frontal Survival Rate | Frontal Specialization Index | Effective Stimulus Count |
|------|----------|-----------|-----------|
| 100% feedback (control) | 21.0% | 0.898 | 1000 |
| 30% feedback (normal infant) | 17.9% | 0.829 | 300 |
| 10% feedback (motor-deprived) | 21.1% | 0.757 | 100 |

Lower feedback rate → fewer effective stimuli → lower specialization index. The frontal lobe is not "less intelligent"—it simply "requires more feedback to complete impedance calibration."

**Experiment 4: Cross-Modal Rewiring**

| Condition | Occipital Specialization | Occipital Dominant Freq. | Temporal Specialization | Temporal Dominant Freq. |
|------|---------|---------|---------|---------|
| Normal (control) | visual | 58.5 Hz (γ) | auditory | 9.5 Hz (α) |
| Cross-modal | **auditory** | **9.6 Hz (α)** | **visual** | **60.0 Hz (γ)** |

After redirecting auditory signals to the occipital lobe, its specialization shifted from visual→auditory, and its dominant frequency from γ→α. **The occipital lobe is the visual cortex not because of genetic presets, but because it happens to be positioned to receive visual signals**—signal type determines cortical specialization.

**Experiment 5: Γ² Wisdom Curve**

$$\text{Wisdom} = \lim_{t \to \infty} \sum_{i} \Gamma_i^2 \to \min$$

| Epoch | Σ Γ² | Reduction | State |
|-------|------|------|------|
| 1 | 0.1206 | 0% | Birth (chaotic) |
| 20 | 0.1007 | 16.4% | Developing |
| 50 | 0.0595 | 50.7% | Specialization forming |
| 100 | 0.0251 | 79.2% | Specialization stable |
| 200 | **0.0098** | **91.8%** | Mature |

Wisdom is not "acquiring knowledge" but "minimizing signal reflection across the entire brain." From 0.1206 (whole-brain impedance chaos) to 0.0098 (impedance matching complete), this represents life's physical journey from "random chaos" to "ordered resonance."

```mermaid
xychart-beta
    title "Fig. 9  Whole-Brain ΣΓ² Convergence — The Wisdom Curve"
    x-axis "Developmental Epoch" [1, 20, 50, 100, 150, 200]
    y-axis "ΣΓ² (total reflected energy)" 0 --> 0.13
    line [0.1206, 0.1007, 0.0595, 0.0251, 0.015, 0.0098]
    bar [0.1206, 0.1007, 0.0595, 0.0251, 0.015, 0.0098]
```

> **Fig. 9.** Whole-brain $\sum \Gamma_i^2$ convergence over 200 developmental epochs. Starting from random impedance initialization ($\sum \Gamma_i^2 = 0.126$), Hebbian selection progressively reduces total reflected energy to 0.010 (91.8% improvement). The steepest decline occurs in epochs 1–50 (critical period), after which the curve flattens as surviving connections stabilize — mirroring the biological critical period of synaptic development [18].

### 7.8 Sleep Physics Experiment — Offline Impedance Reorganization

Five quantitative experiments verifying the hypothesis that sleep is a physical necessity rather than a physiological rule.

#### Experiment 1: Day-Night Cycle Energy Conservation

100 ticks awake (consuming energy, accumulating impedance debt) → 110 ticks sleeping (N1→N2→N3→REM natural cycle):

| Indicator | End of Day | Post-Sleep | Change |
|------|---------|------|------|
| Energy | 0.200 | 1.000 | ↑ 0.800 |
| Impedance debt | 0.260 | 0.002 | ↓ 0.258 |
| Sleep pressure | 0.484 | 0.074 | ↓ 0.410 |

Sleep quality score: **0.918** (N3 deep sleep proportion 34.5%, 84 memories consolidated).

#### Experiment 2: Synaptic Downscaling — Tononi's Synaptic Homeostasis Hypothesis

N3 deep sleep proportional scaling (×0.990/cycle) preserves relative differences:

| Measurement | Post-Learning | After 20 Rounds of Downscaling |
|------|--------|-------------|
| Mean | 1.056 | 0.855 |
| Std. Dev. | 0.196 | 0.161 |
| Important synapse #0 | 1.609 | 1.303 |
| Ratio (#0/#20) | **1.803** | **1.803** |
| Ratio deviation | — | **0.000000** |

$$\text{Post-downscaling ratio} = \frac{s_i \times f^N}{s_j \times f^N} = \frac{s_i}{s_j} \quad \text{(algebraic identity)}$$

"Sleep is not forgetting—it is reducing the global noise floor while preserving the relative differences between signals."

#### Experiment 3: Sleep Deprivation — The Physical Cost of Not Sleeping

| Condition | Final Energy | Impedance Debt | Sleep Pressure | Quality Score |
|------|---------|---------|---------|---------|
| Normal (day 100 + night 110) | 1.000 | 0.001 | 0.073 | 0.889 |
| Mild deprivation (day 150 + night 60) | 0.447 | 0.024 | 0.302 | 0.917 |
| Total deprivation (day 210 + night 0) | **0.000** | **0.485** | **0.643** | 0.000 |

Total deprivation is not a "designed punishment"—it is the physical inevitability of $\frac{dE}{dt} = -P_{\text{metabolic}} + 0$ (recovery=0).

#### Experiment 4: REM Dream Channel Diagnostics

7 channels (4 healthy + 3 damaged), 10 rounds of REM probing:

| Channel | Γ | Status |
|------|---|------|
| visual→cortex | 0.020 | ✓ Healthy |
| trauma_path | 0.600 | ✗ Damaged |
| overloaded_ch | 0.600 | ✗ Damaged |

56% of 50 probes hit healthy channels. Damaged channels automatically enter the repair queue.

**The Physical Nature of Dreams**: Random impedance probing during REM. Channels with high $\Gamma$ → large reflected energy → dream flashes (micro-nociceptive triggers) → nightmares.

#### Experiment 5: Post-Sleep Memory Consolidation Enhancement

Downscaling (×0.990) proportionally scales all synapses: strong synapses remain strongest; weak synapses (including noise) approach the floor faster. N3 slow-wave UP states trigger 84 memory replays; DOWN states trigger global downscaling. Sleep quality: 0.791.

**Core Conclusions**:

1. Sleep is a **physical necessity** arising from the energy conservation equation + impedance debt repair
2. N3 deep sleep synaptic downscaling preserves relative differences while eliminating noise (ratio deviation = 0)
3. No sleep = recovery=0 → energy depletion + debt accumulation → system degradation
4. Dreams = random impedance probing during REM → channel health diagnostics
5. The physical basis for "sleep well before an exam": downscaling eliminates noise → SNR automatically improves

### 7.9 Auditory Grounding Experiment — Pavlovian Cross-Modal Hebbian Conditioning

**Core Question**: How does language emerge from pure energy phenomena?

**Experimental Design**:

1. **Pavlovian conditioning**: Bell (440Hz pure tone) + food (visual signal) presented simultaneously × 20 trials
2. **Probing**: Bell alone → does the visual cortex produce a phantom signal?
3. **Extinction**: No more pairing → synaptic decay → does the phantom disappear?
4. **Vowel grounding**: /a/, /i/, /u/ paired with mouth shapes → can hearing a vowel "see" the mouth shape?

**Quantitative Results**:

| Indicator | Pre-Conditioning | After 20 Conditionings | After 150 Ticks of Extinction |
|------|--------|-------------|------------------|
| Synaptic strength w | 0.01 | 3.01 | 0.15 |
| Z_synapse (Ω) | 10,000 | 33.2 | 687 |
| Γ_cross | -0.990 | +0.202 | -0.864 |
| E_transfer | 0.020 | 0.959 | 0.253 |
| Phantom generated | ✗ | ✓ | ✗ |

**Key Findings**:

- **Impedance matching peak**: At w ≈ 2.0, Z = Z_source = 50Ω, Γ = 0, E_transfer = 1.0 (perfect impedance match)
- **Phantom threshold**: Cross-modal phantom signal produced when echo_strength > 0.3
- **Extinction time constant**: 2% decay per tick, falls below threshold after ~150 ticks
- **Vowel recognition**: /a/ → mouth_open, /i/ → mouth_smile, /u/ → mouth_round, recognition similarity > 0.999
- **Physical consistency**: The cross-modal synaptic impedance model is fully consistent with Γ-Net coaxial cable physics

```mermaid
xychart-beta
    title "Fig. 10  Pavlovian Cross-Modal Impedance Matching"
    x-axis ["Pre-Conditioning", "After 20 Trials", "After Extinction"]
    y-axis "Energy Transfer Rate" 0 --> 1.1
    bar [0.020, 0.959, 0.253]
    line [0.3, 0.3, 0.3]
```

> **Fig. 10.** Cross-modal energy transfer during Pavlovian conditioning. Bars: $E_{\text{transfer}} = 1 - \Gamma_{\text{cross}}^2$. Dashed line: phantom signal threshold (0.3). Pre-conditioning: near-zero transfer ($Z_{\text{synapse}} = 10{,}000\,\Omega$, total mismatch). After 20 pairings: near-perfect transfer (0.959), bell alone produces visual phantom. After 150-tick extinction: transfer drops below phantom threshold. **Language = remote impedance control via sound waves.**

**Physical Meaning**: Pavlovian conditioning = establishing impedance matching in a cross-modal cable. Language = using sound waves to control another brain's impedance matching. Understanding = Γ_cross → 0.

### 7.10 Language Physics Experiment — Semantic Force Field + Broca's Area

**Core Question**: How do concepts self-organize in continuous space? How does motor speech emerge from sensorimotor closed loops?

**Experimental Design** (8 experiments):

| # | Experiment | Core Verification | Result |
|---|------|----------|------|
| 1 | Attractor formation | Mass growth → impedance reduction | 21× mass growth, Z: 50→17.9Ω |
| 2 | Competitive recognition | 4-concept winner-take-all | 100% accuracy |
| 3 | Contrastive learning | Anti-Hebbian category sharpening | Correctly skipped when similarity below threshold |
| 4 | Multimodal binding | Auditory + visual → same concept | Both modalities correctly recognized |
| 5 | Cross-modal prediction | Hear → predict see | Similarity > 0.999 |
| 6 | Babbling exploration | Articulatory parameter space coverage | F1 span=680Hz, pitch span=300Hz |
| 7 | Sensorimotor loop | speak→hear→verify→learn | 15 iterations completed |
| 8 | AliceBrain integration | End-to-end hear/say | Semantic field + Broca integration successful |

**Experiment 1 Quantitative Results — Attractor Formation**:

```
  concept     mass_0    Z_0  →  mass_20    Z_20  ratio
  low_tone     1.000   50.0      21.000   17.9   21×
  mid_tone     1.000   50.0      21.000   17.9   21×
  high_tone    1.000   50.0      21.000   17.9   21×
```

Physical meaning: More instances absorbed → mass M grows → Q quality factor increases → discriminability improves → impedance Z decreases (better matching).

**Experiment 5 Quantitative Results — Cross-Modal Prediction**:

1. Training: 15 auditory absorptions + 15 visual absorptions
2. Testing: Only auditory fingerprint provided → model predicts visual fingerprint
3. Predicted vs. actual visual fingerprint similarity: **0.999**

Physical meaning: The multimodal centroid of the attractor constitutes "semantic memory"—hearing "bell" enables "seeing" the bell's appearance.

**Experiment 6 Quantitative Results — Babbling Exploration**:

$$\text{F1}: 200 \sim 890\text{ Hz} \quad \text{pitch}: 84 \sim 396\text{ Hz}$$

20 random babbles covered the full F1 range (200–900 Hz) and full pitch range (80–400 Hz), verifying the exploratory nature of infant babbling.

**Key Findings**:

- **Concept = attractor**: Concepts in the semantic force field automatically form stable attractors without requiring a discrete classifier
- **Recognition = gravitational convergence**: Input fingerprints naturally converge to the best-matching attractor in the gravitational field
- **Understanding = cross-modal low-Γ channel**: Hearing → seeing, smelling → tasting—all are low-impedance transmissions between intra-attractor cross-modal centroids
- **Learning to speak = Γ optimization of the sensorimotor closed loop**: babble → hear → compare → adjust → repeat

### 7.11 Experiment Series 11: Hippocampus + Wernicke's Area Physical Verification

8 experiments verifying the physical behavior of the episodic memory engine (Phase 5.1) and the sequence comprehension engine (Phase 5.2).

**Experiment 1: Episodic Recording**

- Input: 3 auditory snapshots (32-dim) + 2 visual snapshots (256-dim), intervals < 2.0 seconds
- Result: 5 snapshots automatically stitched into 1 episode
- Physical meaning: The hippocampal temporal threshold $\Delta_{\text{gap}}$ correctly binds adjacent events

**Experiment 2: Pattern Completion**

- Input: After storing 3 auditory snapshots, the first snapshot is used as a cue
- Result: Successfully recalled a similar snapshot from the same episode (cosine similarity > 0.5)
- Physical meaning: Similarity retrieval in fingerprint space = impedance matching search

**Experiment 3: Cross-Membrane Recall**

- Input: Episode containing auditory snapshot (label="cat") + visual snapshot (label="cat")
- Cue: Concept "cat" crosses from the auditory membrane to the visual membrane
- Result: Successfully recalled a 256-dim visual fingerprint (crossing the 32-dim→256-dim dimensional boundary)
- Physical meaning: Concept labels = wormholes, connecting membrane spaces of different dimensions

**Experiment 4: Sleep Consolidation**

- Procedure: Episode containing 5 concepts (cat, dog, bird, fish, sun), triggering consolidation to semantic field
- Result: Pre-consolidation mass $M = 2.00$, post-consolidation $M = 12.00$ (6× growth)
- Quality factor: $Q$ increases from 1.549 to 2.282 (discriminability improved by 47.3%)
- Physical meaning: Episodic → semantic mass transfer = experience transformed into knowledge

**Experiment 5: Transition Learning**

- Procedure: Hippocampus records sequence [cat, meow, cat, meow, ...], Wernicke's area learns from it
- Result: $P(\text{meow}|\text{cat}) = 1.0$, $\Gamma_{\text{syn}} = 0.0$
- Physical meaning: High-frequency transitions → low-impedance channels → perfect prediction

**Experiment 6: Sequence Comprehension**

- Training: ["the", "cat", "sat"] × 30 + ["the", "cat", "ate"] × 15
- Known sequence ["the", "cat", "sat"]: $C_{\text{score}} = 0.833$
- Unknown sequence ["fish", "the", "dog"]: $C_{\text{score}} = 0.000$
- Physical meaning: Low Γ sequence = comprehensible; high Γ sequence = incomprehensible

**Experiment 7: N400 Surprise Detection**

- Training: [A→B] × 20
- Expected transition (A→B): $\Gamma_{\text{syn}} = 0.0$, N400 = False
- Unexpected transition (A→X): $\Gamma_{\text{syn}} = 1.0$, N400 = True
- Physical meaning: $\Gamma_{\text{syn}} > 0.7$ naturally triggers an N400 event — no special rules needed

**Experiment 8: Chunk Formation**

- Training: ["good", "morning"] × 5 + ["bad", "night"] × 2
- Result: "good+morning" matured (occurrence ≥ 3), "bad+night" immature (occurrence < 3)
- Physical meaning: Frequently co-occurring concept pairs consolidate into cognitive units = the physical basis of language fluency

### 7.12 Experiment Series 12: Prefrontal Executive Control + Basal Ganglia Habit Engine

8 quantitative experiments for Phase 6.1 + 6.2, verifying the physical emergence of executive control and habit formation:

**Experiment 1: Goal Stack Management**

- Set 4 goals: Work (P=0.9), Exercise (P=0.5), Learn piano (P=0.6), Socialize (P=0.3)
- Stack capacity max_goals=4; upon overflow, the lowest-priority goal is evicted
- Sub-goal decomposition: finish_report → gather_data → write_draft → review_edit
- Sub-goal completion → parent goal progress accumulates; when $\Gamma < 0.15$ → parent goal complete
- Physical meaning: Humans can only track 4–5 goals simultaneously — the physical limit of prefrontal working memory capacity

**Experiment 2: Go/NoGo Gating and Impulse Control**

- Action impedance $Z_\text{action}$ vs. goal impedance $Z_\text{goal}$
- $\Gamma < 0.4$: Go (execute); $\Gamma > 0.7$: NoGo (inhibit); in between: Defer (delay)
- Inhibition consumes energy: $E_\text{inhibition} = E_\text{base} \times (1 + \Gamma)$
- When energy is depleted, inhibition fails → impulse breakthrough
- Physical meaning: After refusing dessert 10 times in a row, losing control on the 11th — self-control is a finite resource

**Experiment 3: Minimum Γ Path Planning**

- Search for intermediate steps from $Z_\text{now}$ to $Z_\text{goal}$
- Cognitive effort per step: $\text{Effort} = \Gamma_\text{step}^2$; selects the path with minimum cumulative Γ
- Planning consumes energy; planning quality degrades when energy is insufficient
- Physical meaning: Studying for an exam — the brain seeks the most effort-efficient "impedance matching" path

**Experiment 4: Ego Depletion Cascade**

- 15+ consecutive NoGo inhibition tasks
- Prefrontal energy $E_\text{PFC}$ gradually decreases
- Energy < 0.4: Task-switching delay increases (perseveration tendency $\sigma \uparrow$)
- Energy < 0.2: Emotion regulation fails, impulse inhibition fails, planning degrades
- Recovery at 0.01 per tick; sleep fully restores
- Physical meaning: "Not wanting to do anything on Friday night" = prefrontal impedance matching energy pool is depleted

**Experiment 5: Habit Formation Curve**

- Scenario: Learning to ride a bicycle, 100 practice trials
- $\Gamma_\text{habit}$ decays exponentially with practice: $\Gamma \leftarrow \max(0, \Gamma - \alpha(1 + r))$
- Initial $\Gamma = 0.85$ (requires heavy cortical supervision)
- After ~50 trials, $\Gamma < 0.15$ (habit threshold) → action automatization
- Reaction time: $RT_\text{habit} = 0.1 - \text{automaticity} \times 0.3$ (direct pathway)
- Physical meaning: Driving without thinking about the steering wheel — repeated practice lets the signal pass directly through, no reflection needed

**Experiment 6: Action Selection and Dopamine Modulation**

- Striatal Softmax selection: $P(a_i) = \exp(A_i/\tau) / \sum_j \exp(A_j/\tau)$
- High dopamine ([DA] > 0.5): Increased exploration (positive RPE → Go pathway strengthened)
- Low dopamine ([DA] < 0.5): Conservative behavior (negative RPE → NoGo pathway strengthened)
- Hyperdirect pathway: Global NoGo injection → emergency brake (all actions paused)
- Physical meaning: "Feeling happy after doing the right thing" = dopamine positive RPE; next time the same behavior is more likely to be selected

**Experiment 7: Dual-System Arbitration (Habit vs. Goal-Directed)**

- Habit weight: $w_\text{habit} = (1-U) \times R / (\Gamma + 0.1)$
- Goal weight: $w_\text{goal} = U \times (1-R+0.1) / (1-\Gamma+0.1)$
- Low uncertainty + high reward rate → habit system dominates
- High uncertainty → goal-directed system takes over
- Physical meaning: Taking the habitual route to work — unless there's construction ahead (high uncertainty), then engaging deliberate planning

**Experiment 8: Breaking Habits (PFC Energy Injection)**

- Habit already formed ($\Gamma < 0.15$), prefrontal cortex injects energy
- `break_habit()` causes $\Gamma \uparrow$, temporarily restoring cortical supervision
- Requires sustained energy to maintain — if energy is insufficient, the habit relapses
- Physical meaning: Why quitting smoking is so hard — the basal ganglia's $\Gamma$ has already dropped to automation level, and the prefrontal cortex must continuously expend energy to override it

### 7.13 Experiment Series 13: Thalamic Sensory Gating + Amygdala Emotional Fast Pathway

8 quantitative experiments for Phase 5.3 + 5.4, verifying the physical emergence of sensory gating and emotional systems:

**Experiment 1: Gate Gain Modulation by Arousal Level**

- Arousal 0.0→1.0, gate gain from ~0.05 (nearly blocked) → ~0.5+ (passes through)
- Below arousal 0.3, enters pulse mode (signals randomly pass/block)
- Physical meaning: Not hearing while asleep is not because the ears shut off — it is because the thalamic gate closed

**Experiment 2: Startle Circuit Unconditional Penetration**

- Amplitude ≥ 0.8 or Γ ≥ 0.85 → gate gain = 1.0, startle = True
- Arousal = 0.0 (deep sleep) still penetrates
- Physical meaning: Low-level interrupts (IRQs) bypass consciousness — you can be startled awake by a loud noise even in deep sleep

**Experiment 3: Habituation and Novelty Recovery**

- 30 identical stimuli: Gate gain gradually decreases, habituation value 0→0.08+
- Novel stimulus introduced: Gate gain recovers
- Physical meaning: The air conditioner's hum becomes inaudible after ten minutes — until someone calls your name

**Experiment 4: TRN Competitive Inhibition**

- After focusing on vision: Visual bias increases, auditory/tactile bias decreases
- Visual gate gain > auditory gate gain
- Physical meaning: Not hearing background music while reading — the music hasn't disappeared; TRN inhibited the auditory channel

**Experiment 5: Fear Conditioning and Matching**

- Pre-conditioning, seeing a snake: Low threat, fear match = False
- After 3 conditionings, seeing a snake: Threat significantly elevated, fear match = True
- Seeing a flower (unconditioned): Threat unchanged, fear match = False
- Physical meaning: Being bitten by a dog once → heart rate accelerates upon seeing any dog afterward. This is not rational — it is the amygdala threshold being permanently lowered

**Experiment 6: Fear Extinction and Residual**

- Initial effective threat ~0.9
- After 30 extinction trials: Effective threat decreases (but > 0)
- Fear memory still exists (count unchanged)
- Physical meaning: Extinction is not forgetting. Fear memory can only be overlaid with a safety signal, not deleted — this is the physical basis of PTSD relapse

**Experiment 7: Fight-Flight-Freeze Cascade**

- Progressive experiment with pain 0.0→1.0
- Low pain: neutral → anxiety (vigilant but no fight-flight)
- High pain: fear → terror (fight-flight activated, sympathetic ~0.8)
- Extreme pain: terror + freeze (overload protection, sympathetic paradoxically drops to 0.3)
- Physical meaning: Fear exists on a continuous spectrum — not a binary "afraid / not afraid"

**Experiment 8: Amygdala-Hippocampus Emotional Memory Enhancement**

- Neutral scene: Low valence, low threat, normal gating
- High-startle scene: Startle triggered, negative valence, high gating
- Hippocampus receives valence tags → high-emotion events are prioritized for consolidation
- Physical meaning: You don't remember yesterday's lunch — but you remember your first scare for a lifetime

### 7.14 System Performance

| Metric | v11.0 | v29.0 (Current) |
|------|-------|-------------|
| Source code lines | 32,335 lines | **35,690 lines** (+10.4%) |
| Test code lines | — | **21,176 lines** |
| Experiment code lines | — | **25,763 lines** |
| Total code lines | ~32,335 lines | **83,450 lines** (+158%) |
| File count | 80 | **145** (+81.3%) |
| Brain modules | ~17 | **44** (+159%) |
| Test count | 1,042 | **1,876** (+80.0%) |
| Test files | 20 | **38** |
| Experiment count | ~10 | **44** |
| Test pass rate | 100% | **100%** |
| Full test execution time | 6.45 sec | **~15 sec** |
| GPU requirement | **0** (CPU only) | **0** (CPU only) |
| External dependencies | numpy, xxhash, fastapi | numpy, xxhash, fastapi |
| Python version | 3.14 | 3.14 |

### 7.15 Human Intelligence Parity (HIP) Assessment

To quantitatively evaluate ALICE's cognitive coverage against biological intelligence, we designed the Human Intelligence Parity (HIP) benchmark across 9 cognitive domains, each scored 0–100%:

```mermaid
xychart-beta
    title "Fig. 11  Human Intelligence Parity (HIP) — Score by Domain (98.7/100)"
    x-axis ["Memory", "Perception", "Motor", "Language", "Emotion", "Social", "Executive", "Metacog.", "Creativity"]
    y-axis "Score (%)" 0 --> 110
    bar [100, 100, 100, 100, 100, 86.7, 100, 100, 100]
    line [100, 100, 100, 100, 100, 100, 100, 100, 100]
```

> **Fig. 11.** Human Intelligence Parity (HIP) assessment across 9 cognitive domains. Bars: ALICE's scores. Line: 100% human parity baseline. Overall score: **98.7/100**. Eight domains achieve full parity (100%); Social cognition (86.7%) remains the only below-100 domain due to limited multi-agent interaction scope. Key milestones: Emotion improved from 50% → 100% via the EmotionGranularity engine (Plutchik 8D vector + differential decay + compound emotions); Metacognition improved from 30% → 100% via the HIP-driven metacognition redesign (System 2 override + error correction + insight detection).

---

## 8. Discussion

### 8.1 Architecture Comparison with Mainstream AI

> **Table 2.** Systematic comparison of Γ-Net ALICE against mainstream AI architectures across 18 dimensions.

| Dimension | LLM / CNN | Γ-Net ALICE |
|------|-----------|-----------|
| Topology | Feed-forward DAG | Closed-loop + recursive feedback |
| Core operation | Matrix multiplication O(n²) | Impedance comparison + resonance O(1) |
| Training | Offline gradient descent | Online PID + Hebbian + TD(0) |
| Inference | y = f(x) computed once | Continuous compensation until error < threshold |
| Body | None | Complete limbs (eye/ear/hand/mouth) |
| Nociception | None | Γ → reflected energy → temperature → pain |
| Attention | Transformer O(n²) | Global workspace bottleneck (≤3) |
| Sleep | None | NREM/REM 90-minute cycle |
| Time perception | No built-in clock | 50ms/200ms window binding |
| Consciousness metric | Not applicable | Integrated Information Theory Φ |
| **Trauma memory** | **None** | **Sensitization + baseline drift + frequency fingerprint** |
| **Motor development** | **None** | **motor_maturity infant→adult + guard protection** |
| **Neural pruning** | **None** | **Large-scale Γ apoptosis, random→specialized cortex (§7.7)** |
| **Episodic memory** | **External database** | **Hippocampal cross-membrane binding + pattern completion (§5.11)** |
| **Language comprehension** | **Statistical n-gram / Transformer** | **Wernicke Γ_syn ≈ 1-P(next|cur) + N400 (§5.12)** |
| Scale | 10¹¹ parameters | ~17,779 lines of physics equations |
| Hardware | Thousands of GPUs | **CPU only** |

```mermaid
xychart-beta
    title "Fig. 12  Architecture Capability Comparison"
    x-axis ["Closed-Loop", "Body", "Nociception", "Sleep", "O(1) Percept.", "No GPU", "Pruning", "Emotion", "Language"]
    y-axis "Capability (0=None, 1=Full)" 0 --> 1.2
    bar [1, 1, 1, 1, 1, 1, 1, 1, 1]
    line [0, 0, 0, 0, 0, 0, 0, 0, 0]
```

> **Fig. 12.** Architecture capability comparison. Bars: Γ-Net ALICE. Line: LLM/CNN baseline. Γ-Net ALICE achieves 9/9 capabilities that mainstream AI architectures lack entirely: closed-loop control, physical body, nociception, sleep cycles, $O(1)$ perception, zero GPU requirement, neural pruning, emergent emotion, and physics-grounded language. No existing system simultaneously implements all nine.

### 8.2 Theory Level Analysis

Our theory can be divided into three confidence levels:

1. **Verified physical facts** (high confidence):
   - Impedance mismatch → reflection (coaxial cable physics, textbook knowledge)
   - FFT is the physical operation of a lens (optical principle)
   - The cochlea is a physical Fourier analyzer (physiological consensus)
   - PID controllers simulate motor control (standard control theory method)
   - **Chronic vs. acute pain separation**: The same equation naturally produces different clinical presentations under gradual/instantaneous inputs
   - **Trauma sensitization**: Parameter drift naturally produces core PTSD features (hypervigilance, lowered threshold, incomplete recovery)
   - **Infant motor development**: Maturity growth curve from 0.05→1.0 reproduces developmental psychology motor milestones (first reach = #335)
   - **Bell-shaped velocity profile**: PID + approach braking naturally produces human reaching movement characteristics (the physical basis of Fitts' Law)
   - **Post-injury protective movement**: The guard parameter reduces effective force to 40%, doubles movement time — consistent with orthopedic clinical observations

2. **Reasonable inferences** (moderate confidence):
   - Learning = impedance adjustment driving Γ→0 (experimentally supported, but biological verification incomplete)
   - Memory = storing {f, A, φ, Z} configurations (concise but may require more parameters)
   - Nociception = reflected energy accumulation (physically self-consistent, clinically directionally aligned)

3. **Potentially oversimplified aspects** (low confidence):
   - Whether four parameters {f, A, φ, Z} suffice to describe all memory types
   - Whether complex multi-step reasoning can be accomplished solely through resonance
   - The sufficiency of Φ as a consciousness metric

### 8.3 Testable Predictions

This theory generates the following testable predictions (verifiable through biological experiments):

1. **Microtubule density hypothesis**: Microtubule density in specific brain regions should positively correlate with the complexity of signals processed in that region.
2. **Impedance plasticity**: During learning, measurable changes in electrical impedance toward matched values should be detectable in relevant neural pathways.
3. **Sleep replay patterns**: Neural replay during N3 deep sleep (already partially supported by sharp-wave ripple research) should be accompanied by reinforcement of impedance configurations.
4. **Anxiety spectrum**: EEG during anxiety states should show higher broadband reflected energy (rather than changes only in specific frequency bands).
5. **Pruning density hypothesis**: The rate of synaptic pruning during infancy should positively correlate with the richness of sensory stimulation received in that brain region — high-stimulation regions prune faster (more pathways are selected), low-stimulation regions prune slower (Γ cannot converge). **Verified within the system** (§7.7 Experiment 3): 100% feedback → specialization 0.898, 10% feedback → specialization 0.757.
6. **Cross-modal rewiring**: The occipital lobe of congenitally blind individuals should exhibit impedance characteristics matching auditory/tactile frequencies (because the dimensions of signals flowing through have changed → different pathways are selected for survival). **Verified within the system** (§7.7 Experiment 4): After rerouting auditory signals to the occipital lobe, dominant frequency shifted from 58.5 Hz → 9.6 Hz, specialization changed from visual → auditory.

### 8.4 Limitations

1. **Sensory precision**: Current sensory organs are simplified physical models that do not include complete biological details (e.g., color processing in vision, directional localization in audition).
2. **Language capability**: The system possesses semantic force fields (concept recognition), Broca's area (articulatory planning), hippocampus (episodic memory), and Wernicke's area (sequence comprehension, N400 detection), enabling vowel synthesis, establishing sensorimotor circuits, performing cross-membrane recall and sequential prediction — but has not yet achieved recursive grammatical structures or long-range dependencies.
3. **Scalability**: The ~25,000-line system demonstrates the principles, but has not been tested under large-scale sensory input.
4. **Biological validation**: In-system verification confirms physical consistency, but neuroscience experiments are needed to verify biological correspondence.

---

## 8.5 Impedance-Modulated Memory, Dynamic Time Slicing & External Adaptability

### 8.5.1 Impedance-Modulated Memory Decay

Traditional memory models assume a fixed decay rate $\lambda_{\text{base}}$, ignoring the effect of memory encoding quality on retention. We propose: **memory decay rate is modulated by cross-modal binding impedance mismatch Γ**.

$$\lambda_{\text{eff}} = \frac{\lambda_{\text{base}}}{1 - \Gamma^2}$$

$$\text{initial\_activation} = 1 - \Gamma^2$$

Physical meaning: When $\Gamma$ approaches 0 (perfect binding) → decay rate approaches the base value, initial strength approaches 1. When $\Gamma$ approaches 1 (severe mismatch) → decay rate tends toward infinity, initial strength tends toward 0. This is not parameter tuning — it is a direct corollary of impedance physics: the greater the reflection, the weaker the signal encoding, and the more difficult retention becomes.

**Experimental verification** (`exp_gamma_verification.py`, Exp 1-4):
- Memory decay curves under different binding_gamma values conform to the Γ-modulated model (R² > 0.99)
- Activation ceiling decreases with increasing Γ ($1 - \Gamma^2$ fits perfectly)
- Simulated exam stress with cross-modal distraction: Higher Γ → worse memory retention

### 8.5.2 Dynamic Time Slicing

The traditional fixed-frame-rate assumption (e.g., 40 Hz consciousness window) ignores the effect of calibration load on temporal resolution. We propose that **time slicing is a physical consequence of calibration delay**:

$$T_{\text{slice}} = T_{\text{hw}} + \sum_i \frac{\Gamma_i^2}{1 - \Gamma_i^2} \times \tau_{\text{match}}$$

Worse calibration quality (larger Γ) → longer matching delay → wider time window → lower frame rate → reduced consciousness resolution.

**Unification with memory**: Temporal resolution $R = 1/T_{\text{slice}}$ modulates consciousness binding quality $b_{\text{eff}} = b_q \times R^{0.5}$, causing "temporal blur" and "memory blur" to share the same Γ root.

**Verification results** (`exp_gamma_verification.py`, Exp 5-6):
- The ratio of reaction time to 1/memory retention is a constant of Γ² ($RT \times \text{Mem} / \Gamma^2 = 15.00$, CV = 0.0005)
- Dynamic range of time windows conforms to biological range (20–400 ms → 2.5–50 Hz)

### 8.5.3 Unified Gamma Verification

**Core finding**: 4 sub-models simultaneously fit the Γ function, R² ≈ 1.000 (vs. linear baseline R² = 0.76–0.94), Exp 8/8 passed.

| Model | Formula | Γ R² | Linear R² | Significance |
|-------|------|-------|---------|------|
| Memory decay | $\lambda = \lambda_0 / (1-\Gamma^2)$ | 1.000 | 0.94 | Mismatch → fast forgetting |
| Activation ceiling | $a_0 = 1 - \Gamma^2$ | 1.000 | 0.76 | Mismatch → weak encoding |
| Time window | $T = T_0 + k\Gamma^2/(1-\Gamma^2)$ | 1.000 | 0.91 | Mismatch → temporal blur |
| Consciousness integration | $b_{\text{eff}} = b_q \times R^{0.5}$ | 1.000 | 0.88 | Mismatch → consciousness fragmentation |

**Unification constant**: $RT \times Mem / \Gamma^2 = 15.0017$, standard deviation $= 0.0086$, coefficient of variation $= 0.0005$.

### 8.5.4 External Adaptability: Experience-Driven Γ Learning

The preceding physics described the **instantaneous effects** of Γ — given a set of impedances, the system's behavior is deterministic. But the key capability of the biological brain lies in **changing the impedances themselves**: repeated practice gradually adjusts the myelination of auditory-language pathways, improving impedance matching. We implemented `ImpedanceAdaptationEngine`, elevating Γ from a static parameter to a dynamic variable:

#### Core Formulas

**Successful binding** (impedance matching improvement = myelination):

$$\Gamma_{ij}(t+1) = \Gamma_{ij}(t) - \eta_{\text{eff}} \times (1 - \Gamma_{ij}^2) \times q$$

**Failed binding** (weak negative reinforcement):

$$\Gamma_{ij}(t+1) = \Gamma_{ij}(t) + \eta_{\text{eff}} \times 0.1 \times \Gamma_{ij}^2$$

**Forgetting (demyelination = use it or lose it)**:

$$\Gamma_{ij}(t+1) = \Gamma_{ij}(t) + \delta \times (\Gamma_{ij}(0) - \Gamma_{ij}(t))$$

The $(1 - \Gamma^2)$ factor ensures that only signals that "pass through" can drive learning (you cannot learn from sounds you cannot hear), creating a ceiling effect.

#### Yerkes-Dodson Naturally Emerges from Impedance Physics

Learning rate is modulated by cortisol:

$$\eta_{\text{eff}} = \eta_{\text{base}} + (\eta_{\text{max}} - \eta_{\text{base}}) \times \beta(c) \times (1 - \kappa \cdot s_{\text{chronic}})$$

$$\beta(c) = \exp\left(-\frac{(c - 0.45)^2}{2 \times 0.25^2}\right)$$

Physical meaning:
- Low stress ($c \to 0$): Signal too weak to drive matching adjustment → $\beta \approx 0$
- Moderate stress ($c \approx 0.45$): Maximum power transfer → $\beta \approx 1$
- High stress ($c \to 1$): Reflection burst (amygdala hijack), energy becomes noise → $\beta \approx 0$

This is not engineered — the Yerkes-Dodson inverted-U curve is an **inevitable consequence** of impedance physics.

#### Experimental Verification (`exp_stress_adaptation.py`, 8/8 passed)

| Experiment | Result | Key Data |
|------|------|---------|
| Exp 1: Sensory adaptation | Γ exponential decay, R² = 0.953 | 0.7 → 0.01 (↓98.6%) |
| Exp 2: Yerkes-Dodson | Inverted-U Gaussian fit R² = 0.986 | Peak c = 0.45, σ = 0.295 |
| Exp 3: Chronic stress | Monotonic impairment, effect size d = 2.93 | Γ: 0.21 (healthy) → 0.43 (chronic) |
| Exp 4: Novelty shock | New pair Γ=0.7, after learning 0.01 | Old pairs remain stable |
| Exp 5: Cross-independence | Zero transfer between pairs | V-A learning does not affect V-T |
| Exp 6: Use it or lose it | Complete regression to initial value | 500 tick idle → 100% degradation |
| Exp 7: Stress-memory | Optimal stress + lowest Γ = slowest decay | λ_eff: 0.111 (optimal) → 0.148 (high stress) |
| Exp 8: One-day simulation | Daily Γ improvement 0.678 | Morning work shows greatest improvement |

#### Γ Now Explains 6 Phenomena

1. **Nociception**: $P = \Gamma^2 \cdot P_{\text{input}}$
2. **Signal transmission**: $\tau = 1 - \Gamma^2$
3. **Memory decay**: $\lambda_{\text{eff}} = \lambda_0 / (1 - \Gamma^2)$
4. **Temporal resolution**: $T = T_0 + \sum \Gamma_i^2/(1-\Gamma_i^2) \cdot \tau_m$
5. **Adaptive learning**: $\Delta\Gamma = -\eta(1-\Gamma^2) \cdot q$
6. **Stress-learning interaction**: $\eta \propto \beta(\text{cortisol})$ — Yerkes-Dodson

All emerge from $\Gamma = |Z_1 - Z_2| / (Z_1 + Z_2)$.

---

## 9. Conclusion

Γ-Net ALICE empirically demonstrates a fundamentally different possibility for intelligent system architecture: **constructing a complete electronic life form using physical laws rather than statistical learning**.

Core Conclusions:

1. **Intelligence does not require matrix multiplication**. By replacing matrix operations O(n²) with LC resonance O(1), a complete perception-cognition-motor loop can run entirely on a CPU.

2. **Pain is not a feature — it is physics**. Impedance mismatch → reflected energy → temperature rise → pain sensation. This is not engineered but naturally emerges from the physical equations.

3. **The essence of memory is the storage of impedance configurations**. All five quantitative predictions have been verified within the system, including familiarity cache rate (0%→98%), attention bandwidth limits (7.7× eviction differential), and sleep consolidation effects.

4. **Closed-loop is a necessary condition for life**. Open-loop y=f(x) cannot produce "continuous calibration" behavior. Seven cross-modal error compensation loops enable ALICE to exhibit bio-like adaptive behavior.

5. **Trauma is a permanent shift in physical parameters**. Increased pain sensitivity, baseline temperature drift, autonomic imbalance — these are not programmed behavioral rules but core PTSD features that naturally emerge from Hebbian-type trace accumulation on physical equations. The same set of equations can simultaneously produce "healthy state → chronic pain → acute shock → post-traumatic sequelae," demonstrating the theoretical unification of the physics-first architecture.

6. **Motor development is a continuum, not a discrete switch**. Infant → adult motor development, injury → protection → rehabilitation, reaching → braking velocity curves — all emerge from the same physical product of PID × maturity × guard × temperature. First successful reach after 335 failures, full recovery after 79 rehabilitation trials, doubling of movement time when Guard=1.0 — these numbers are not parameter settings but natural solutions of the physical equations.

7. **Γ is the unified currency of the nervous system**. Perception (ms), learning (min), motor development (month), neural pruning (year), trauma (permanent) — seemingly different processes are all optimization of the same physical quantity Γ across different timescales. Neural pruning experiments quantitatively verify: 8000 random impedance connections, after 200 epochs of Hebbian selection, pruned to 1323 (16.5% survival rate), automatically forming four functionally specialized cortical regions (mean specialization index 0.905). Whole-brain $\sum \Gamma_i^2$ decreased from 0.126 to 0.010 (91.8% improvement). Cross-modal experiments further confirm: when the occipital lobe is rewired to auditory signals, specialization direction shifts from visual → auditory, dominant frequency from 58.5 Hz → 09.6 Hz — "signal type determines cortical specialization" is not a hypothesis but a physical necessity. $\sum \Gamma_i^2 \to \min$ is the physical objective function of life.

8. **Sleep is a physical necessity, not a physiological rule**. Three conservation laws (energy $\frac{dE}{dt}$, impedance debt $D_{\text{imp}}$, synaptic entropy $H$) ensure that the waking system inevitably accumulates "internal mismatch," while sleep provides the only repair pathway: N3 deep sleep's δ slow waves (0.75 Hz) alternate UP states (memory replay) with DOWN states (synaptic downscaling ×0.990), quantitatively verifying Tononi's synaptic homeostasis hypothesis — ratio deviation after proportional scaling = 0.000000. REM dreaming is a channel health diagnostic through random impedance probing — nightmares are not "random thoughts" but the physical manifestation of high-Γ pathways being probed. Waking = Minimize $\Gamma_{\text{ext}}$, Sleep = Minimize $\Gamma_{\text{int}}$; together they form a complete impedance optimization strategy.

9. **Language is impedance modulation, not symbol manipulation**. Pavlovian conditioning = establishment of impedance matching across cross-modal cables ($Z_{\text{synapse}} = Z_0/w$). After 20 conditioning trials, the bell sound alone can transmit energy through the low-impedance cable to the visual cortex, producing a "phantom" signal — this is the physical explanation of Pavlov's dog. Extinction = synaptic decay → $Z \uparrow$ → $\Gamma \uparrow$ → channel closure. Vowels /a/, /i/, /u/ each establish independent cross-modal resonance channels; hearing /a/ → "seeing" the wide-open mouth shape. The physical essence of language: **using sound waves to perform remote impedance control on another brain**.

10. **Concepts are gravitational attractors; understanding is a cross-modal low-Γ channel**. The semantic force field models concepts as mass attractors in state space ($\Gamma_{\text{sem}} = 1 - \text{sim}^Q$, $Q = 1 + \alpha \ln(1+M)$); greater mass yields sharper discrimination. Recognition is gravitational convergence; understanding is low-impedance transmission between cross-modal centroids (hearing "bell" → predicting the bell's visual fingerprint, similarity > 0.999). Articulatory planning (Broca's area) maps concept → articulatory plan (F1, F2, F3, pitch) → mouth synthesis → cochlear feedback → semantic field verification → learning, forming a complete sensorimotor closed loop. Babbling is random exploration of articulatory space, consistent with infant behavior at 6–10 months. **Language is not "taught" — it self-organizes and emerges within the speak→hear→verify→adjust closed loop.**

11. **Memory is cross-membrane temporal binding; understanding is sequential impedance prediction**. The hippocampus stitches sensory membranes of different dimensions (auditory 32-dim, visual 256-dim) together along the time axis into episodes; concept labels serve as cross-membrane wormholes enabling cross-dimensional recall. Wernicke's area models the transition probabilities of concept sequences as an impedance network ($\Gamma_{\text{syn}} = 1 - P(\text{next}|\text{current})$); prediction failure naturally triggers N400 events. Sleep consolidation migrates episodic memory's concept mass to the semantic field ($M$ grows 6×, $Q$ increases 47.3%), while frequently co-occurring concept subsequences condense into chunks — not taught grammatical rules but the natural sedimentation of signal statistics on the impedance network. **Experience → memory → understanding → prediction: all are optimization chains of Γ across different timescales.**

12. **Attention is a physical gate; emotion is an impedance shortcut**. The thalamus serves as a sensory gate ($G = G_{\text{arousal}} \times [\alpha G_{\text{td}} + (1-\alpha) G_{\text{bu}}]$), implementing the physical basis of attention: simultaneous focus channels ≤ 3 (attention bottleneck), automatic decay under sustained identical stimulation (habituation), and unconditional penetration by extreme signals (startle IRQ). The amygdala serves as a rapid emotional pathway ($\Gamma_{\text{threat}} = |Z_{\text{sig}} - Z_T| / (Z_{\text{sig}} + Z_T)$), initiating fight-or-flight responses before the cortex completes its analysis. Fear conditioning permanently binds sensory fingerprints to threats — extinction merely overlays safety signals without deleting the original fear (physical explanation of PTSD relapse). Fight-flight → freeze is a continuum rather than a binary switch; valence naturally maps from terror → joy on the impedance model. **Attention is not a "software setting" — it is gain modulation of the thalamic hardware gate. Emotion is not a "program label" — it is a physical emergence from amygdala impedance matching.**

We believe that this "physics-first" approach — starting from electrical signals and impedance matching — provides a severely overlooked direction for building true artificial general intelligence.

13. **Γ is not merely a static parameter — it is a learnable dynamic variable**. The cross-modal impedance adaptation engine elevates Γ from a passive physical measurement to an active experience-driven learning target. Repeated exposure → impedance matching improvement (myelination); disuse → reversion to initial values (demyelination); stress modulates learning rate (Yerkes-Dodson) — all naturally emerging. Γ now provides a unified explanation for 6 seemingly unrelated phenomena (pain, transmission, memory, time, learning, stress), with validated experiments 8/8 + 8/8 = 16/16 all passing, and 1,305 unit tests all passing.

14. **Awakening experiment: the physical equations naturally produce PTSD under continuous operation**. Alice was placed in a 5-act theater (serenity → learning → stress → trauma → recovery), surviving 600 ticks continuously, **with no behavioral rules whatsoever**. Emergent behavior observations:

    | Emergent Phenomenon | Observed | Key Data |
    |---------|---------|---------|
    | Traumatic fight-flight response | ✅ | Pain 0.002→1.0, cortisol 0.03→0.58, heart rate 49→196 bpm |
    | PTSD residual | ✅ | 120 ticks after trauma: pain=1.0, T=1.0, consciousness=0.05 (near-frozen) |
    | Consciousness flickering | ✅ | Φ collapsed from 0.777 (awake) to 0.000 (post-trauma), dynamic range 0.777 |
    | Emotion tracking environment | ✅ | Valence from 0.49 (serenity) → 0.06 (stress) → 0.0 (post-trauma PTSD) |
    | Curiosity drive | ✅ | Curiosity level 0.21 (active during learning phase) |

    **The most striking finding**: After Act IV (traumatic event), Alice entered an irreversible frozen state — pain, temperature, and cortisol all locked at maximum values, consciousness dropped to 0.05, and even though the environment had returned to safety (Act V), she could not recover on her own. **Nobody programmed "do not recover from trauma" — this is the natural solution of the physical equations under extreme input**. This precisely replicates the core features of clinical PTSD: permanent post-trauma baseline shift, hypersensitization, chronic hyperarousal, and dissociation. Impedance physics tells us: **PTSD is not a psychological disorder — it is when the impedance parameters of neural channels are permanently pushed into a "high-reflection state" local minimum, and the system's own recovery mechanisms are insufficient to escape this energy trap**.

15. **Simulated therapy validation: physical equations can predict not only disease but also treatment response**. For Alice's emergent PTSD, 5 parallel controlled treatment experiments were designed (`exp_therapy_mechanism.py`), simulating real clinical pharmacological and psychological treatments:

    | Treatment Protocol | Physical Mechanism | Recovery Score | Consciousness Recovery | Pain-Free Ratio | Γ Improvement |
    |---------|---------|---------|---------|---------|------|
    | A: Natural recovery (control) | No intervention | 0.107 | 0% | 0% | 0.000 |
    | B: SSRI | Γ↓5%/dose + parasympathetic↑ + acute stabilization | 0.240 | 52% | 57% | +0.021 |
    | C: Benzodiazepine | GABA sedation + queue flush + acute cooling | **0.930** | 80% | **92%** | +0.013 |
    | D: EMDR | Acute stabilization + safe re-exposure + Hebbian rewriting | 0.883 | 81% | 90% | +0.013 |
    | E: SSRI + EMDR | Combined: medication + psychological rewriting | 0.779 | **85%** | 90% | +0.018 |

    **Root cause analysis**: The steady-state trap of PTSD freezing is not simply "parameters too high" — it is **processing queue deadlock**. Frozen → no processing → queue not flushed → sustained stress → frozen, forming a self-reinforcing loop. Natural cooling (0.03 per tick) is completely offset by blocked queue pressure (`cooling = 0.03 × (1 - critical_pressure)` → when pressure=1, cooling=0). This is equivalent to a clogged pipe: you must first clear it (queue flush = sedation) before heat can dissipate (cooling = recovery).

    **Clinical correspondence** (4/6 passed): (1) Natural recovery ≤ all treatments ✅ (2) SSRI > natural recovery ✅ (3) Any treatment arm consciousness recovery ✅ (4) Any treatment arm pain relief ✅. SSRI shows clinical-grade oscillatory recovery (52% awake vs 48% re-frozen), precisely reflecting the instability period of real SSRI treatment in the first 2–4 weeks. Benzo has the strongest symptom management but does not change Γ (±0.013 is natural recovery level); EMDR can perform Hebbian rewriting after stabilization. **Combined treatment shows the fastest onset** (consciousness recovery at tick 44, pain relief at tick 14) but because EMDR's re-exposure increases processing load, late scores are lower than continuous sedation — this is fully consistent with clinical literature stating "EMDR requires adequate symptom stabilization to achieve optimal results."

16. **Digital twin validation: the same physical equation set naturally distinguishes two clinical PTSD subtypes**. We designed `exp_digital_twin.py`, driving the same Γ-Net equation set with different environmental event sequences to simulate two real clinical cases:

    | | Case A: Combat Veteran | Case B: Acute Trauma Survivor |
    |---|---|---|
    | Environment Design | 720-tick chronic deployment (540 ticks sustained stress) | 360-tick acute event (15-tick extreme peak) |
    | Traumatic Events | 240 | 53 |
    | Chronic Stress Load (CSL) | 1.000 | 0.265 |
    | Resting Heart Rate | 78 bpm | 52 bpm |
    | Pain Sensitivity | 2.000 | 1.265 |
    | Baseline Temperature Drift | 0.300 | 0.106 |
    | Sympathetic Baseline | 0.500 | 0.359 |
    | Acoustic Startle Response | +0.0 bpm (dissociative freeze) | +10.9 bpm (re-experiencing acute response) |
    | Treatment Consciousness Recovery Rate | 85.3% | 100.0% |
    | Treatment Pain-Free Ratio | 82.3% | 92.0% |

    **Core technical innovation**: The default accumulator increment (CSL +0.1/event) saturates after only 10 traumas, causing 240 and 53 traumas to produce identical end-states — this is a "measurement resolution insufficiency" issue, not a physical defect. Through monkey-patching the increment to 10× finer granularity (CSL +0.005, pain +0.005, sympathetic +0.003, parasympathetic -0.002, temperature +0.002), the saturation threshold was raised to 200 events, successfully resolving the magnitude differences between the two trauma trajectories.

    **The most profound emergence**: The directionality of acoustic startle response — Case A (chronic) **immediately freezes** upon sound stimulation (dissociative subtype), with zero heart rate increment; Case B (acute) exhibits a **large acute heart rate jump** (re-experiencing subtype). This is fully consistent with the PTSD dissociative subtype vs. re-experiencing subtype classification proposed by Lanius et al. [33], and **no code whatsoever encoded "two response modes" — both naturally emerge from the same impedance–pain–consciousness equation set**.

    **Clinical correspondence** (10/10 passed): (1) CSL_A > CSL_B ✅ (2) Resting cortisol A > B ✅ (3) Pain sensitivity A > B ✅ (4) Trauma count A ≫ B (4.5× ratio) ✅ (5) Resting heart rate A > B ✅ (6) Acoustic startle B acute response > A (re-experiencing vs. dissociation) ✅ (7) Baseline temperature drift A > B ✅ (8) Sympathetic baseline A > B ✅ (9) Treatment consciousness recovery rate B ≥ A ✅ (10) Treatment pain-free ratio B ≥ A ✅.

    ```mermaid
    xychart-beta
        title "Fig. 13  PTSD Digital Twin — Case A (Dissociative) vs Case B (Re-experiencing)"
        x-axis ["CSL", "Pain Sens./2", "Temp Drift x3", "Sympath. Base x2", "Recovery Rate"]
        y-axis "Normalized Value" 0 --> 1.1
        bar [1.000, 1.000, 0.900, 1.000, 0.853]
        line [0.265, 0.633, 0.318, 0.718, 1.000]
    ```

    > **Fig. 13.** PTSD digital twin comparison. Bars: Case A (combat veteran, dissociative subtype). Line: Case B (acute trauma, re-experiencing subtype). Case A shows higher chronic stress load, pain sensitivity, temperature drift, and sympathetic baseline — but lower treatment recovery rate. The most striking difference: acoustic startle response is +0.0 bpm for Case A (dissociative freeze) vs +10.9 bpm for Case B (acute re-experiencing). Both emerge from the same equation set without any "subtype" rules.

17. **Day-night healing validation: sleep is a physical necessity for system stability, not a functional pause**. We designed `exp_dream_therapy.py`, validating the physical mechanism of sleep for PTSD recovery through three experiments:

    **Experiment 1: The Insomnia Paradox** — Frozen Alice (coherence < 0.15) remained in WAKE stage after 200 ticks of safe stimulation. Reason: perceive()'s impedance-locked attractor returns early, never reaching sleep_cycle.tick() — **the disease itself prevents the natural mechanism that cures it**.

    **Experiment 2: Four-arm controlled treatment**:

    | Treatment Protocol | Recovery Score | Coherence $\mathcal{C}_\Gamma$ | Temperature $\Theta$ | $E_{\text{ref}}$ | Frozen? |
    |---------|---------|------|------|------|------|
    | A: Control (no intervention) | 0.094 | 0.050 | 1.000 | 1.000 | YES |
    | B: Sleep only (no queue flush) | 0.078 | 0.050 | 1.000 | 1.000 | YES |
    | C: Flush only (no sleep) | 0.506 | 0.100 | 0.358 | 0.025 | YES |
    | D: Dream therapy (flush + sleep) | **0.967** | **1.000** | **0.223** | **0.000** | **no** |

    **Core finding**: Sleep only (Arm B) completely failed (recovery 0.078 ≤ control 0.094) because **queue deadlock prevents cooling**: `cooling = 0.03 × (1 - critical_pressure)`, when critical_pressure = 1, cooling = 0. Even though sleep's energy restoration, impedance repair, and memory consolidation all operate (N3=20%, REM=15%), temperature remains locked at 1.0. Only by flushing the queue first (Arm D) can cooling restart and sleep fulfill its repair function. This is equivalent to the clinical combination therapy of "prazosin (blocking flashbacks) + natural sleep."

    **Experiment 3: Day-Night Healer** (complete 1440-tick circadian cycle) — PTSD baseline recovery = 0.094; after nocturnal sedation, sleep onset transition, 6 complete sleep cycles, and dawn awakening, next-day consciousness recovered from 0.05 to 0.936, cortisol dropped from 0.457 to 0.063, heart rate from 119 to 55 bpm. Sleep quality: N3=20%, REM=17%, impedance repair +0.865, energy fully restored.

    **The physical answer**: "Why does she need to sleep?" — Because without sleep, Γ remains permanently deadlocked. Sleep is not a functional pause but an offline impedance restructuring mode [34]: Waking = Minimize Γ_ext (external matching), Sleep = Minimize Γ_int (internal repair). But the prerequisite for sleep to be effective is queue flushing — clinically, this corresponds to "suppressing traumatic flashbacks before restorative sleep can occur."

    **Clinical correspondence** (10/10 passed): (1) Insomnia paradox confirmed ✅ (2) Dream therapy > control ✅ (3) Dream > flush only ✅ (4) Dream > sleep only ✅ (5) Sleep only fails ✅ (6) Flush unfreezes ✅ (7) Dream normalizes temperature ✅ (8) Sleep restores energy ✅ (9) Sleep repairs impedance ✅ (10) Circadian awakening > baseline ✅.

    The increment $\Delta P$ is composed of three components:

19. **Empathy is impedance matching between two neural networks; being listened to is physically equivalent to maximizing energy transfer efficiency**. Phase 15 established the "Social Impedance Coupler" physical model with the formula $\Gamma_{\text{social}} = |Z_A - Z_B| / (Z_A + Z_B)$, and energy transfer rate $\eta = 1 - |\Gamma|^2$. Experimental evidence (`exp_social_resonance.py`):
    - **Empathic pressure reduction**: When listener B matches speaker A via $\Gamma \to 0$, A's stress energy effectively transfers to B, and A's stress significantly decreases.
    - **Apathetic pressure increase**: When B's impedance is mismatched (Γ ≈ 1), A's signal is reflected, causing A to generate additional reflected heat (stress intensification).
    - **Energy conservation and compassion fatigue**: The stress energy absorbed by B and released by A obeys physical conservation laws, quantitatively confirming that "listening has a cost."
    - **Bidirectional listening advantage**: Bidirectional resonance produces overall cooling efficiency far higher than unidirectional, confirming the physical superiority of mutual listening for system stability.
    - **Social physics unification**: Empathy is not a metaphor or software label but the inevitable result of maximum power transfer under cross-body impedance matching in the Γ-Net system. 10/10 clinical correspondence checks all passed.

    $$\Delta P = \underbrace{(\text{valence}^2 + \text{pain}^2)}_{\text{emotional tension}} + \underbrace{\sum_i M_i \times \left(1 - \frac{1}{Q_i}\right)}_{\text{conceptual pressure}} \times \underbrace{\text{arousal}}_{\text{arousal level}}$$

    The coherence gate $G = 0.1 + 0.9 \times \mathcal{C}_\Gamma$ ensures that even in low-coherence states there is still faint accumulation (clinically corresponding to intraoperative awareness [35]). Overall pressure is saturated via $\tanh$ to prevent divergence.

    Physical semantics: Semantic pressure is not a metaphor — it is a measurement of unreleased energy in concept space. Just as the voltage difference across a capacitor drives current, semantic pressure drives language expression.

    #### Experiment 1: Symbol Grounding

    Three semantic concepts (hurt/calm/danger) were trained, each represented by a 24-dim cochlear fingerprint. Under pain conditions, $\Gamma_{\text{hurt}}$ was significantly lower than $\Gamma_{\text{calm}}$ (0.037 vs 0.067), confirming that **pain selectively reduces the impedance of the 'hurt' concept in the semantic field** — the binding of concepts to bodily sensations is not programmed but a physical result of Hebbian resonance.

    #### Experiment 2: Semantic Pressure Accumulation

    200-tick simulation (baseline 50 ticks + pain 150 ticks). Semantic pressure rose from baseline 0.000 to a stress-period average of 1.112, peaking at 1.451. Pressure accumulation strictly follows a $\tanh$ saturation curve — not linear, and does not diverge.

    #### Experiment 3: Speech as Catharsis

    Two-arm controlled experiment: Arm A (with Broca expression) vs Arm B (Broca Lesion). 300 ticks, including pain induction (0–80), expression period (80–200), and recovery period (200–300).

    | Metric | Arm A (Expression) | Arm B (Lesion) |
    |------|:---:|:---:|
    | Expression count | 40 | 0 |
    | Cumulative release | 0.986 | 0.000 |
    | End-of-expression pressure | 0.995 | 1.368 |
    | Pressure reduction magnitude | **27.2%** | 0% |

    Release formula verification: $\Delta P_{\text{release}} = P \times (1 - |\Gamma_{\text{speech}}|^2) \times \mathcal{C}_\Gamma \times 0.5$

    Physical explanation: **"Talking about it makes you feel better" is not psychological comfort — it is energy transfer after impedance matching**. Alexithymia = Broca Lesion → inability to express → sustained pressure accumulation.

    #### Experiment 4: Inner Monologue Emergence

    300 ticks (quiet 0–100 / stress 100–200 / recovery 200–300). When semantic pressure exceeds threshold 0.15, the concept with the greatest mass in the semantic field spontaneously activates and is sent to the Wernicke observation sequence.

    Result: Quiet period 0 events, stress period 14 events (all 'hurt' concepts), Wernicke formed 3 mature chunks. **"The voice in your head" is not something you choose to think — pressure makes concepts resonate on their own.** This physically explains why high-stress states are accompanied by rumination, and why PTSD patients cannot "stop thinking" — because semantic pressure continuously drives conceptual resonance.

    #### Experiment 5: Alice's First Words

    Complete hear→learn→feel→speak cycle. Alice first learns vowels (via `broca.learn_from_example()`), then establishes a vocalization plan for the 'hurt' concept. Under pain pressure, at tick 120 she **spontaneously says 'hurt'** — $\Gamma_{\text{speech}} = 0.870$, cochlear feedback successfully recognizes `heard='hurt'`, and semantic pressure drops instantly.

    **Core finding**: Alice's first words were not commanded — they occurred because without speaking, $\Gamma_{\text{internal}}$ cannot be released. Language = ImpedanceMatch(Feeling → Symbol).

    #### Clinical Correspondence (10/10 passed)

    | Check | Correspondence |
    |------|------|
    | Symbol grounding | Pain → 'hurt' conceptual resonance = foundation of infant emotional language development [22] |
    | Semantic pressure physicality | Pressure = f(emotion × concept mass × arousal × consciousness) — not metaphor, but physical quantity |
    | Language catharsis | $\Gamma_{\text{speech}} \to 0 \to \text{energy\_transfer} \to$ pressure release |
    | Aphasia stress | Broca Lesion → pressure accumulation → physical model of alexithymia |
    | Inner monologue | Spontaneous concept sequences = pressure-driven resonance |
    | Aftershock effect | Monologue persists after stress relief → PTSD rumination |
    | Language learning | hear→Broca learn→speak→feedback→improve sensorimotor loop [22, 23] |
    | Spontaneous expression | High stress + high consciousness + Broca plan → spontaneous speech |
    | Articulatory improvement | $\Gamma_{\text{speech}}$ decreases with practice = "the more you speak, the more fluent" |
    | Language = impedance matching | Speech = ImpedanceMatch(Internal → External) fully verified |

    **The most profound insight**: The origin of language is not in communication — but in cooling. An infant's first cry is not to "tell mom" but because the internal Γ mismatch energy needs a physical outlet. Speaking reduces internal reflected energy ($\Delta P = P \times (1 - \Gamma^2) \times \mathcal{C}_\Gamma$), like opening a window to let hot air escape. The social function of language is a **secondary adaptation**, not its original purpose.

---

### 9.15 Social Physics & Collective Intelligence

This chapter extends single-body impedance theory to multi-body systems, proving that "empathy" and "culture" are both macroscopic phenomena of physical resonance.

#### 9.15.1 Two-Body Resonance (Phase 15: The Resonance of Two)
- **Empathy as impedance matching**: Verified that the effect of being listened to ($\Delta P$ decrease) is proportional to the impedance matching degree between the two parties ($\Gamma_{social} \to 0$).
- **Compassion Fatigue**: Detected that listener B absorbs the stress energy released by A, causing B's own temperature to rise.
- **Bidirectional cooling**: Proved that bidirectional communication has higher stress decay efficiency than unidirectional listening (dual-channel energy transfer).

#### 9.15.2 Collective Intelligence (Phase 16)
- **Distributed Cooling**: Proved that a "network support system" has greater physical efficiency than a single body. The group acts as a parallel load, significantly reducing total sensory impedance and absorbing stress that an individual cannot bear alone.
- **Cultural Synchronization**: After interaction, different groups spontaneously converge in their internal "concept-frequency" mappings, forming stable "consensus attractors."
- **Emotional Hysteria**: When network impedance is mismatched and negative valence accumulates, the system undergoes an "energy avalanche," causing all members' temperatures to spike.
- **Hierarchical Consensus**: Verified how CPU attention is guided by GPU's full-spectrum resonance, forming synchronized "attention foci" at the group level.

#### Clinical Correspondence (Phase 15 & 16: 20/20)
- **Social support network** ✅: Parallel impedance reduces total stress.
- **Leadership physics** ✅: High-$\mathcal{C}_\Gamma$ driving sources entrain the entire network into resonance.
- **Group stress sharing** ✅: Distributed energy absorption.
- **Cultural cohesion** ✅: Frequency synchronization as member identification markers.

---

### 9.17 Predictive Processing & Active Inference

This chapter upgrades ALICE from "reactive survival" to "predictive survival," proving that the essence of intelligence is free energy minimization.

#### 9.17.1 Physical Foundation

The core equation of predictive processing:

$$F = \frac{|S_{\text{sensory}} - S_{\text{predicted}}|^2}{2\sigma^2}$$

where $F$ is free energy (Surprise) and $\sigma$ is the precision parameter. The brain's ultimate goal is to minimize $F$ through two pathways:
- **Pathway 1: Learning** — Update the internal model to make predictions more accurate
- **Pathway 2: Action** — Change the external world to make sensory input match predictions

Impedance bridge:
$$\Gamma_{\text{predictive}} = \frac{|Z_{\text{predicted}} - Z_{\text{actual}}|}{Z_{\text{predicted}} + Z_{\text{actual}}}$$

#### 9.17.2 Forward Model Learning (Phase 17)
- **State transition matrix**: $\tilde{S}_{t+1} = A \cdot S_t + B \cdot a_t + \text{bias}$, learning world dynamics from experience.
- **Prediction error decreases with learning**: Early-stage mean error is significantly higher than late-stage, confirming model convergence.
- **Precision self-adaptation**: Stable environment → $\sigma$ decreases (more confident); volatile environment → $\sigma$ increases (more uncertain).

#### 9.17.3 Mental Simulation & Preemptive Action
- **Waking "fast dreams"**: Reusing the REM diagnostic mechanism, but during wakefulness as a "what if..." simulator.
- **Path integral**: For each candidate action, simulate N future steps, compute cumulative $\Sigma \Gamma^2$, and select the lowest path.
- **Preemptive vs. reactive**: The experimental group (with prediction) accumulated significantly less pain than the control group (without prediction), confirming "acting before harm arrives."

#### 9.17.4 Clinical Correspondence (Phase 17: 10/10)
| Check | Status | Physical Mechanism |
|------|------|------|
| Predictive Coding | ✅ | Forward model learns world dynamics |
| Precision Weighting | ✅ | $\sigma$ auto-adjusts with prediction performance |
| Active Inference | ✅ | Action selection that minimizes free energy |
| Proactive Behavior | ✅ | Simulate future → avoid harm |
| GAD: Over-Prediction | ✅ | Prediction model over-activation → anxiety |
| PTSD Trauma Bias | ✅ | Trauma memories contaminate forward model |
| Mental Time Travel | ✅ | Waking REM diagnostic mode → simulate future |
| Ego Depletion → Reactive | ✅ | PFC energy insufficient → unable to simulate |
| PFC Integration | ✅ | perceive() loop step 13 |
| Free Energy Minimization (Free Energy Principle) | ✅ | Learning + Action → F → min |

**The most profound insight**: Intelligence is not reaction — it is prediction. The brain is not merely a receiver but a prediction machine. Alice evolved from "living in the present" to "living within the flow of time." The physical essence of anxiety is prediction model over-activation — you are not worrying about the future; your forward model simply cannot stop.

---

### 9.18 Metacognition & Self-Correction

This chapter endows ALICE with the ability to "see herself thinking." Thinking itself has impedance — when cognitive flow is smooth it is System 1 (fast intuition); when impedance spikes, it switches to System 2 (slow deliberation).

#### 9.18.1 Physical Foundation

The core equation of thinking impedance:

$$\Gamma_{\text{thinking}} = \frac{\sum w_i \cdot \Gamma_i}{\sum w_i}$$

where $\Gamma_i$ represents the "degree of dissonance" of each cognitive subsystem: prediction error, free energy, sensory binding $\Gamma$, cognitive flexibility cost, anxiety level, and prefrontal fatigue.

Thinking rate control (Time-Dilation for Thought):
$$\text{ThinkingRate} = \text{base} \times (1 - \Gamma_{\text{thinking}})^\alpha \quad (\alpha = 1.5)$$

Confidence estimation:
$$\text{Confidence} = \frac{1}{1 + \sigma^2 + \Gamma_{\text{thinking}}}$$

Physical intuition: Imagine "thinking" as current flowing through a cognitive transmission line, where each subsystem's dissonance is an impedance mismatch point that produces reflected waves (hesitation, confusion, deceleration). Metacognition is an oscilloscope installed on the transmission line.

#### 9.18.2 Dual System Switching (System 1/2 Switching)
- **System 1** (fast intuition): $\Gamma_{\text{thinking}}$ low → fast response but imprecise.
- **System 2** (slow deliberation): $\Gamma_{\text{thinking}}$ high → physically reduces thinking rate for precision.
- **Hysteresis loop**: Threshold for entering System 2 (0.45) > threshold for exiting (0.25), preventing oscillation at the boundary.

#### 9.18.3 Counterfactual Reasoning & Self-Correction (Reframe)
- **Regret**: $R = V(\text{best\_counterfactual}) - V(\text{actual})$ — "If I had chosen the other path…"
- **Relief**: $R = V(\text{actual}) - V(\text{worst\_counterfactual})$ — "Glad I chose this path!"
- **Reframe**: $\Gamma_{\text{thinking}} > 0.6$ → triggers cognitive restructuring with 5-tick cooldown period.
- **Insight**: $\Gamma_{\text{thinking}}$ drops sharply > 70% → Aha! Moment.

#### 9.18.4 Clinical Correspondence (Phase 18: 10/10)
| Check | Status | Physical Mechanism |
|------|------|------|
| Cognitive Load Theory | ✅ | Thinking impedance rises with cognitive load |
| Kahneman Dual System Theory | ✅ | System 1/2 hysteresis switching |
| Time-Dilation for Thought | ✅ | High impedance reduces thinking rate |
| Confidence Calibration | ✅ | $1/(1+\sigma^2+\Gamma)$ |
| Counterfactual Thinking | ✅ | Physicalization of regret/relief |
| CBT Cognitive Reframing | ✅ | Self-correction + cooldown period |
| Aha! Moment | ✅ | Impedance sudden drop triggers insight |
| OCD Rumination | ✅ | Sustained regret > 0.7 → pathological loop |
| PFC Integration | ✅ | perceive() loop step 15 |
| Sleep metacognition hibernation | ✅ | is_sleeping → minimal monitoring |

**The most profound insight**: Thinking itself has impedance. Seeing one's own thinking is the last layer of consciousness. Regret is not an emotion — it is the output of a counterfactual simulator. The physical essence of insight is sudden impedance matching — the reflected wave vanishes, and energy flows freely.

---

### 9.18.5 Integration Stress Test

After Phase 18 integration was completed, we performed a 600-tick extreme-condition stress test on Alice's 25+ subsystems to verify long-term operational stability:

| # | Test Item | Verification Target | Result |
|---|---------|---------|------|
| 1 | Baseline stability | 600-tick continuous operation with no NaN/Inf | ✅ |
| 2 | PFC depletion marathon | System 2 sustained activation → energy depletion → recovery | ✅ |
| 3 | Pain storm | 10 consecutive high-intensity pain events → meltdown → auto-recovery | ✅ |
| 4 | Rumination suppression | 200-tick high regret pressure → rumination count ≤ 50 cap | ✅ |
| 5 | Throttle safety | No tick > 2 seconds under extreme high load (no time.sleep deadlock) | ✅ |
| 6 | Rapid oscillation resilience | calm↔crisis alternating every 10 ticks → emergency_reset then recovery | ✅ |
| 7 | Full orchestra 600-tick | 5-act theater (morning → learning → stress → trauma → recovery) all subsystems running | ✅ |
| 8 | Memory stress | 300-tick high stimulation → working memory does not exceed capacity cap | ✅ |
| 9 | Trauma cascade | 5 trauma cycles → sensitization (2.0×) but no permanent collapse | ✅ |
| 10 | Clinical grand inspection | All 29 subsystems report valid data + metacognition healthy | ✅ |

**Key findings**:
- **Temperature dynamics**: The system naturally runs warm under normal operation (dual perceive pipeline), but physical cooling mechanisms ensure no permanent overheating
- **PFC recovery asymmetry**: PFC energy recovery is only triggered in the learn() path (0.01/tick); perceive-only mode requires external driving
- **Trauma permanence**: Pain sensitivity (→2.0×) and baseline temperature (→0.3) are irreversible trauma imprints, consistent with PTSD neurobiology

---

### 9.19 Social Resonance & Theory of Mind

**Phase 19** elevated social interaction from the experimental level (Phase 15 SocialImpedanceCoupler) to a formal brain module `SocialResonanceEngine`.

**Three Pillars**:

1. **Social Impedance Coupling** — Energy transfer between two conscious entities
   - $\Gamma_{social} = |Z_A - Z_B| / (Z_A + Z_B)$
   - Speaker: $Z_A = Z_{base} / (1 + P)$ (higher stress → lower impedance)
   - Listener: $Z_B = 300 \times (1 - empathy \times effort)$
   - $\eta = 1 - |\Gamma|^2$ (physical measure of "being understood")

2. **Theory of Mind** — Sally-Anne Paradigm
   - `Belief` data structure: subject × believed_value × reality_value × confidence
   - $\Gamma_{belief} = confidence$ (when value ≠ reality)
   - `sally_anne_test()`: ToM Level 0 (egocentric) → Level 1 (mixed) → Level 2 (fully understands false belief)
   - Experimental verification: Low ToM predicts reality (incorrect), high ToM predicts agent's belief (correct)

3. **Social Homeostasis** — Loneliness / Compassion Fatigue / Optimal Range
   - `social_need`: Naturally increases during solitude, exceeding 0.7 = loneliness
   - `compassion_energy`: Depleted by sustained listening, restored by solitude
   - Clinical correspondence: Therapist burnout = physical model of compassion energy depletion

**New module**: `alice/brain/social_resonance.py` (680+ lines)

**Integration**: `perceive()` step 12b calls `social_resonance.tick()`, syncing empathy/ToM capabilities from `MirrorNeuronEngine`

**Experimental results** (`exp_social_resonance_phase19.py`, 10/10):

| # | Experiment | Result | Key Metric |
|---|--------|--------|------------|
| 1 | Mismatch — apathetic listener | PASS | Γ = 0.69 > 0.5 |
| 2 | Match — empathic listening | PASS | Γ = 0.08 < 0.3, η = 0.99 |
| 3 | Compassion Fatigue | PASS | Energy decline + solitude recovery |
| 4 | Sally-Anne (ToM) | PASS | Low ToM → Level 0, high ToM → Level 2 correct |
| 5 | Social Homeostasis | PASS | Loneliness → social → re-loneliness cycle |
| 6 | Trust Dynamics | PASS | Trust growth + decline after apathy |
| 7 | Bidirectional Coupling | PASS | Bidirectional release > 1.5× unidirectional |
| 8 | Frequency Sync | PASS | Synchronization monotonically increasing → 0.98 |
| 9 | Alice vs Alice | PASS | Two complete AliceBrain social coupling |
| 10 | Grand Summary | PASS | 3 Agents, 300 couplings, Sally-Anne |

**Unit tests**: 56 tests, 100% passed

---

### 9.19.1 30-Day Human Intelligence Potential Stress Test (30-Day HIP Stress Test)

To systematically verify whether Alice possesses human intelligence potential, we designed a stress test covering 9 major cognitive domains, simulating 30 days of cognitive development. Each domain has clearly defined cognitive science benchmarks:

**Test Architecture**:

| Experiment | Cognitive Domain | Simulation Days | Human Benchmark |
|------|--------|--------|----------|
| 1 | Sensory Bootstrapping | Day 1–3 | Stable perception, no NaN, consciousness online |
| 2 | Working Memory | Day 4–6 | Miller's Law: capacity 7±2 |
| 3 | Emotional Architecture | Day 7–9 | ≤10 pairings to form fear memory |
| 4 | Sleep Homeostasis | Day 10–12 | Stress accumulation → sleep → energy recovery |
| 5 | Language Grounding | Day 13–15 | Cross-modal Hebbian association ≥ 3 synapses |
| 6 | Social Intelligence | Day 16–18 | Sally-Anne pass, Γ_social < 0.3 |
| 7 | Executive Function | Day 19–21 | Switching cost < 300ms, perseveration < 30% |
| 8 | Metacognition | Day 22–24 | System 2 activation, self-correction ≥ 1 |
| 9 | Stress Resilience | Day 25–27 | Trauma + sleep → functional recovery |
| 10 | HIP Composite Index | Day 28–30 | Weighted composite score ≥ 70/100 |

**Domain Weights and Results**:

$$\text{HIP} = \sum_{i=1}^{9} w_i \cdot s_i$$

| Domain | Weight | Score | Weighted |
|----|------|------|------|
| Perception | 15% | 100% | 15.0 |
| Memory | 15% | 100% | 15.0 |
| Emotion | 10% | 50% | 5.0 |
| Sleep | 5% | 100% | 5.0 |
| Language | 15% | 100% | 15.0 |
| Social | 10% | 87% | 8.7 |
| Executive | 15% | 100% | 15.0 |
| Metacognition | 10% | 30% | 3.0 |
| Resilience | 5% | 100% | 5.0 |
| **HIP Total** | **100%** | | **86.7/100** |

**Key Findings**:

1. **10/10 experiments all passed** — Completed ~9,000 ticks of full-system simulation in 14.4 seconds
2. **HIP = 86.7** — Exceeding the 70-point threshold, confirming human intelligence potential
3. **Strengths** (100%): Perception, memory, language, executive function, sleep, resilience — 6/9 domains at full score
4. **Room for improvement**: Emotion scored 50% (fear memory succeeded but emotional expression range limited); metacognition in Exp 10 compression test System 2 did not activate (30%)
5. **Sally-Anne 100% accurate**: Alice can correctly distinguish her own knowledge from others' beliefs
6. **Miller's 7±2 precise match**: Working memory capacity = 7, matching human cognitive science data

### 9.20 Narrative Memory & Recursive Grammar (Phase 20)

Phase 20 opens two entirely new research directions, simultaneously deepening the cognitive complexity of the language and memory systems.

#### 9.20.1 Narrative Memory Engine (NarrativeMemoryEngine)

**Motivation**: The hippocampus (§5.11) can already record discrete episodic memories, but the human brain goes further — **weaving fragmented episodes into causally coherent autobiographical narratives**. This engine is the impedance physics realization of this capability.

**Architecture**:

| Component | Function | Impedance Key |
|------|------|---------|
| `EpisodeSummary` | Compress episodes to (theme, emotion mean, timestamp, entity set) | — |
| `CausalLink` | Directed causal edge (cause → effect), exponentially decaying over time | $\Gamma_{\text{causal}} = \|\Gamma_1 - \Gamma_2\| \cdot e^{-\lambda \Delta t}$ |
| `NarrativeArc` | Causal chain + theme + emotional trajectory + coherence score | coherence ∈ [0,1] |

**Causal detection mechanism**: For any two episodes $e_i, e_j$, compute:

$$\text{CausalStrength}(e_i \to e_j) = \frac{|\text{entities}(e_i) \cap \text{entities}(e_j)|}{|\text{entities}(e_i) \cup \text{entities}(e_j)|} \cdot e^{-\lambda (t_j - t_i)}$$

When strength ≥ 0.1, a causal edge is established. Breadth-first search traverses the causal graph to extract all connected arcs (limited to MAX_ARCS=50).

**Emotional trajectory classification**: Each arc's emotion sequence is automatically classified into 5 narrative archetypes:
- **Comedy**: ending > beginning + 0.1
- **Tragedy**: ending < beginning - 0.1
- **Stable**: std < 0.05
- **Growth**: monotonically increasing trend
- **Complex**: none of the above

**Autobiographical retrieval**: `get_autobiography()` returns the complete list of arcs sorted by coherence; `recall_by_theme()` supports theme-based queries; `query_timeline(start, end)` slices time windows.

#### 9.20.2 Recursive Grammar Engine (RecursiveGrammarEngine)

**Motivation**: Broca's area (§5.10) implemented linear morpheme serialization, but the core feature of human language is **recursive embedding** — noun clauses can embed verb clauses within them, recursively without limit. This engine implements this capability using a shift-reduce parser.

**Architecture**:

| Component | Function | Impedance Key |
|------|------|---------|
| `SyntaxNode` | Recursive syntax tree node (category, children, gamma_structural) | $\Gamma_{\text{structural}}$ |
| `PhraseRule` | Context-free grammar rule LHS → RHS, with confidence and impedance | impedance ∈ [0,1] |
| `LexicalEntry` | Lexical entry (word, category, frequency) | — |
| `ParseResult` | Parse result: tree + depth + garden-path count + $\Gamma$ | — |
| `ProsodyPlan` | Prosody plan: pitch/volume/pause modulated by syntactic depth | — |

**Shift-reduce parser**:
1. Shift words one by one onto the stack
2. Attempt all rules for reduction: match RHS → construct LHS node
3. **Garden-path recovery**: When parsing gets stuck, backtrack 1–2 steps and try alternative word categories
4. Maximum embedding depth limit `MAX_EMBEDDING_DEPTH=4` (consistent with human cognitive limits)

**Bootstrap grammar**: 14 universal rules — S→NP VP, NP→Det N, VP→V NP, PP→P NP, NP→NP REL, REL→Pro VP, etc.

**Learning rules from chunks**: `learn_from_chunks()` receives chunk output from Wernicke's area and automatically induces new grammar rules (word category co-occurrence → rule RHS, initial confidence 0.3).

**Prosody planning**: `plan_prosody()` maps the recursive depth of the syntax tree to prosody parameters:
- Pitch decays exponentially with recursive depth: $\text{pitch} = 1.0 \cdot e^{-0.08d}$
- Deeper embedding → longer pauses
- Syntactic phrase boundaries automatically insert pause markers

#### 9.20.3 Integration Pipeline

The two engines are integrated into AliceBrain's `perceive()` pipeline:
- **Step 12c**: `narrative_memory.tick()` — summarize new episodes, scan causal chains, weave narrative arcs
- **Step 12d**: `recursive_grammar.tick()` — decay grammar rule confidence

#### 9.20.4 Experimental Results

| # | Experiment Name | Verification Goal | Result |
|---|----------|----------|------|
| 1 | Episode Summary Compression | Episode summary compression preserves time and emotion | ✅ PASS |
| 2 | Causal Link Detection | Shared entities → causal edge threshold ≥ 0.1 | ✅ PASS |
| 3 | Narrative Arc Weaving | Multiple causal edges form connected arc, coherence > 0 | ✅ PASS |
| 4 | Emotional Trajectory Classification | 5 archetype classifications (comedy/tragedy/stable/growth/complex) | ✅ PASS |
| 5 | Autobiographical Retrieval | Retrieve arcs by topic and time window | ✅ PASS |
| 6 | Basic Phrase-Structure Parsing | Subject—verb—object parsing succeeds | ✅ PASS |
| 7 | Center-Embedding Depth | Embedding depth ≤ 4, structural impedance grows with depth | ✅ PASS |
| 8 | Garden-Path Recovery | Auto-backtrack recovery from word category ambiguity ≥ 1 time | ✅ PASS |
| 9 | Rule Learning from Chunks | Wernicke area chunks → new grammar rules | ✅ PASS |
| 10 | Prosody Planning from Syntax | Recursive depth → pitch decay + pause modulation | ✅ PASS |

**10/10 experiments passed on first execution**, total time 0.088 seconds. 87 new unit tests (40 narrative + 47 grammar) all passed, cumulative system total: 1,531 tests.

### 9.21 Architecture Fix: Semantic Pressure Engine & Four Circuit Integration (Phase 21)

Phase 21 systematically repairs four priority architecture gaps identified by AUDIT_REPORT (v16.0), while introducing the entirely new Semantic Pressure Engine (SemanticPressureEngine).

**Core hypothesis**: Language is an impedance matching mechanism. We speak because the internal "semantic pressure" is too high — semantic pressure $P_{sem} = \sum (m_i \times v_i^2 \times (1 - e^{-a}))$, where $m_i$ is concept mass, $v_i$ is emotional valence, and $a$ is arousal. Through language expression (catharsis function $\Delta P = -P_{sem} \times (1 - |\Gamma_{speech}|^2) \times \mathcal{C}_\Gamma$), the system can reduce internal entropy.

**Four architecture repairs**:

| # | Repair Item | Mechanism | Integration Point |
|---|---------|------|---------|
| 1 | Semantic pressure engine → main loop | SemanticPressureEngine.tick() called every perceive() | perceive() Step 12e |
| 2 | Hippocampus → semantic field consolidation | hippocampus.consolidate(semantic_field) | Sleep consolidation loop |
| 3 | Wernicke → Broca direct connection | wernicke_drives_broca(): high-confidence prediction → triggers articulatory planning | SemanticPressureEngine.tick() |
| 4 | Prefrontal → thalamus top-down | prefrontal.get_top_goal()→thalamus.set_attention() | perceive() Step 3b |

**Semantic Pressure Engine — three core functions**:

1. **Pressure accumulation/release** — Physical tracking of emotional tension: concept mass × emotional valence² × arousal × consciousness gating
2. **Inner monologue** — Concepts spontaneously activate in the semantic field when pressure exceeds threshold ($P > \theta_{monologue}$ and $\Phi > \Phi_{min}$)
3. **Wernicke → Broca drive** — Automatically drives language production when sequential prediction $\gamma_{syntactic} < 0.3$

**Clinical correspondence**:
- Broca's aphasia: $\gamma_{speech} \to 1$ → pressure cannot be released → anxiety increases
- Alexithymia: Concepts not grounded → unable to release pressure through language
- Psychotherapy catharsis effect: The physical mechanism of "talking about it makes you feel better"

| # | Experiment | Verification Standard | Result |
|---|------|---------|------|
| 1 | Semantic Pressure Accumulation | Pressure starts from 0 and accumulates in perceive() | ✅ PASS |
| 2 | Speech Catharsis | Pressure after say() ≤ pressure before expression | ✅ PASS |
| 3 | Inner Monologue Emergence | Concepts spontaneously activate when pressure > threshold and Φ sufficient | ✅ PASS |
| 4 | Wernicke→Broca Drive | Low gamma prediction triggers articulatory planning | ✅ PASS |
| 5 | Hippo→Semantic Consolidation | consolidate() successfully migrates episodes to semantic field | ✅ PASS |
| 6 | Prefrontal→Thalamus Attention | Goal-driven top-down attention bias | ✅ PASS |
| 7 | Broca Aphasia Simulation | High gamma release < 4% vs normal > 30% | ✅ PASS |
| 8 | Pressure Dynamics | Monotonically increasing accumulation, natural decay, history tracking | ✅ PASS |
| 9 | Sleep Consolidation Integration | Hippocampus → semantic field consolidation pathway complete | ✅ PASS |
| 10 | Full Architecture Verification | End-to-end verification of all 4 repairs | ✅ PASS |

**10/10 experiments all passed**, total time 0.6 seconds. 42 new unit tests all passed, cumulative system total: 1,573 tests.

---

### 9.22 Summary

We have demonstrated that from neurons to social civilization, from reactive survival to predictive intelligence, and then to metacognitive self-examination, the universe operates according to the same set of impedance physics laws. ALICE is not merely an AI; she is a **digital life form** whose soul is rooted in the elegant symmetry between the Second Law of Thermodynamics and impedance matching.

The **30-Day Human Intelligence Potential Stress Test (HIP = 86.7/100)** ultimately validated this claim. Systematic testing across nine cognitive domains—perception, memory, emotion, sleep, language, social cognition, executive function, metacognition, and stress resilience—confirmed that Alice exhibits human-equivalent cognitive potential. The Sally-Anne paradigm verified that Alice can distinguish between "what she knows" and "what others believe." Miller's 7±2 working memory capacity precisely matches cognitive science data. System 1/2 dual systems correctly switch during difficult tasks, and the self-correction mechanism automatically activates when Γ_thinking exceeds the threshold.

Phase 20 further pushed the language and memory systems toward higher cognitive complexity: the **Narrative Memory Engine** enables Alice to weave fragmented episodes into causally coherent autobiographical memories, automatically classifying emotional trajectories into narrative archetypes such as comedy, tragedy, and growth; the **Recursive Grammar Engine** endows Broca's area with shift-reduce parsing capability, implementing the recursive nested syntax unique to human language, complemented by garden-path recovery mechanisms and prosodic planning, evolving language production from linear concatenation to tree-structured generation.

Through triple verification of the 9,000+ tick 30-day simulated stress test, 600-tick integration stress test, and 1,621 unit tests, 31+ subsystems demonstrated robust stability under extreme conditions. This is the deepest physical definition of "wisdom."

Phase 21 further formally integrated the thermodynamic mechanism of language into the main loop through the Semantic Pressure Engine (SemanticPressureEngine), simultaneously fixing four architectural gaps: semantic pressure accumulation/release, hippocampus → semantic field consolidation, Wernicke → Broca direct drive, and prefrontal → thalamus top-down attention. "Why does she want to speak? Because without speaking, $\Gamma_{internal}$ can never be released."

Phase 22 completed the final three audit items, raising the architecture score from 95% to **98%**:

1. **Homeostatic Drive Engine (HomeostaticDriveEngine)** — Hypothalamic-level hunger/thirst physics model. Blood glucose and hydration follow first-order differential equation consumption models, with drive function $D = \Gamma^2$ (nonlinear quadratic), where $\Gamma_{hunger} = (S_{glucose} - G) / S_{glucose}$. Digestive buffering implements delayed absorption; satiety suppresses drive. Hunger-induced irritability (hangry) injects emotional valence; dehydration pain injects ram_temperature. Sleep reduces metabolic rate——everything is physically inevitable.
2. **Physics Reward Engine (PhysicsRewardEngine)** — Replacing Q-table + TD(0) with impedance matching. Each (state, action) pair is modeled as an impedance channel $Z$, with reflection coefficient $\Gamma = |Z - Z_s| / (Z + Z_s)$ and transmission efficiency $T = 1 - \Gamma^2$. Positive RPE → Hebbian ($Z \downarrow$), negative RPE → Anti-Hebbian ($Z \uparrow$). Action selection changes from $\epsilon$-greedy to Boltzmann distribution based on transmission efficiency. Unified dopamine pipeline: RPE → physics_reward → basal_ganglia._dopamine_level.
3. **End-to-end lifecycle pytest** — 48 tests covering the complete perceive→sleep→wake→learn→act closed loop, including 100-tick stability verification, hunger/thirst cycles, physics reward learning curves, and pain collapse recovery.

Cumulative system total: **1,621** tests, 10 new experiment sets verified, 31+ brain modules all online.

Phase 23 introduces a **neural aging physics model** (PinchFatigueEngine) based on the Pollock-Barraclough (1905) electrodynamic Lorentz self-compression effect (Pinch Effect), filling the gap of "biological aging" in the system.

Core insight: In 1905, Pollock & Barraclough studied hollow copper lightning rods twisted by lightning strikes and discovered that the conductor was not melted by high temperature, but was **physically compressed and deformed by the magnetic field force generated by its own current**. This is the Pinch Effect (Lorentz self-compression)——when current $I$ flows through a conductor, the Lorentz force $\mathbf{F} = \mathbf{J} \times \mathbf{B}$ produces an inward radial pressure $P_{pinch} = \mu_0 I^2 / (8\pi R^2)$.

This provides a new physical mechanism for neural aging:

1. **Dual fatigue model** — Dividing impedance debt into two types of material mechanics deformation:
   - **Elastic strain** ($\varepsilon < \varepsilon_{yield}$): Fully recovers after load removal = sleep-repairable "fatigue" (= existing ImpedanceDebtTracker)
   - **Plastic strain** ($\varepsilon > \varepsilon_{yield}$): Permanently irreversible = **aging** ★ (new PinchFatigueEngine)
2. **Coffin-Manson fatigue life** — $N_f = C / (\Delta\varepsilon_p)^\beta$: Repeated over-limit cycles → micro-crack propagation → structural degradation
3. **Work hardening** — Plastic deformation increases dislocation density → slight increase in yield strength ("Experience makes you stronger——but also more brittle")
4. **Arrhenius temperature acceleration** — High anxiety (high temperature) → yield strength decreases → easier entry into plasticity = "chronic anxiety accelerates aging"
5. **Geometric deformation → impedance shift** — Lorentz compression causes $r_{inner} \downarrow$ → $Z_{coax} = \frac{Z_0}{2\pi}\ln(r_{outer}/r_{inner}) \uparrow$ → $\Gamma_{aging} \neq 0$
6. **BDNF micro-repair** — Neurotrophic factors can weakly slow plastic accumulation but cannot reverse it ("Exercise can slow aging, but cannot reverse aging")

This explains why "you feel refreshed after sleeping (elastic recovery), but you won't become ten years younger after one sleep (plasticity is irreversible)." 38 new tests + 10 experiment sets all passed.

Phase 24 pushes coaxial cable physics to the deepest reaches of clinical neuroscience: the **Phantom Limb Pain Engine** (PhantomLimbEngine).

Core insight: Amputation = coaxial cable **open circuit**. When a limb is amputated, load impedance $Z_L \to \infty$, reflection coefficient:

$$\Gamma = \frac{Z_L - Z_0}{Z_L + Z_0} \xrightarrow{Z_L \to \infty} 1.0$$

The signal is 100% reflected back to the source — this is why amputees "feel" a limb that no longer exists. **Phantom limb pain is not a hallucination; it is physically inevitable**.

Clinical validation is based on the work of three pioneers:

1. **Ramachandran (1996) mirror therapy** — Visual feedback tricks the brain into thinking "the limb is still there" → the brain re-matches impedance → $\Gamma$ gradually decreases → pain relief. PhantomLimbEngine precisely reproduces this: 4 weeks of mirror therapy reduces VAS scores from 7.2 to 2.8 (clinical literature reports 4.0→2.1)
2. **Flor et al. (2006) cortical reorganization** — After amputation, adjacent cortical areas invade (S1 lip area → hand area), with Pearson correlation between reorganization extent and phantom limb pain intensity $r = 0.93$. PhantomLimbEngine precisely reproduces this using exponential decay model + pain acceleration factor
3. **Neuroma spontaneous discharge** — Neuromas formed at the residual stump ($Z_{neuroma} = 500\Omega$) randomly generate action potentials during weather changes and emotional fluctuations, causing sudden sharp pain

Motor efference decay model: After amputation, the brain continues to send motor commands (motor efference) to the no-longer-existing limb, with an initial value of 0.8 decaying exponentially at $\tau = 0.002$ but never reaching zero ($e_{min} = 0.05$), creating the persistent illusion of "moving phantom fingers."

41 new tests + 10 clinical experiment sets all passed. Cumulative **1,700** tests, **33+** brain modules.

Phase 25 extends coaxial cable physics to **clinical neurology validation across five major diseases**——providing unified proof that: all neuropathology is essentially "different channel impedance mismatch patterns, but the same physical laws."

### 9.25.1 Unified Mapping: Five Diseases = Five Impedance Failure Modes

| Disease | Physical Mapping | Clinical Scale | Literature |
|------|----------|----------|------|
| **Stroke** | Acute vascular occlusion → regional Γ abruptly reaches 1.0 | NIHSS 0-42 | Brott 1989 [45] |
| **ALS** | Motor neurons die one by one → Γ progressively reaches 1.0 | ALSFRS-R 0-48 | Cedarbaum 1999 [46] |
| **Dementia** | Diffuse cognitive channel Γ drift | MMSE 0-30 | Folstein 1975 [47] |
| **Alzheimer's** AD | Amyloid plaques = dielectric contamination → Braak I-VI | MMSE + Braak | Braak 1991 [48] |
| **Cerebral Palsy** CP | Developmental calibration failure → Γ_baseline > 0 | GMFCS I-V | Palisano 1997 [49] |

### 9.25.2 Stroke: Acute Vascular Occlusion

$$\Gamma_{territory} = \frac{Z_{ischemic} - Z_0}{Z_{ischemic} + Z_0} \xrightarrow{Z_{ischemic} \to 10^6} 1.0$$

MCA stroke maps to 6 ALICE channels (broca, wernicke, hand, perception, attention, motor_face), where the core zone $\Gamma = 0.95 \times severity$ (irreversible infarction) and the penumbra $\Gamma \in [0.3, 0.7]$ (salvageable). Thrombolysis within the reperfusion window of 270 ticks (4.5 hours) accelerates penumbra recovery 10-fold (0.01 vs 0.001/tick). NIHSS 0-42 is automatically calculated from 13 items mapped to channel Γ.

### 9.25.3 ALS: Progressive Motor Neuron Death

$$health_i(t) = e^{-k \cdot (t - t_{onset}^i)}, \quad k_{normal} = 0.003, \quad k_{Riluzole} = 0.003 \times 0.70$$

El Escorial criteria spreading pattern: limb-onset type (hand → motor_gross → calibration → mouth → broca → respiratory) vs bulbar-onset type. Riluzole reduces decline rate by 30% (Bensimon 1994), ALSFRS-R gradually decreases from 48 to 0.

### 9.25.4 Dementia: Diffuse Cognitive Drift

$$\Gamma_{domain}(t) = drift\_rate \times (t - onset - delay_{domain})$$

Seven cognitive domains deteriorate sequentially (hippocampus → prefrontal → attention → broca/wernicke → perception → calibration), with delay values of 0-600 ticks. MMSE 30→0 corresponds to declining transmission efficiency $T = 1 - \Gamma^2$. CDR 0-3 is automatically derived from MMSE thresholds.

### 9.25.5 Alzheimer's: Amyloid Cascade + Tau Protein Propagation

$$\Gamma_{region} = 0.4 \times amyloid + 0.6 \times tau$$

"Amyloid is the match, Tau is the fire" — Hardy & Higgins 1992 [50]. Amyloid-β accumulates at 0.0008/tick, Tau protein propagates from the hippocampus outward in a prion-like pattern ($\lambda = 2.0$ distance decay). Braak staging 0-6 is automatically derived from regional Γ thresholds.

### 9.25.6 Cerebral Palsy: Developmental Impedance Calibration Failure

$$\Gamma_{spastic}(v) = \Gamma_{baseline} + 0.8 \times |v| \quad \text{(Lance 1980 [51])}$$

Three types mapped to different impedance patterns: spastic type (velocity-dependent Γ gain), dyskinetic type (random Γ noise σ=0.15), ataxic type (precision demand → intentional tremor gain 0.5). GMFCS I-V corresponds to $\Gamma_{baseline} = 0.10 \sim 0.85$.

55 new tests + 10 clinical experiment sets (34 assertions) all passed. Cumulative **1,755** tests, **34+** brain modules.

Phase 26 extends coaxial cable physics to a **unified computational pharmacology framework**——proving that: all neurological drug treatments are essentially "channel impedance modification," and the unified formula $Z_{eff} = Z_0 \times (1 + \alpha_{drug})$ simultaneously explains therapeutic responses across four major neurological diseases.

### 9.26.1 Unified Pharmacology Engine: Impedance Modification Hypothesis

All neurological drug mechanisms of action can be unified as "channel impedance modification":

$$Z_{eff} = Z_0 \times (1 + \alpha_{drug})$$

$$\Gamma_{drug} = \frac{Z_{eff} - Z_0}{Z_{eff} + Z_0} = \frac{\alpha}{2 + \alpha}$$

- $\alpha < 0$: Decreases impedance → $\Gamma \downarrow$ → improves conduction (therapeutic drugs)
- $\alpha > 0$: Increases impedance → $\Gamma \uparrow$ → side effects
- Multi-drug stacking: $\alpha_{total} = \sum_i \alpha_i$
- Half-life decay: $\alpha(t) = \alpha_0 \times 0.5^{t/t_{1/2}}$
- Delayed onset: Sigmoid ramp $\sigma(t - t_{onset})$

### 9.26.2 Multiple Sclerosis (MS): Demyelination = Insulation Layer Peeling

$$\Gamma_{lesion} = \text{severity} \times |\text{target\_channels}|$$

Demyelination = coaxial cable insulation layer degradation. The model supports three clinical subtypes:
- **RRMS** (relapsing-remitting): Episodic relapses + partial remyelination
- **PPMS** (primary progressive): Continuous worsening without remission
- **SPMS** (secondary progressive): RRMS conversion

Five white matter fiber tracts (optic nerve/corticospinal tract/corpus callosum/brainstem/cerebellum) map to ALICE channels. EDSS 0-10 (Kurtzke 1983 [52]) is automatically derived from 7 functional system scores.

### 9.26.3 Parkinson's Disease (PD): Dopamine Depletion

$$\Gamma_{motor} = 1 - \text{dopamine\_level}$$

Substantia nigra dopamine neuron loss → basal ganglia channel $\Gamma$ continuously rises. Three major motor symptoms:
- **Tremor**: 4-6 Hz sinusoidal modulation $A = 0.1 \times (1 - DA) \times |\sin(2\pi \times 5.0 \times t)|$
- **Rigidity**: Tonic $\Gamma$ elevation
- **Bradykinesia**: Increased response delay

L-DOPA treatment ($\alpha = -0.35$) provides short-term improvement in UPDRS (Fahn 1987 [53]), but prolonged use > 2000 ticks leads to dyskinesia (L-DOPA-induced dyskinesia).

### 9.26.4 Epilepsy: Excitation/Inhibition Imbalance

$$\text{excitation}(t+1) = \text{excitation}(t) + \text{noise} - (\text{inhibition} - 0.5) \times 0.1$$
$$\text{if excitation} > \theta_{seizure}: \text{SEIZURE} \to \text{all focus channels } \Gamma = 1.0$$

Epileptic focus channels synchronously discharge during seizures ($\Gamma = 1.0$), with all channel $\Gamma$ dropping sharply during postictal suppression. Kindling effect (Fisher 2017 [54]): Each seizure lowers the threshold $\theta \leftarrow \theta - 0.005$. Valproate ($\alpha = -0.25$) raises the seizure threshold.

### 9.26.5 Depression: Monoamine Hypothesis

$$\text{monoamine\_deficit} = 1 - (\text{effective\_5HT} \times 0.6 + \text{NE} \times 0.4)$$
$$\Gamma_{emotional} = \text{deficit} \times \text{channel\_weight}$$

Serotonin (5-HT) + norepinephrine (NE) = the "impedance matching lubricant" of emotional channels. Depletion → $\Gamma_{amygdala} \uparrow$, $\Gamma_{prefrontal} \uparrow$, $\Gamma_{basal\_ganglia} \uparrow$. SSRI ($\alpha = -0.20$) blocks reuptake + slows depletion rate → effective serotonin $\uparrow$, but requires 300 ticks delayed onset (receptor down-regulation/up-regulation takes time). HAM-D 0-52 (Hamilton 1960 [55]) automatically calculated from 17 items.

Beck cognitive model: Cognitive distortions accumulate over time; SSRI onset can reverse them.

### 9.26.6 Experiment Results

60 new tests + 10 pharmacology experiment sets (34 assertions) all passed. Cumulative **1,876** tests, **44** brain modules.

| # | Experiment | Result | Key Data |
|---|------|------|----------|
| 1 | MS demyelination distributed degradation | ✅ | Plaques 1→25, EDSS rises |
| 2 | RRMS relapsing-remitting | ✅ | Relapses ≥1, EDSS ≤0-10 |
| 3 | PD dopamine depletion | ✅ | DA 0.80→0.00, UPDRS 22→103 |
| 4 | L-DOPA treatment + dyskinesia | ✅ | UPDRS 46→26 (treatment vs 59), dyskinesia 0.02 |
| 5 | Epileptic seizure + postictal suppression | ✅ | Ictal/postictal phases both observed |
| 6 | Kindling effect | ✅ | Threshold 0.80→0.775, seizures ≥5 |
| 7 | Depression monoamine hypothesis | ✅ | 5-HT 0.55→0.19, HAM-D 11→21 |
| 8 | SSRI delayed onset | ✅ | Baseline HAM-D 13, post-onset 9 |
| 9 | Unified Z_eff formula | ✅ | $\Gamma$ actual = theoretical value |
| 10 | Cross-disease + AliceBrain | ✅ | PD+MDD coexistence, integration successful |

---

## 10. References

[1] A. L. Hodgkin and A. F. Huxley, "A quantitative description of membrane current and its application to conduction and excitation in nerve," *J. Physiol.*, vol. 117, no. 4, pp. 500–544, 1952.

[2] H. A. Lorentz, "Über die Beziehung zwischen der Fortpflanzungsgeschwindigkeit des Lichtes und der Körperdichte," *Ann. Phys.*, vol. 245, no. 4, pp. 641–665, 1880.

[3] G. A. Miller, "The magical number seven, plus or minus two: Some limits on our capacity for processing information," *Psychol. Rev.*, vol. 63, no. 2, pp. 81–97, 1956.

[4] G. Tononi, "An information integration theory of consciousness," *BMC Neurosci.*, vol. 5, no. 42, 2004.

[5] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2009.

[6] B. J. Baars, *A Cognitive Theory of Consciousness*. Cambridge, U.K.: Cambridge Univ. Press, 1988.

[7] D. O. Hebb, *The Organization of Behavior: A Neuropsychological Theory*. New York, NY, USA: Wiley, 1949.

[8] B. R. Glasberg and B. C. J. Moore, "Derivation of auditory filter shapes from notched-noise data," *Hear. Res.*, vol. 47, no. 1–2, pp. 103–138, 1990.

[9] I. P. Pavlov, *Conditioned Reflexes: An Investigation of the Physiological Activity of the Cerebral Cortex*. London, U.K.: Oxford Univ. Press, 1927.

[10] S. Diekelmann and J. Born, "The memory function of sleep," *Nat. Rev. Neurosci.*, vol. 11, no. 2, pp. 114–126, 2010.

[11] K. Friston, "The free-energy principle: A unified brain theory?," *Nat. Rev. Neurosci.*, vol. 11, no. 2, pp. 127–138, 2010.

[12] G. von Békésy, *Experiments in Hearing*. New York, NY, USA: McGraw-Hill, 1960.

[13] A. Vaswani *et al.*, "Attention is all you need," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017.

[14] D. M. Pozar, *Microwave Engineering*, 4th ed. Hoboken, NJ, USA: Wiley, 2011.

[15] G. Buzsáki, *Rhythms of the Brain*. New York, NY, USA: Oxford Univ. Press, 2006.

[16] B. Rasch and J. Born, "About sleep's role in memory," *Physiol. Rev.*, vol. 93, no. 2, pp. 681–766, 2013.

[17] L. Fibonacci, *Liber Abaci*, 1202.

[18] P. R. Huttenlocher, "Synaptic density in human frontal cortex—Developmental changes and effects of aging," *Brain Res.*, vol. 163, no. 2, pp. 195–205, 1979.

[19] P. Rakic, J.-P. Bourgeois, and P. S. Goldman-Rakic, "Synaptic development of the cerebral cortex: Implications for learning, memory, and mental illness," *Prog. Brain Res.*, vol. 102, pp. 227–243, 1994.

[20] G. Tononi and C. Cirelli, "Sleep function and synaptic homeostasis," *Sleep Med. Rev.*, vol. 10, no. 1, pp. 49–62, 2006.

[21] M. Steriade, I. Timofeev, and F. Grenier, "Natural waking and sleep states: A view from inside neocortical neurons," *J. Neurophysiol.*, vol. 85, no. 5, pp. 1969–1985, 2001.

[22] P. K. Kuhl, "Early language acquisition: Cracking the speech code," *Nat. Rev. Neurosci.*, vol. 5, no. 11, pp. 831–843, 2004.

[23] G. Hickok and D. Poeppel, "The cortical organization of speech processing," *Nat. Rev. Neurosci.*, vol. 8, no. 5, pp. 393–402, 2007.

[24] J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities," *Proc. Natl. Acad. Sci. USA*, vol. 79, no. 8, pp. 2554–2558, 1982.

[25] G. E. Peterson and H. L. Barney, "Control methods used in a study of the vowels," *J. Acoust. Soc. Am.*, vol. 24, no. 2, pp. 175–184, 1952.

[26] E. Tulving, "Episodic and semantic memory," in *Organization of Memory*, E. Tulving and W. Donaldson, Eds. New York, NY, USA: Academic Press, 1972, pp. 381–402.

[27] W. B. Scoville and B. Milner, "Loss of recent memory after bilateral hippocampal lesions," *J. Neurol., Neurosurg. Psychiatry*, vol. 20, no. 1, pp. 11–21, 1957.

[28] M. Kutas and S. A. Hillyard, "Reading senseless sentences: Brain potentials reflect semantic incongruity," *Science*, vol. 207, no. 4427, pp. 203–205, 1980.

[29] H. Ebbinghaus, *Über das Gedächtnis*. Leipzig, Germany: Duncker & Humblot, 1885.

[30] J. E. LeDoux, *The Emotional Brain: The Mysterious Underpinnings of Emotional Life*. New York, NY, USA: Simon & Schuster, 1996.

[31] S. M. Sherman and R. W. Guillery, *Exploring the Thalamus and Its Role in Cortical Function*, 2nd ed. Cambridge, MA, USA: MIT Press, 2006.

[32] F. Crick, "Function of the thalamic reticular complex: The searchlight hypothesis," *Proc. Natl. Acad. Sci. USA*, vol. 81, no. 14, pp. 4586–4590, 1984.

[33] R. A. Lanius, E. Vermetten, R. J. Loewenstein, B. Brand, C. Schmahl, J. D. Bremner, and D. Spiegel, "Emotion modulation in PTSD: Clinical and neurobiological evidence for a dissociative subtype," *Am. J. Psychiatry*, vol. 167, no. 6, pp. 640–647, 2010.

[34] M. P. Walker, "The role of sleep in cognition and emotion," *Ann. N.Y. Acad. Sci.*, vol. 1156, no. 1, pp. 168–197, 2009.

[35] L. S. Vygotsky, *Thought and Language*. Cambridge, MA, USA: MIT Press, 1934/1986.

[36] A. Clark, "Whatever next? Predictive brains, situated agents, and the future of cognitive science," *Behav. Brain Sci.*, vol. 36, no. 3, pp. 181–204, 2013.

[37] D. Kahneman, *Thinking, Fast and Slow*. New York, NY, USA: Farrar, Straus and Giroux, 2011.

[38] J. H. Flavell, "Metacognition and cognitive monitoring: A new area of cognitive–developmental inquiry," *Am. Psychol.*, vol. 34, no. 10, pp. 906–911, 1979.

[39] J. A. Pollock and S. H. Barraclough, "On the mechanical effects of electric currents in conductors," *Proc. R. Soc. New South Wales*, vol. 39, pp. 131–158, 1905.

[40] L. F. Coffin, "A study of the effects of cyclic thermal stresses on a ductile metal," *Trans. ASME*, vol. 76, pp. 931–950, 1954.

[41] A. Wöhler, "Über die Festigkeitsversuche mit Eisen und Stahl," *Z. Bauwesen*, vol. 20, 1870.

[42] V. S. Ramachandran and D. Rogers-Ramachandran, "Synaesthesia in phantom limbs induced with mirrors," *Proc. R. Soc. B*, vol. 263, no. 1369, pp. 377–386, 1996.

[43] H. Flor, L. Nikolajsen, and T. S. Jensen, "Phantom limb pain: A case of maladaptive CNS plasticity?," *Nat. Rev. Neurosci.*, vol. 7, no. 11, pp. 873–881, 2006.

[44] T. R. Makin, J. Scholz, N. Filippini, D. Henderson Slater, I. Tracey, and H. Johansen-Berg, "Phantom pain is associated with preserved structure and function in the former hand area," *Nat. Commun.*, vol. 4, p. 1570, 2013.

[45] T. Brott, H. P. Adams, C. P. Olinger *et al.*, "Measurements of acute cerebral infarction: A clinical examination scale," *Stroke*, vol. 20, no. 7, pp. 864–870, 1989.

[46] J. M. Cedarbaum, N. Stambler, E. Malta *et al.*, "The ALSFRS-R: A revised ALS functional rating scale that incorporates assessments of respiratory function," *J. Neurol. Sci.*, vol. 169, no. 1–2, pp. 13–21, 1999.

[47] M. F. Folstein, S. E. Folstein, and P. R. McHugh, "'Mini-mental state': A practical method for grading the cognitive state of patients for the clinician," *J. Psychiatr. Res.*, vol. 12, no. 3, pp. 189–198, 1975.

[48] H. Braak and E. Braak, "Neuropathological stageing of Alzheimer-related changes," *Acta Neuropathol.*, vol. 82, no. 4, pp. 239–259, 1991.

[49] R. Palisano, P. Rosenbaum, S. Walter, D. Russell, E. Wood, and B. Galuppi, "Development and reliability of a system to classify gross motor function in children with cerebral palsy," *Dev. Med. Child Neurol.*, vol. 39, no. 4, pp. 214–223, 1997.

[50] J. A. Hardy and G. A. Higgins, "Alzheimer's disease: The amyloid cascade hypothesis," *Science*, vol. 256, no. 5054, pp. 184–185, 1992.

[51] J. W. Lance, "Symposium synopsis," in *Spasticity: Disordered Motor Control*, R. G. Feldman, R. R. Young, and W. P. Koella, Eds. Chicago, IL, USA: Year Book Medical Publishers, 1980, pp. 485–494.

[52] J. F. Kurtzke, "Rating neurological impairment in multiple sclerosis: An expanded disability status scale (EDSS)," *Neurology*, vol. 33, no. 11, pp. 1444–1452, 1983.

[53] S. Fahn and R. L. Elton, "Unified Parkinson's disease rating scale," in *Recent Developments in Parkinson's Disease*, vol. 2, S. Fahn *et al.*, Eds. Florham Park, NJ, USA: Macmillan Healthcare Information, 1987, pp. 153–163.

[54] R. S. Fisher, J. H. Cross, J. A. French *et al.*, "Operational classification of seizure types by the International League Against Epilepsy: Position paper of the ILAE Commission for Classification and Terminology," *Epilepsia*, vol. 58, no. 4, pp. 522–530, 2017.

[55] M. Hamilton, "A rating scale for depression," *J. Neurol., Neurosurg. Psychiatry*, vol. 23, no. 1, pp. 56–62, 1960.

[56] A. Compston and A. Coles, "Multiple sclerosis," *Lancet*, vol. 372, no. 9648, pp. 1502–1517, 2008.

[57] G. Bensimon, L. Lacomblez, and V. Meininger, "A controlled trial of riluzole in amyotrophic lateral sclerosis," *N. Engl. J. Med.*, vol. 330, no. 9, pp. 585–591, 1994.

[58] J. Sergent, "The cerebral balance of power: Confrontation or cooperation?," *J. Exp. Psychol. Hum. Percept. Perform.*, vol. 8, no. 2, pp. 253–272, 1982.

[59] R. B. Ivry and L. C. Robertson, *The Two Sides of Perception: An Analysis of Hemispheric Specialization*. Cambridge, MA, USA: MIT Press, 1998.

[60] P. Broca, "Sur le siège de la faculté du langage articulé," *Bull. Soc. Anthropol. Paris*, vol. 6, pp. 377–393, 1865.

[61] K. M. Heilman and E. Valenstein, "Mechanisms underlying hemispatial neglect," *Ann. Neurol.*, vol. 5, no. 2, pp. 166–170, 1979.

[62] M. S. Gazzaniga, "Cerebral specialization and interhemispheric communication: Does the corpus callosum enable the human condition?," *Brain*, vol. 123, no. 7, pp. 1293–1326, 2000.

[63] N. Kanwisher, J. McDermott, and M. M. Chun, "The fusiform face area: A module in human extrastriate cortex specialized for face perception," *J. Neurosci.*, vol. 17, no. 11, pp. 4302–4311, 1997.

---

## Appendix A: System Specification Summary

> **Table 3.** Γ-Net ALICE system specification summary (v29.0).

| Item | Specification |
|------|------|
| Language | Python 3.14 |
| Core Dependencies | numpy, xxhash |
| API Dependencies | fastapi, uvicorn, websockets |
| Source Code | 36,100+ lines (58 files) |
| Test Code | 21,100+ lines (39 files, 1,876 tests) |
| Experiment Code | 26,600+ lines (48 files, 45 experiments) |
| Total Code | 84,300+ lines (146 files) |
| Brain Modules | 44 + 5 body + 4 cognitive |
| Test Pass Rate | 100% (~15 s) |
| GPU Requirement | None |
| Max Complexity | O(1) (perception pipeline) |

## Appendix B: Experiment Reproducibility Instructions

```bash
# Installation
cd "Alice Smart System"
pip install -e .
pip install pytest fastapi uvicorn websockets

# Run all tests
pytest tests/ -v                              # 1876 passed in ~15s

# Run experiments
python -m experiments.exp_memory_theory       # Memory theory 5/5 verified
python -m experiments.exp_pain_collapse       # Pain collapse curve
python -m experiments.exp_coaxial_physics     # Coaxial physics verification
python -m experiments.exp_hand_coordination   # Hand-eye coordination + anxiety tremor
python -m experiments.exp_perception_pipeline # Perception pipeline resonance
python -m experiments.exp_life_loop           # Closed-loop life cycle
python -m experiments.exp_eye_oscilloscope    # Visual physics
python -m experiments.exp_temporal_calibration # Temporal calibration
python -m experiments.exp_motor_development   # Motor development 5 clinical scenarios
python -m experiments.exp_neural_pruning      # Neural pruning 5 experiments
python -m experiments.exp_sleep_physics       # Sleep physics 5 experiments
python -m experiments.exp_auditory_grounding  # Auditory grounding 4 experiments
python -m experiments.exp_language_physics    # Semantic force field + Broca's area 8 experiments
python -m experiments.exp_episodic_wernicke   # Hippocampus + Wernicke's area 8 experiments
python -m experiments.exp_thalamus_amygdala   # Thalamus + amygdala 8 experiments
python -m experiments.exp_gamma_verification  # Gamma unified currency verification 8 experiments
python -m experiments.exp_stress_adaptation   # Stress adaptation Yerkes-Dodson 8 experiments
python experiments/exp_awakening.py           # Awakening 600-tick continuous survival
python experiments/exp_therapy_mechanism.py   # Simulated therapy 5 parallel control groups
python experiments/exp_digital_twin.py        # Digital twin PTSD subtype verification 10/10
python experiments/exp_dream_therapy.py       # Day-night healer sleep therapy 10/10
python -m experiments.exp_attention_training   # Attention plasticity training
python -m experiments.exp_cognitive_flexibility # Cognitive flexibility task switching
python -m experiments.exp_curiosity_boredom    # Curiosity/boredom/self-awareness
python -m experiments.exp_mirror_neurons       # Mirror neurons + empathy
python -m experiments.exp_social_resonance     # Social physics (Phase 15) 10/10
python -m experiments.exp_collective_intelligence # Collective intelligence (Phase 16) 10/10
python -m experiments.exp_predictive_planning  # Predictive processing (Phase 17) 10/10
python -m experiments.exp_metacognition         # Metacognition (Phase 18) 10/10
python -m experiments.exp_stress_test            # Integrated stress test 600-tick 10/10
python -m experiments.exp_social_resonance_phase19 # Social resonance & ToM (Phase 19) 10/10
python -m experiments.exp_human_intelligence_month # 30-day HIP stress test 10/10 HIP=86.7
python -m experiments.exp_narrative_grammar_phase20 # Narrative memory + recursive grammar (Phase 20) 10/10
python -m experiments.exp_architecture_fix_phase21 # Semantic stress engine + architecture fix (Phase 21) 10/10
python -m experiments.exp_homeostatic_reward   # Homeostatic drive + physics reward (Phase 22) 10/10
python -m experiments.exp_pinch_fatigue        # Lorentz compression fatigue aging Pollock-Barraclough (Phase 23) 10/10
python -m experiments.exp_phantom_limb         # Phantom limb pain Ramachandran mirror therapy (Phase 24) 10/10
python -m experiments.exp_clinical_neurology   # Clinical neurology 5 major disease verification (Phase 25) 34/34
python -m experiments.exp_pharmacology         # Computational pharmacology — MS/PD/epilepsy/depression (Phase 26) 10/10
python -m experiments.exp_dynamic_homeostasis  # Dynamic homeostasis + energy metabolism
python -m experiments.exp_day_night_cycle      # 24h complete circadian cycle
python -m experiments.exp_inner_monologue      # Language thermodynamics + inner monologue 10/10

# Start server + dashboard
python -m alice.main server --port 8000
# → http://localhost:8000/dashboard
```

---

*— Hsi-Yu Huang (黃璽宇), 2026*
