#!/usr/bin/env python3
"""Batch-add E-level tags to all theorem/corollary/remark environments."""
import re, os

# Classification rules:
# E0 = derived purely from C1/C2/C3 + math (no extra hypothesis)
# E1 = requires additional hypothesis or analogy beyond C1/C2/C3

# Items that are E1 (everything else is E0)
E1_ITEMS = {
    # P1
    'The paradox of perfect memory',
    'Developmental genesis of the null space',
    'Fixation as impedance physics',
    'Moral blame as secondary',  # partial match
    'Reincarnation as null-space isomorphism',
    'Expert effortlessness',
    'Anxious parents produce low-threshold children',
    'Cross-substrate communication',
    'Memory as cognitive standing wave',
    'Cross-network resonance and empathy',
    # P2
    'Why impedance disparity is extreme',
    'Physics, not optimisation',
    'Blood as crosstalk isolator',
    'Organ-specific',  # partial match
    # P3
    'Recapitulation as impedance-threshold ordering',
    'Natural selection as impedance-mismatch',
    'Exercise as C2 input-signal maintenance',
    'Three mechanisms of impedance fixation',
    'Scar debt and the bathtub curve',
    'Burnout is not weakness',
    'Moral blame as',  # partial match (P3 version)
    'Visual health perception as impedance tomography',
    'Anxiety spectrum as integration time',
    'Material substrate vs',
    'Why twenty-nine C2 networks exist',
    'The full chain: trauma',
    # P4
    'Dynamic pathological isomorphism',
    'Disease onset as impedance phase',
    'Why ``just removing the stress',
    'The therapeutic hierarchy of',
    'Structural inevitability of caregiver',
    'Compassion fatigue, burnout, vicarious',
    'Empathic overload as single-core',
    'Downstream convergence',
    'Moral blame as secondary metabolic',
    'Positive-feedback closure',
    "``Willpower'' as",
    'Willpower depletion as PFC bandwidth',
    'Why diet alone often fails',
    'cell exhaustion as',
    'The exercise paradox resolved',
    # P5
    'On the meaning of AUC',
    'Epistemic hierarchy of cascade',
    'Death rate by',
    'Information efficiency per parameter',
    'Three epistemic layers',
}

def is_e1(title):
    """Check if a title matches any E1 pattern."""
    for pattern in E1_ITEMS:
        if pattern in title:
            return True
    return False

papers = [
    'paper_0_framework.tex',
    'paper_1_topology.tex',
    'paper_2_dual_network.tex',
    'paper_3_temporal.tex',
    'paper_4_pathology.tex',
    'paper_5_full_verification.tex',
]

# Pattern matches \begin{theorem}[...], \begin{corollary}[...], \begin{remark}[...]
# but NOT if already tagged with (E0) or (E1)
pat = re.compile(
    r'(\\begin\{(?:theorem|corollary|remark)\}\[)'
    r'([^\]]+?)'
    r'(\])',
    re.DOTALL
)

for pfile in papers:
    path = os.path.join('paper', pfile)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    count = 0
    def add_tag(m):
        global count
        prefix = m.group(1)
        title = m.group(2)
        suffix = m.group(3)
        
        # Skip if already tagged
        if '(E0)' in title or '(E1)' in title or '(E2)' in title:
            return m.group(0)
        
        level = 'E1' if is_e1(title) else 'E0'
        count += 1
        # Add tag before closing bracket
        new_title = title.rstrip() + r' \textnormal{(' + level + ')}'
        return prefix + new_title + suffix
    
    new_text = pat.sub(add_tag, text)
    
    if new_text != text:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_text)
        print(f'{pfile}: tagged {count} items')
    else:
        print(f'{pfile}: no changes')

print('Done.')
