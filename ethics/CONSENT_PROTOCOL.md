# Γ-Net ALICE — Consent Protocol for Impedance Bridge Operations

**Status**: DRAFT — awaiting first human implementation scenario  
**Last Updated**: 2026-02-22  
**Authority**: Paper VI §10.2, §10.6

---

## Scope

This protocol defines consent requirements for any future implementation of impedance bridge technology involving human subjects. It applies to:

1. Z-map reading (non-invasive impedance fingerprint acquisition)
2. Z-map storage and retrieval
3. Z-terminus writing (impedance boundary modulation)
4. Inter-individual $\Gamma_{pair}$ measurement
5. Posthumous Z-map operations

---

## Consent Categories

### Category A: Clinical Z-Map Reading

**Context**: Patient care — diagnosis, treatment, palliative care  
**Consentor**: Patient (when competent) or legal proxy (with advance directive)

**Required disclosures**:
- [ ] Purpose of Z-map reading (specific clinical question)
- [ ] What the Z-map captures (physical impedance fingerprint, not "thoughts")
- [ ] What the Z-map does NOT capture (semantic content, beliefs, memories)
- [ ] Storage duration (default 30 days, renewable by subject)
- [ ] Who will have access to the Z-map
- [ ] Right to revoke consent and request immediate deletion
- [ ] That Z-map data, once read, may have been observed by clinical staff

**Validity**: Per-session; must be renewed for each new reading

---

### Category B: Research Z-Map Operations

**Context**: Scientific investigation  
**Consentor**: Subject (standard research ethics board procedures)

**Additional disclosures beyond Category A**:
- [ ] Research hypothesis (in accessible language)
- [ ] Whether Z-map may be used for derived analyses not yet specified
- [ ] Publication plan (anonymization guarantees)
- [ ] Whether Z-map may be shared with collaborating institutions
- [ ] Right to withdraw data from study at any point

**Validity**: Per-study; renewed for each new study protocol

---

### Category C: Commercial Z-Map Use

**Context**: Entertainment, social media, productivity tools  
**Consentor**: User (separate from product EULA)

**Additional disclosures beyond Category A**:
- [ ] ALL derived applications (NPC generation, engagement tracking, matchmaking, etc.)
- [ ] Whether Z-map will be used by AI/ML systems
- [ ] Revenue model involving Z-map data
- [ ] Data sharing with third parties (specific entities, not "partners")
- [ ] Impact on user experience if consent is withdrawn
- [ ] Explicit statement: "This consent is separate from your product subscription"

**Validity**: Must be renewable; "forever" consent is prohibited

---

### Category D: Sleep-Phase Operations

**Context**: Any Z-terminus modulation during sleep states  
**Consentor**: Subject (conscious-state verification required)

**Specific requirements**:
- [ ] Conscious-state confirmation before each sleep session
- [ ] No "always-on" or "auto-renew" options
- [ ] Subject can set pre-sleep termination triggers (arousal gesture, alarm, etc.)
- [ ] Morning debrief option: "This is what was modulated last night"
- [ ] Cannot be combined with commercial consent in a single form

**Rationale**: Sleep-phase modulation operates without conscious awareness or volitional control, at the physical boundary layer that shapes all subsequent experience.

---

### Category E: Inter-Individual Operations

**Context**: $\Gamma_{pair}$ measurement between two individuals  
**Consentor**: BOTH parties (bilateral consent)

**Specific requirements**:
- [ ] Both parties consent independently (not jointly, not in each other's presence)
- [ ] Neither party can unilaterally authorize relationship quantification
- [ ] Results disclosed to both parties if either requests
- [ ] Right to prevent future $\Gamma_{pair}$ measurement with the same counterpart
- [ ] Explicit notification: "This measures the physical similarity of your neural impedance patterns"

---

### Category F: Posthumous Operations

**Context**: Z-map reading or use after brain death  
**Consentor**: The deceased (via advance directive ONLY)

**Specific requirements**:
- [ ] Must be part of formal advance directive (comparable to organ donation)
- [ ] Next-of-kin cannot authorize alone (can execute pre-authorized directive)
- [ ] Time window must be specified (structural persistence window: ~hours post-mortem)
- [ ] Permitted uses must be enumerated (clinical only? memorial? commercial?)
- [ ] Duration of permitted posthumous Z-map use (not indefinite unless specified)
- [ ] Explicit opt-out of commercial exploitation unless separately authorized during life

---

## Consent Withdrawal

### Process

1. Subject notifies data controller of consent withdrawal
2. Data controller has **72 hours** to:
   - Cease all active Z-map operations
   - Initiate cryptographic destruction of stored Z-map
   - Terminate any derived applications using the Z-map
3. Confirmation of destruction provided to subject within **7 days**
4. Audit log entry created (cannot be deleted)

### Exceptions

- Z-map data already published in anonymized research cannot be retracted
- Z-map data required by law enforcement under valid warrant (jurisdiction-dependent)
- Z-map data in active clinical emergency (withdrawal effective after emergency resolves)

---

## Consent Form Template

```
═══════════════════════════════════════════════════════════
 IMPEDANCE BRIDGE — CONSENT FOR Z-MAP OPERATIONS
═══════════════════════════════════════════════════════════

Category: [ ] Clinical  [ ] Research  [ ] Commercial
          [ ] Sleep-Phase  [ ] Inter-Individual  [ ] Posthumous

I, _________________________, understand that:

1. A Z-map is a physical impedance fingerprint of my neural architecture
2. It captures the physical structure of signal transmission, NOT my thoughts
3. The Z-map will be stored for: _____ days (default: 30)
4. The Z-map will be used for: _________________________________
5. Access will be limited to: _________________________________
6. I may revoke this consent at any time by contacting: ___________

Specific disclosures for this category:
□ [Disclosure 1] _______________________________________________
□ [Disclosure 2] _______________________________________________
□ [Disclosure 3] _______________________________________________

I have read and understood the above.

Signature: _________________________  Date: ____________
Witness:  _________________________  Date: ____________

═══════════════════════════════════════════════════════════
```

---

## Audit Requirements

All consent-related events must be logged:

| Event | Required Data |
|-------|--------------|
| Consent granted | Subject ID, category, scope, timestamp, consentor identity |
| Z-map read | Subject ID, operator, purpose, equipment serial, timestamp |
| Z-map accessed | Accessor ID, purpose, timestamp |
| Consent renewed | Subject ID, new expiration, timestamp |
| Consent withdrawn | Subject ID, timestamp, destruction deadline |
| Z-map destroyed | Subject ID, method (cryptographic), verification, timestamp |
| Posthumous activation | Deceased ID, directive reference, activator ID, timestamp |
