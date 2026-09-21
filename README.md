# musemem-ex

Cognitive-science-driven persistent memory for LLM agents.

## Overview

musemem-ex gives LLM agents human-inspired memory: it stores conversation evidence, consolidates durable facts, retrieves relevant context, and forgets gracefully — all grounded in established cognitive science, neuroscience, and psychology models of how human memory works.

The system is organized around three principles:

1. **Memory is layered** — raw evidence, semantic facts, and procedural knowledge have different lifespans and retrieval dynamics
2. **Forgetting is functional** — decay is not data loss; it's a retrieval filter that keeps the active set relevant
3. **Retrieval is associative** — semantic similarity, temporal context, and spreading activation combine to surface what's relevant

## Architecture

### Three-Layer Memory Model

| Layer | Content | Scope | Decay | Purpose |
|-------|---------|-------|-------|---------|
| **L0 Evidence** | Raw conversation turns | Session | Turn-based (Ebbinghaus) | Immutable record of what was said |
| **L1 Semantic** | Distilled facts, promoted evidence | Cross-session | Supersession only | Durable knowledge extracted from evidence |
| **L2 Procedural** | Reusable patterns, procedures | Long-term | Never | How-to knowledge for recurring tasks |

### Retrieval Pipeline

```
Query
  → Lexical Scoring (BM25-like token overlap)
  → Semantic Scoring (full-pool embedding cosine)
  → Max-Blend (max of lexical and semantic)
  → Temporal Weighting (encoding specificity)
  → Persona Injection (user profile context)
  → Pattern Separation (discriminative contrast)
  → Threshold Filter → Ranked Results
```

### Memory Lifecycle

```
Add → Evidence (L0) → Consolidate (L0→L1) → Dream (offline) → Archive (decay)
  ↓         ↓               ↓                  ↓
Search ← Rank ← Retrieve ← Activate ← Consolidate
```

## Cognitive Science Foundations

### Core Memory Models

| Mechanism | Implementation | Reference |
|-----------|---------------|-----------|
| **Multi-Store Model** | L0 (sensory/short-term) → L1 (long-term) → L2 (procedural) | Atkinson & Shiffrin (1968) — *Human Memory: A Proposed System* |
| **ACT-R Base-Level Activation** | Power-law decay over access history; O(1) incremental update (Petrov optimization) | Anderson et al. (2004) — *An Integrated Theory of the Mind*; Petrov (2006) |
| **New Theory of Disuse** | Storage Strength (monotonic, non-decreasing) vs Retrieval Strength (volatile, decaying) | Bjork & Bjork (1992) — *A New Theory of Disuse* |
| **SAM Retrieval Model** | Strongest available cue drives retrieval; context-dependent activation | Raaijmakers & Shiffrin (1981) — *Search of Associative Memory* |
| **Working Memory Capacity** | Injection budget limits active context; Cowan 4±1 chunks for immediate buffer | Cowan (2001) — *The Magical Number 4*; Miller (1956) — *The Magical Number Seven* |
| **Levels of Processing** | L0 (shallow/verbatim) → L1 (semantic) → L2 (procedural/deep) | Craik & Lockhart (1972) — *Levels of Processing* |

### Encoding and Retrieval

| Mechanism | Implementation | Reference |
|-----------|---------------|-----------|
| **Encoding Specificity** | Temporal metadata indexed with each record; time-based retrieval cues | Tulving & Thomson (1973) — *Encoding Specificity and Retrieval Processes* |
| **Spreading Activation** | Semantic similarity propagates through memory network; passive emergence surfaces related memories | Collins & Loftus (1975) — *A Spreading-Activation Theory of Semantic Processing* |
| **Reconstructive Memory** | Fact distillation stores gist, not verbatim record; schema-driven reconstruction | Bartlett (1932) — *Remembering* |
| **Transfer-Appropriate Processing** | Retrieval format matches encoding format (XML structured context) | Morris, Bransford & Franks (1977) |
| **Context-Dependent Memory** | Session-scoped evidence; environmental context as retrieval cue | Godden & Baddeley (1975) — *Context-Dependent Memory in Two Natural Environments* |
| **Testing Effect** | Retrieval practice (accessLog) strengthens memory traces | Roediger & Karpicke (2006) — *Test-Enhanced Learning* |
| **Spacing Effect** | Spaced repetition via retrieval strength boost on access | Cepeda, Pashler, Vul, Wixted & Rohrer (2006) — *Distributed Practice* |

### Forgetting and Consolidation

| Mechanism | Implementation | Reference |
|-----------|---------------|-----------|
| **Ebbinghaus Forgetting Curve** | RS decays per-turn; archived but never deleted (decay without loss) | Ebbinghaus (1885) — *Über das Gedächtnis* |
| **Memory Consolidation** | Offline promotion of repeated evidence to semantic facts (hippocampal → neocortical) | Diekelmann & Born (2010) — *The Memory Function of Sleep* |
| **Systems Consolidation** | L0 evidence → L1 facts → L2 procedures (hippocampal → neocortical transfer) | Squire & Alvarez (1995) — *Retrograde Amnesia and Memory Consolidation* |
| **Interference Theory** | Newer records supersede outdated ones via explicit links; proactive interference resolution | Wixted (2004) — *The Psychology and Neuroscience of Forgetting* |
| **Pattern Separation** | Confusable memories annotated with discriminative tokens (dentate gyrus analogue) | O'Reilly & Norman (2002) — *Hippocampal and Neocortical Contributions to Memory*; Treves & Rolls (1994) |
| **Synaptic Pruning** | Dream pass archives low-RS records; maintains storage efficiency | Huttenlocher (1979) — *Synaptic Density in Human Frontal Cortex* |
| **Fuzzy-Trace Theory** | Aggregate cards pre-compute gist for counting/aggregation questions | Brainerd & Reyna (1990) — *Gist is the Grist* |

### Social and Metacognitive

| Mechanism | Implementation | Reference |
|-----------|---------------|-----------|
| **Impression Formation** | User profile inferred from behavioral frequency patterns, not explicit statements | Asch (1946) — *Forming Impressions of Personality* |
| **Spontaneous Trait Inference** | Traits inferred from repeated behavioral patterns; no explicit labeling needed | Uleman & Moskowitz (1994) — *Spontaneous Trait Inferences* |
| **Zeigarnik Effect** | Blocked tasks and open loops stay in active memory until resolved | Zeigarnik (1927) — *Über das Behalten von erledigten und unerledigten Handlungen* |
| **Primacy Effect** | Session's initial goal preserved as high-strength supersession chain | Murdock (1962) — *The Serial Position Effect of Free Recall* |
| **Metamemory** | Memory leads provide metacognitive cues about what's stored; not instructions | Nelson & Narens (1990) — *Metamemory: A Theoretical Framework* |
| **Prospective Memory** | Blocked todos surface as reminders at contextually relevant moments | Einstein & McDaniel (1990) — *Normal Aging and Prospective Memory* |

### Neuroscience Foundations

| Mechanism | Implementation | Reference |
|-----------|---------------|-----------|
| **Hippocampal-Neocortical Dialogue** | L0 (hippocampal, fast-learning) → L1/L2 (neocortical, slow-learning) | McClelland, McNaughton & O'Reilly (1995) — *Why There Are Complementary Learning Systems* |
| **Dentate Gyrus Pattern Separation** | Contrast rendering orthogonalizes similar inputs | Treves & Rolls (1994) — *Computational Analysis of the Role of the Hippocampus in Memory* |
| **Prefrontal Working Memory** | Working state snapshot preserves active goal, decisions, constraints | Goldman-Rakic (1995) — *Cellular Basis of Working Memory* |
| **Default Mode Network** | Dream pass consolidates during idle periods | Raichle et al. (2001) — *A Default Mode of Brain Function* |
| **Autobiographical Memory** | Persona construction from behavioral patterns | Conway & Pleydell-Pearce (2000) — *The Construction of Autobiographical Memories* |

## Service Modules

| Module | Function | Cognitive Basis |
|--------|----------|-----------------|
| `context-builder.ts` | Per-turn injection assembly | Working memory buffer (Cowan, 2001; Baddeley, 2000) |
| `ranker.ts` | Hybrid lexical-semantic ranking | SAM model (Raaijmakers & Shiffrin, 1981) |
| `clock.ts` | ACT-R activation, Ebbinghaus decay | Anderson et al. (2004); Bjork & Bjork (1992); Petrov (2006) |
| `consolidate.ts` | L0→L1 promotion | Systems consolidation (Squire & Alvarez, 1995) |
| `dream.ts` | Offline consolidation | Sleep-dependent consolidation (Diekelmann & Born, 2010) |
| `persona.ts` | User profile construction | Impression formation (Asch, 1946); spontaneous trait inference (Uleman & Moskowitz, 1994) |
| `contrast.ts` | Discriminative contrast rendering | Pattern separation (O'Reilly & Norman, 2002; Treves & Rolls, 1994) |
| `distill.ts` | Write-time fact extraction | Reconstructive memory (Bartlett, 1932) |
| `emergence.ts` | Passive spreading activation | Collins & Loftus (1975) |
| `aggregate.ts` | Cross-episode aggregation | Fuzzy-trace theory (Brainerd & Reyna, 1990) |
| `primacy.ts` | First-goal preservation | Primacy effect (Murdock, 1962) |
| `procedural.ts` | Procedural pattern detection | Procedural memory (Cohen & Squire, 1980) |
| `working-state.ts` | Structured compaction snapshot | Working memory (Baddeley, 2000); prefrontal cortex (Goldman-Rakic, 1995) |
| `memory-leads.ts` | Metacognitive cue layer | Metamemory (Nelson & Narens, 1990) |
| `counting-aid.ts` | Counting question pre-aggregation | Working memory offloading (Miller, 1956) |
| `xml-format.ts` | Structured context formatting | External memory aids; transfer-appropriate processing |
| `temporal.ts` | Time-aware retrieval | Encoding specificity (Tulving & Thomson, 1973) |

## API

### POST /add

Store conversation turns.

```json
{
  "request_id": "add:0",
  "user_id": "user123",
  "session_id": "session456",
  "messages": [{"role": "user", "content": "..."}]
}
```

### POST /search

Retrieve relevant memories.

```json
{
  "query": "What degree did I graduate with?",
  "user_id": "user123",
  "top_k": 100
}
```

## Quick Start

```bash
npm install
node --experimental-strip-types aml/server.ts
```

## Architecture Documentation

- [Overview](architecture/overview.md) — System design and scope model
- [Retrieval Pipeline](architecture/retrieval-pipeline.md) — Scoring, blending, and ranking
- [Memory Lifecycle](architecture/memory-lifecycle.md) — Creation, consolidation, and decay

## License

[MIT](LICENSE)
