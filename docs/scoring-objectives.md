# Reconciling the two branches' scoring objectives

The spelling branch and the semantic branch look superficially similar — both
draw candidates from the CEFR-J wordlist, both gate on the learner's level,
both emit a ranked list of distractors — but they optimize for **different and
deliberately independent** notions of what makes a wrong answer tempting. This
note records why they are kept separate and never blended into a single score.

## Two different failure modes

A multiple-choice distractor works when a learner who does *not* know the
answer is pulled toward it. There are two unrelated reasons that happens:

| | Spelling / phonetic branch | Semantic branch |
|---|---|---|
| **Optimizes for** | surface-form confusability | contextual / semantic plausibility |
| **The learner is fooled because…** | the word *looks or sounds* like the target (or like the same katakana reading) | the word *fits the sentence's meaning* and could plausibly go in the blank |
| **Primary signal** | orthographic edit distance, katakana-reading collision, rule-based sound swaps (L/R, B/V, TH→S/Z, F/H) | masked-LM fill-probability for the blanked position |
| **Needs the sentence?** | no — operates on the word form alone | yes — the sentence *is* the context being scored |
| **A good distractor is often…** | semantically *unrelated* to the target (quiet/quite, bus/bath) | semantically *near* the target, or at least topically adjacent |

The key point is that the two objectives are close to **orthogonal**. `quite`
is an excellent spelling distractor for `quiet` and a terrible semantic one
(it doesn't fit "Please be ___ in the library"). `whisper` might be a decent
semantic distractor for `quiet` and a useless spelling one (nothing about it
looks or sounds like the target). Optimizing hard for one axis says almost
nothing about the other.

## Why they are not combined into one score

Because the axes measure different things, there is no meaningful common unit
in which to average an edit distance against a fill-mask probability. Blending
them into a single ranking would:

- **suppress the strongest distractors of each kind** — a pure surface
  confusion scores ~0 on contextual fit and vice versa, so any weighted sum
  penalizes exactly the candidates each branch exists to find; and
- **hide *why* a distractor was chosen** from the human reviewer, who wants to
  know whether an item is testing form discrimination or meaning
  discrimination.

So the branches run independently and their outputs are **merged as a union**
for review, not fused into one score. A candidate that happens to surface in
both branches is a legitimate double hit (dedupe at merge), not evidence that
the scores should have been combined.

## What they *do* share

One constraint, and only one: the **CEFR-J level window**. Both branches only
ever draw from CEFR-J words at the target's level, ±1 as a fallback (see
`cefr.matches` / `cefr.candidate_pool`). A distractor above the learner's level
is not a distractor, it's a giveaway — so neither branch widens the level
window to rescue a low-yield sentence. This shared gate is a *pedagogical*
constraint, not a scoring objective; it bounds the candidate universe both
branches optimize *within*, and is the reason their outputs can be pooled at
all.

## Internal spread within the semantic branch

The semantic branch has a second-stage objective the spelling branch has no
equivalent for. After fill-mask scoring, FastText cosine similarity sorts the
survivors into tiers (near-miss / thematic / control) so the three chosen
distractors are a *spread* of semantic distances rather than three
near-synonyms. This is still the "contextual/semantic plausibility" objective —
it just recognizes that three equally-plausible near-synonyms make a worse item
than one near-miss, one thematically-related word, and one plausible-but-clearly-
distinct control. The spelling branch's analogue is simply its four
distinct signals feeding one pool in priority order.
