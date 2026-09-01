"""
Semantic distractor branch: constrained-vocabulary masked-LM mask scoring ->
cosine-similarity percentile tiering.

Candidate generation strategy (revised 2026-08-31): instead of asking
the model for its unconstrained top-K fill-mask guesses and then throwing away
whatever isn't a CEFR-J word at the right level (which left some sentences
with zero surviving candidates -- the model's raw suggestions skew well above
A1/A2 vocabulary), we score the CEFR-J words directly. For every CEFR-J word
at the target POS/level (or an adjacent level), we read off the model's own
probability for that exact word at the masked position. This guarantees a
candidate pool bounded only by how many CEFR-J words exist at that POS/level
-- never zero just because the model's global top-K didn't happen to include
a simple word -- and every candidate is already at the learner's level, so
we're never tempted to widen the level window to compensate (a distractor
that's obviously too hard isn't a distractor, it's a giveaway).

Most short A1/A2 words are single tokens in the model's vocab and are scored
by that token's fill-mask probability. Words that split into k>1 subwords are
also scored (via a k-mask pass, length-normalized to the same [0, 1] scale) --
see score_cefr_candidates() for the approximation and its limits. Multi-*word*
CEFR-J entries (containing a space) are still skipped -- scoring a phrase as a
single blank fill isn't meaningful here.

IMPORTANT (2026-09-01): the default model_name below is roberta-large, not
microsoft/deberta-v3-large. deberta-v3-large was pretrained with ELECTRA-style
replaced-token-detection, not masked-language-modeling, so it has no trained
fill-mask head -- transformers silently bolts on a freshly RANDOM-initialized
cls.predictions.* head every time it's loaded (visible as a "this checkpoint
seem corrupted" / tied-weights warning at load time, which is NOT benign).
Every score_cefr_candidates()/semantic_distractors() call made with that
model was scoring against random noise, not real contextual fit. If you ever
change model_name, first confirm the model was actually pretrained with a
real MLM objective (BERT/RoBERTa family are safe; ELECTRA-style models are
not), and check the load report for MISSING head weights before trusting
any output.

Every stage of the funnel (full CEFR-J vocab -> POS/level-matched pool ->
top-K scored by the model -> candidates with a FastText vector -> tiered ->
final picks) is recorded in the returned `funnel` list and in `tiers_debug`
(now word + lm_score + cosine per candidate, not just bare words), so the
pool reduction can actually be studied instead of only seeing the end
result.

This has to run in YOUR native venv (the one with torch/transformers/network
access), not through the sandboxed device_bash environment -- that's why
this file is plain enough to need nothing beyond what's already in your
venv plus the standard library.

Usage (from the DiagnosticDistractors folder, with your venv active):
    python3 pipeline/semantic_branch.py

Writes results to pipeline/cache/semantic_branch_result.json so the rest of
the pipeline (running elsewhere) can pick them up.
"""
import json
import math
import os

from pipeline import cefr_lookup as cefr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_PATH = os.path.join(ROOT, "data", "fasttext", "pruned_cefr_j.vec")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "semantic_branch_result.json")

# One model/tokenizer load per process, reused across every row -- the old
# version reloaded a fresh pipeline() on every call, which is why the
# terminal log showed "Loading weights" repeated for ask/brush/buy.
_MODEL_CACHE = {}


def _get_model_and_tokenizer(model_name):
    if model_name not in _MODEL_CACHE:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


def load_pruned_vectors():
    vecs = {}
    with open(VEC_PATH, "r", encoding="utf-8") as f:
        next(f)  # header line: "<n_words> <dim>"
        for line in f:
            parts = line.rstrip().split(" ")
            vecs[parts[0]] = [float(x) for x in parts[1:]]
    return vecs


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_cefr_candidates(stem, target_word, target_pos, target_level,
                           allow_adjacent=True, top_n=50, model_name="roberta-large"):
    """
    Score every CEFR-J word at the target POS/level (or an adjacent level)
    for how well it fits the masked position in `stem`, using the model's own
    mask-prediction probability. Returns (candidates, pool_size) where
    candidates is the top `top_n` by lm_score and pool_size is how many
    CEFR-J words were eligible before truncating to top_n.

    Words that tokenize to a single subword are scored by that token's
    fill-mask probability directly. Words that tokenize to k>1 subwords (e.g.
    "apologize" -> ["apolog", "ize"]) are scored by a k-mask pass: the blank is
    replaced by k mask tokens and the score is the geometric mean of the k
    per-position marginal probabilities of the word's own subwords. The
    geometric mean is length-normalized so a single-token word's score is
    exactly its fill-mask probability (k=1 is the identity case), keeping
    single- and multi-token scores on one comparable [0, 1] scale. The
    approximation treats the k positions as independent -- it ignores
    intra-word dependence between subwords -- so multi-token scores are
    coarser than single-token ones; good enough to stop dropping these words
    entirely, not a substitute for true pseudo-log-likelihood scoring.
    """
    import torch
    import math

    tokenizer, model = _get_model_and_tokenizer(model_name)

    # One forward pass per distinct subword-length k, cached: replace the blank
    # with k mask tokens and return the softmax marginals at each mask position.
    _probs_by_k = {}
    def mask_marginals(k):
        if k not in _probs_by_k:
            filled = stem.replace("___", " ".join([tokenizer.mask_token] * k))
            enc = tokenizer(filled, return_tensors="pt")
            positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(positions) != k:
                _probs_by_k[k] = None  # couldn't place k masks cleanly
            else:
                with torch.no_grad():
                    logits = model(**enc).logits[0]
                _probs_by_k[k] = [torch.softmax(logits[p], dim=-1) for p in positions]
        return _probs_by_k[k]

    if mask_marginals(1) is None:
        raise ValueError(f"no mask token found after substitution for stem: {stem!r}")

    candidates = []
    # Pre-filtered, cached pool for this (pos, level) instead of rescanning
    # the whole CEFR-J vocabulary on every request (see cefr.candidate_pool).
    for w, level, pos in cefr.candidate_pool(target_pos, target_level,
                                             allow_adjacent=allow_adjacent):
        if w == target_word or not w.isalpha():
            continue
        # Tokenize as it would appear after a space (how it actually sits in
        # the sentence). Multi-word CEFR entries are already excluded by the
        # isalpha() check above; this handles single words that split into
        # multiple subwords.
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        marginals = mask_marginals(len(ids))
        if marginals is None:
            continue
        mean_logprob = sum(math.log(max(marginals[i][t].item(), 1e-45))
                           for i, t in enumerate(ids)) / len(ids)
        candidates.append({"word": w, "level": level, "pos": pos,
                           "lm_score": math.exp(mean_logprob)})

    candidates.sort(key=lambda c: c["lm_score"], reverse=True)
    return candidates[:top_n], len(candidates)


def semantic_distractors(stem, target_word, target_pos=None, target_level=None,
                          allow_adjacent=True, top_k=50, model_name="roberta-large"):
    target_word = target_word.lower().strip()
    if target_pos is None or target_level is None:
        auto = cefr.entries(target_word)
        if not auto:
            raise ValueError(f"'{target_word}' not in CEFR-J; pass target_pos/target_level explicitly")
        target_pos = target_pos or auto[0][0]
        target_level = target_level or auto[0][1]

    n_full_vocab = len(list(cefr.all_words()))
    filtered, n_pool = score_cefr_candidates(stem, target_word, target_pos, target_level,
                                              allow_adjacent=allow_adjacent, top_n=top_k,
                                              model_name=model_name)
    # top_candidates_debug: the model's own ranking, before FastText/cosine
    # touches anything -- useful to see whether the model's contextual fit and
    # the eventual cosine-based tiers agree or disagree.
    top_candidates_debug = [{"word": c["word"], "lm_score": c["lm_score"]} for c in filtered[:15]]

    vecs = load_pruned_vectors()
    target_vec = vecs.get(target_word)
    scored = []
    for c in filtered:
        v = vecs.get(c["word"])
        if target_vec is None or v is None:
            continue
        c["cosine"] = cosine(target_vec, v)
        scored.append(c)
    scored.sort(key=lambda c: c["cosine"], reverse=True)

    n = len(scored)
    # NOTE (found 2026-08-31 while building the funnel/export view): the
    # original boundaries left candidates between the 70th and 95th
    # percentile completely untracked -- neither picked nor even visible in
    # tiers_debug, ~25% of every scored pool just vanishing silently. That
    # band is real "too weak to be near_miss/thematic, not extreme enough to
    # be control" territory, so it's kept out of `picks` on purpose, but it
    # now lands in its own `unclassified` bucket instead of disappearing --
    # needed to actually study where the pool goes.
    tiers = {"discard": [], "near_miss": [], "thematic": [], "unclassified": [], "control": []}
    for i, c in enumerate(scored):
        pct = (i + 1) / n if n else 1.0  # 1/n = most similar, 1.0 = least similar
        if pct <= 0.10:
            tiers["discard"].append(c)
        elif pct <= 0.30:
            tiers["near_miss"].append(c)
        elif pct <= 0.70:
            tiers["thematic"].append(c)
        elif pct < 0.95:
            tiers["unclassified"].append(c)
        else:
            tiers["control"].append(c)

    picks = {}
    if tiers["near_miss"]:
        picks["near_miss"] = tiers["near_miss"][0]
    if tiers["thematic"]:
        picks["thematic"] = tiers["thematic"][len(tiers["thematic"]) // 2]
    if tiers["control"]:
        picks["control"] = tiers["control"][-1]

    funnel = [
        {"stage": "cefr_j_full_vocab", "count": n_full_vocab},
        {"stage": "matching_pos_and_level(±1)", "count": n_pool},
        {"stage": "scored_by_model_topk", "count": len(filtered)},
        {"stage": "has_fasttext_vector", "count": n},
        {"stage": "tiered", "counts": {k: len(v) for k, v in tiers.items()}},
        {"stage": "final_picks", "count": len(picks)},
    ]

    result = {
        "stem": stem, "target": target_word, "target_pos": target_pos, "target_level": target_level,
        "model": model_name,
        "n_cefr_candidate_pool": n_pool,      # every CEFR-J word at this POS/level (or adjacent)
        "n_scored_by_model": len(filtered),   # top-K of those, ranked by the model's mask probability
        "n_with_vectors": n,                  # of those, how many had FastText vectors for tiering
        "funnel": funnel,                     # the whole pool-reduction trace, stage by stage
        "top_candidates_debug": top_candidates_debug,  # the model's own top-15, pre-cosine
        "distractors": picks,
        "tiers_debug": {
            k: [{"word": c["word"], "lm_score": c["lm_score"], "cosine": round(c["cosine"], 4)} for c in v]
            for k, v in tiers.items()
        },
    }
    return result


import csv
import re

PILOT_CSV = os.path.join(ROOT, "data", "pilot", "sentencestudio-A1-20260831.csv")


# Irregular past / past-participle forms the suffix heuristic below can't
# derive (only the forms that differ from base and from a regular +ed/+s;
# verbs whose past equals the base -- cut, put, hit, let, read -- need no
# entry). This is still a curated table, not a morphological analyzer, but it
# now covers the common English irregular verbs a learner corpus is likely to
# use, not just the handful the original pilot happened to contain. British
# variants (learnt/burnt/dreamt) are included alongside the American ones.
#
# Caveat: a couple of forms are homographs of other verbs' base forms (e.g.
# "lay" is both the past of "lie" and the base of "lay"). make_stem only ever
# searches for the *target* word's forms in the target's own sentence, so this
# is harmless in practice, but be aware when reading a matched_surface_form.
IRREGULAR_FORMS = {
    "arise": {"arose", "arisen"}, "awake": {"awoke", "awoken"},
    "be": {"was", "were", "been"}, "bear": {"bore", "borne"},
    "beat": {"beaten"}, "become": {"became"}, "begin": {"began", "begun"},
    "bend": {"bent"}, "bet": {"bet"}, "bind": {"bound"}, "bite": {"bit", "bitten"},
    "bleed": {"bled"}, "blow": {"blew", "blown"}, "break": {"broke", "broken"},
    "bring": {"brought"}, "build": {"built"}, "burn": {"burnt"},
    "buy": {"bought"}, "catch": {"caught"}, "choose": {"chose", "chosen"},
    "come": {"came"}, "cost": {"cost"}, "cut": {"cut"}, "deal": {"dealt"},
    "dig": {"dug"}, "do": {"did", "done"}, "draw": {"drew", "drawn"},
    "dream": {"dreamt"}, "drink": {"drank", "drunk"}, "drive": {"drove", "driven"},
    "eat": {"ate", "eaten"}, "fall": {"fell", "fallen"}, "feed": {"fed"},
    "feel": {"felt"}, "fight": {"fought"}, "find": {"found"}, "fly": {"flew", "flown"},
    "forget": {"forgot", "forgotten"}, "forgive": {"forgave", "forgiven"},
    "freeze": {"froze", "frozen"}, "get": {"got", "gotten"}, "give": {"gave", "given"},
    "go": {"went", "gone"}, "grow": {"grew", "grown"}, "hang": {"hung"},
    "have": {"had"}, "hear": {"heard"}, "hide": {"hid", "hidden"}, "hit": {"hit"},
    "hold": {"held"}, "hurt": {"hurt"}, "keep": {"kept"}, "know": {"knew", "known"},
    "lay": {"laid"}, "lead": {"led"}, "learn": {"learnt"}, "leave": {"left"},
    "lend": {"lent"}, "let": {"let"}, "lie": {"lay", "lain"}, "light": {"lit"},
    "lose": {"lost"}, "make": {"made"}, "mean": {"meant"}, "meet": {"met"},
    "pay": {"paid"}, "put": {"put"}, "read": {"read"}, "ride": {"rode", "ridden"},
    "ring": {"rang", "rung"}, "rise": {"rose", "risen"}, "run": {"ran"},
    "say": {"said"}, "see": {"saw", "seen"}, "sell": {"sold"}, "send": {"sent"},
    "set": {"set"}, "shake": {"shook", "shaken"}, "shine": {"shone"},
    "shoot": {"shot"}, "show": {"showed", "shown"}, "shut": {"shut"},
    "sing": {"sang", "sung"}, "sink": {"sank", "sunk"}, "sit": {"sat"},
    "sleep": {"slept"}, "speak": {"spoke", "spoken"}, "spend": {"spent"},
    "spread": {"spread"}, "stand": {"stood"}, "steal": {"stole", "stolen"},
    "stick": {"stuck"}, "swim": {"swam", "swum"}, "swing": {"swung"},
    "take": {"took", "taken"}, "teach": {"taught"}, "tear": {"tore", "torn"},
    "tell": {"told"}, "think": {"thought"}, "throw": {"threw", "thrown"},
    "understand": {"understood"}, "wake": {"woke", "woken"}, "wear": {"wore", "worn"},
    "win": {"won"}, "write": {"wrote", "written"},
}


def inflections(word):
    """Heuristic surface forms for a base word -- good enough for A1-level
    regular verbs/nouns/adjectives, not a real morphological analyzer."""
    forms = {word} | IRREGULAR_FORMS.get(word, set())
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        forms |= {word[:-1] + "ies", word[:-1] + "ied"}
    else:
        forms |= {word + "s", word + "es", word + "ed"}
    if word.endswith("e"):
        forms |= {word[:-1] + "ing", word + "d"}
    else:
        forms.add(word + "ing")
        if len(word) >= 3 and word[-1] not in "aeiouwxy" and word[-2] in "aeiou" and word[-3] not in "aeiou":
            forms |= {word + word[-1] + "ing", word + word[-1] + "ed"}
    forms |= {word + "er", word + "est"}
    return forms


def make_stem(sentence, word):
    """Blank out the (possibly inflected) target word in `sentence`. Returns
    (stem_with___, matched_surface_form) or (None, None) if no form matched."""
    forms = inflections(word.lower())
    for m in re.finditer(r"[A-Za-z']+", sentence):
        token = m.group(0)
        if token.lower() in forms:
            return sentence[:m.start()] + "___" + sentence[m.end():], token
    return None, None


def load_pilot_rows(path=PILOT_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_pilot(n_words=3, path=PILOT_CSV):
    """First sentence for each of the first `n_words` distinct words in the
    pilot CSV -- a quick, varied smoke-test sample."""
    rows = load_pilot_rows(path)
    seen = set()
    sample = []
    for r in rows:
        if r["word"] not in seen:
            seen.add(r["word"])
            sample.append(r)
        if len(sample) >= n_words:
            break
    return sample


if __name__ == "__main__":
    results = []
    for row in sample_pilot(n_words=3):
        word, pos, level, sentence = row["word"], row["pos"], row["level"], row["sentence"]
        stem, matched = make_stem(sentence, word)
        print(f"\n{'='*60}\nword={word} ({pos}, {level})  sentence={sentence!r}")
        if stem is None:
            print(f"  could not locate '{word}' (or an inflection) in the sentence -- skipping")
            continue
        print(f"  stem: {stem}  (matched surface form: {matched!r})")
        result = semantic_distractors(stem, word, target_pos=pos, target_level=level)
        for stage in result["funnel"]:
            if "counts" in stage:
                print(f"  funnel: {stage['stage']}: {stage['counts']}")
            else:
                print(f"  funnel: {stage['stage']}: {stage['count']}")
        print(f"  distractors: {result['distractors']}")
        results.append({"row": row, "stem": stem, "matched": matched, "result": result})

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_PATH}")
