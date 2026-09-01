"""
Verifies the roberta-large swap actually fixed the "randomly-initialized
head" problem: run this TWICE, as two SEPARATE `python3` process launches
(not twice in one script -- that would hide the bug, since a random head is
fixed for the life of one process and only reveals itself across restarts).

    python3 -m scripts.debug.verify_roberta_fix > /tmp/run1.txt
    python3 -m scripts.debug.verify_roberta_fix > /tmp/run2.txt
    diff /tmp/run1.txt /tmp/run2.txt

No output from diff = the model's own head loaded correctly and is
deterministic across restarts, as expected for a real MLM checkpoint.
"""
from pipeline import semantic_branch as sb

stem = "Did you ___ the price?"
candidates, pool_size = sb.score_cefr_candidates(stem, "ask", "verb", "A1", top_n=15)
print(f"pool_size={pool_size}")
for c in candidates:
    print(f"  {c['word']:12s} {c['lm_score']:.8f}")
