# scripts/debug/

Incident-specific diagnostic scripts. Not part of the reproducible pipeline
and not imported by anything — kept only as institutional memory.

## verify_roberta_fix.py

From the 2026-09-01 incident where the semantic branch defaulted to
`microsoft/deberta-v3-large`, an ELECTRA-pretrained checkpoint with no trained
masked-LM head. `transformers` silently attached a **randomly-initialized**
fill-mask head on every load, so semantic-branch scores were noise. The fix was
switching the default to `roberta-large` (a real MLM checkpoint).

This script reproduces the check that caught it. Run it as **two separate
process launches** and diff the output:

```
python3 scripts/debug/verify_roberta_fix.py > /tmp/run1.txt
python3 scripts/debug/verify_roberta_fix.py > /tmp/run2.txt
diff /tmp/run1.txt /tmp/run2.txt
```

Identical output (empty diff) = the checkpoint's own MLM head loaded and is
deterministic across restarts, as a real MLM checkpoint should be. A
random-head checkpoint fixes its head per-process, so the bug only shows up
across restarts — running twice in one process would hide it.

A companion script, `diagnose_drift.py`, was deleted after the incident: it
compared live output against a hardcoded baseline that turned out to be the
broken deberta values, making it actively misleading once the fix landed.
