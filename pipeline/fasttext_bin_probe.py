"""
Low-memory probe into Facebook's cc.en.300.bin.gz WITHOUT decompressing the
whole ~7GB file to disk (the local VM has 3.8GB RAM and no swap; a normal
gensim/fasttext load would try to materialize a (nwords+bucket) x dim float32
matrix, which alone is >4.8GB and would likely OOM).

Streams the gzip stream once with Python's gzip module, parses the header +
vocab (format verified against gensim's _fasttext_bin.py), then keeps reading
row-by-row through the vectors_ngrams matrix, keeping ONLY the rows that
belong to a small set of wanted words (their own dictionary row + their
character n-gram hash-bucket rows), discarding everything else without
holding it in memory. This never allocates the giant matrix.
"""
import gzip
import struct
import sys
import time

PATH = "/root_placeholder"  # set by caller
DIM_HINT = None

def read_struct(f, fmt):
    n = struct.calcsize(fmt)
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"expected {n} bytes, got {len(data)}")
    return struct.unpack(fmt, data)

FNV_OFFSET = 2166136261
FNV_PRIME = 16777619
MASK32 = 0xFFFFFFFF

def fnv_hash(s: bytes) -> int:
    h = FNV_OFFSET
    for b in s:
        h = (h ^ b) & MASK32
        h = (h * FNV_PRIME) & MASK32
    return h

def word_ngram_hashes(word, minn, maxn, bucket):
    w = "<" + word + ">"
    grams = set()
    L = len(w)
    for n in range(minn, maxn + 1):
        if n > L:
            break
        for start in range(0, L - n + 1):
            grams.add(w[start:start + n])
    return {fnv_hash(g.encode("utf-8")) % bucket for g in grams}

def probe(path, wanted_words, time_budget_s=38):
    t0 = time.time()
    wanted = {w.lower() for w in wanted_words}
    f = gzip.open(path, "rb")

    magic, version = read_struct(f, "@2i")
    new_format = magic == 793712314
    assert new_format, f"unexpected magic {magic}"
    names = ["dim", "ws", "epoch", "min_count", "neg", "word_ngrams",
             "loss", "model", "bucket", "minn", "maxn", "lr_update_rate"]
    header = {}
    for name in names:
        header[name] = read_struct(f, "@i")[0]
    header["t"] = read_struct(f, "@d")[0]
    dim, bucket, minn, maxn = header["dim"], header["bucket"], header["minn"], header["maxn"]
    print(f"[{time.time()-t0:.1f}s] header: dim={dim} bucket={bucket} minn={minn} maxn={maxn}", file=sys.stderr)

    vocab_size, nwords, nlabels = read_struct(f, "@3i")
    assert nlabels == 0, "supervised model, not supported"
    ntokens = read_struct(f, "@q")[0]
    pruneidx_size = read_struct(f, "@q")[0]
    print(f"[{time.time()-t0:.1f}s] vocab_size={vocab_size} nwords={nwords} pruneidx_size={pruneidx_size}", file=sys.stderr)

    word_to_idx = {}
    for i in range(vocab_size):
        buf = bytearray()
        b = f.read(1)
        while b != b"\x00":
            buf += b
            b = f.read(1)
        w = buf.decode("utf-8", errors="replace")
        read_struct(f, "@qb")  # count, entry_type
        if i < nwords and w.lower() in wanted:
            word_to_idx[w.lower()] = i
    for _ in range(pruneidx_size):
        read_struct(f, "@2i")
    print(f"[{time.time()-t0:.1f}s] vocab scanned, found {len(word_to_idx)}/{len(wanted)} wanted words in-vocab",
          file=sys.stderr)

    # figure out every matrix row index we'll need, for every wanted word
    need_rows = {}  # row_idx -> list of words that need it
    for w in wanted:
        rows = []
        if w in word_to_idx:
            rows.append(word_to_idx[w])
        for h in word_ngram_hashes(w, minn, maxn, bucket):
            rows.append(nwords + h)
        for r in rows:
            need_rows.setdefault(r, []).append(w)
    max_needed = max(need_rows) if need_rows else -1
    print(f"[{time.time()-t0:.1f}s] need {len(need_rows)} distinct rows, max row idx {max_needed}", file=sys.stderr)

    quant_input = read_struct(f, "@?")[0]
    num_vectors, mdim = read_struct(f, "@2q")
    assert mdim == dim
    print(f"[{time.time()-t0:.1f}s] matrix: {num_vectors} rows x {mdim} dim, quant_input={quant_input}",
          file=sys.stderr)

    row_bytes = dim * 4
    collected = {}  # row_idx -> tuple of floats
    row_idx = 0
    while row_idx <= max_needed and row_idx < num_vectors:
        chunk = f.read(row_bytes)
        if len(chunk) != row_bytes:
            break
        if row_idx in need_rows:
            collected[row_idx] = struct.unpack(f"@{dim}f", chunk)
        row_idx += 1
        if row_idx % 500000 == 0:
            elapsed = time.time() - t0
            print(f"[{elapsed:.1f}s] scanned {row_idx}/{max_needed} rows, collected {len(collected)}/{len(need_rows)}",
                  file=sys.stderr)
            if elapsed > time_budget_s:
                print(f"[{elapsed:.1f}s] TIME BUDGET HIT at row {row_idx}, stopping early", file=sys.stderr)
                break
    f.close()

    vectors = {}
    for w in wanted:
        rows = []
        if w in word_to_idx and word_to_idx[w] in collected:
            rows.append(collected[word_to_idx[w]])
        for h in word_ngram_hashes(w, minn, maxn, bucket):
            r = nwords + h
            if r in collected:
                rows.append(collected[r])
        if rows:
            avg = [sum(vals) / len(vals) for vals in zip(*rows)]
            vectors[w] = (avg, len(rows))
    print(f"[{time.time()-t0:.1f}s] done. resolved vectors for {len(vectors)}/{len(wanted)} words", file=sys.stderr)
    return header, vectors

if __name__ == "__main__":
    path = sys.argv[1]
    words = sys.argv[2:]
    header, vectors = probe(path, words)
    for w in words:
        w = w.lower()
        if w in vectors:
            vec, n_components = vectors[w]
            print(f"{w}: {n_components} components, first 5 dims = {[round(x,4) for x in vec[:5]]}")
        else:
            print(f"{w}: NOT RESOLVED")
