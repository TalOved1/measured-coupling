"""Paired comparison of two lm-eval runs on the questions both answered: exact McNemar test and the paired
95% Wald interval of the accuracy difference A - B, as in the paper.
Usage: python scripts/paired.py <run_dir_A> <run_dir_B> [--filter flexible-extract]"""
import argparse, glob, json, math, os

ap = argparse.ArgumentParser()
ap.add_argument('a'); ap.add_argument('b')
ap.add_argument('--metric', default='auto', help="'auto': math_verify for MATH, else exact_match")
ap.add_argument('--filter', default='auto', help="'auto': flexible-extract if present, else none (MATH)")
ap.add_argument('--min_id', type=int, default=0, help='only questions with doc_id >= this (held-out part)')
args = ap.parse_args()


def load(d):
    out = {}
    if os.path.isdir(os.path.join(d, 'merged')):  # chunked run: use only the merged samples
        d = os.path.join(d, 'merged')
    rows = []
    for f in glob.glob(os.path.join(d, '**', 'samples_*.jsonl'), recursive=True):
        task = os.path.basename(f)[len('samples_'):].rsplit('_', 1)[0]
        rows += [(task, json.loads(line)) for line in open(f)]
    filt = args.filter
    if filt == 'auto':
        filt = 'flexible-extract' if any(r.get('filter') == 'flexible-extract' for _, r in rows) else 'none'
    metric = args.metric
    if metric == 'auto':
        metric = 'math_verify' if any('math_verify' in r for _, r in rows) else 'exact_match'
    for task, r in rows:
        if r.get('filter', filt) == filt:
            out[(task, int(r['doc_id']))] = float(r[metric])
    return out


A, B = load(args.a), load(args.b)
ids = sorted(i for i in set(A) & set(B) if i[1] >= args.min_id)
a_only = sum(A[i] > B[i] for i in ids)
b_only = sum(B[i] > A[i] for i in ids)
n = a_only + b_only
p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2 ** n) if n else 1.0
print(f"n={len(ids)}  A={100 * sum(A[i] for i in ids) / len(ids):.2f}  B={100 * sum(B[i] for i in ids) / len(ids):.2f}  "
      f"A-only={a_only}  B-only={b_only}  exact McNemar p={p:.4f}")
N = len(ids); d = (a_only - b_only) / N; se = math.sqrt(a_only + b_only - (a_only - b_only) ** 2 / N) / N
print(f"A - B = {100 * d:+.2f} points, 95% CI [{100 * (d - 1.96 * se):+.2f}, {100 * (d + 1.96 * se):+.2f}]")
