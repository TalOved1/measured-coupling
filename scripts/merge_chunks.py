"""Merge a run that was split into doc_range chunks (dirs named c<a>_<b>) into one run dir that
summarize.py and paired.py can read. Keeps each chunk's own questions only.
Usage: python scripts/merge_chunks.py results/llada_gsm8k/full/oracle_signed_t0.85_b20"""
import glob, json, os, sys

root = sys.argv[1]
by_key, nfe = {}, []
for d in sorted(glob.glob(os.path.join(root, 'c*_*'))):
    a, b = (int(v) for v in os.path.basename(d)[1:].split('_'))
    for f in glob.glob(os.path.join(d, '**', 'samples_*.jsonl'), recursive=True):
        task = os.path.basename(f)[len('samples_'):].rsplit('_', 1)[0]
        for line in open(f):
            r = json.loads(line)
            key = (task, int(r['doc_id']), r.get('filter'))
            generated = bool(r['resps'] and r['resps'][0] and r['resps'][0][0])
            # a chunk only generates its own requests; skipped ones have empty responses
            if key not in by_key or (generated and not by_key[key][1]):
                by_key[key] = (r, generated)
    for line in open(os.path.join(d, 'nfe.jsonl')):
        r = json.loads(line)
        if a <= r['idx'] < b:
            nfe.append(r)

tasks = sorted({k[0] for k in by_key})
out = os.path.join(root, 'merged')
os.makedirs(out, exist_ok=True)
results, n_total = {}, 0
metric_keys = ('exact_match', 'math_verify')
for task in tasks:
    rows = [v[0] for k, v in sorted(by_key.items()) if k[0] == task]
    with open(os.path.join(out, f'samples_{task}_merged.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    res = {}
    for filt in sorted({r.get('filter') for r in rows}):
        for mk in metric_keys:
            v = [float(r[mk]) for r in rows if r.get('filter') == filt and mk in r]
            if v:
                res[f'{mk},{filt}'] = sum(v) / len(v)
    n = len({int(r['doc_id']) for r in rows})
    res['n'] = n
    results[task] = res
    n_total += n
if len(tasks) > 1:  # micro average over the group, as lm-eval's weight_by_size
    grp = {}
    for mk in {k for t in tasks for k in results[t] if k != 'n'}:
        grp[mk] = sum(results[t][mk] * results[t]['n'] for t in tasks if mk in results[t]) / sum(results[t]['n'] for t in tasks if mk in results[t])
    results = {'group': grp, **results}
json.dump({'results': results, 'n-samples': {'all': {'effective': n_total}}}, open(os.path.join(out, 'results_merged.json'), 'w'))
with open(os.path.join(root, 'nfe.jsonl'), 'w') as f:
    for r in sorted(nfe, key=lambda r: r['idx']):
        f.write(json.dumps(r) + '\n')
with open(os.path.join(root, 'speed.jsonl'), 'w') as f:
    f.write(json.dumps({'Average NFE': sum(r['nfe'] for r in nfe) / len(nfe)}) + '\n')
print(f"merged {n_total} questions in {len(tasks)} task(s) from {root}: {results.get('group', results[tasks[0]])}, avg NFE {sum(r['nfe'] for r in nfe) / len(nfe):.1f}")
