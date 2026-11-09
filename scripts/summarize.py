"""Collect lm-eval accuracy and DAWN speed stats into one table.
Usage: python scripts/summarize.py results/llada_gsm8k/smoke [--ref baseline_b32]"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--ref', default='baseline_b32', help='run whose avg NFE / time is the 1.0x reference')
args = ap.parse_args()

rows = {}
for d in sorted(glob.glob(os.path.join(args.root, '*'))):
    name = os.path.basename(d)
    merged = os.path.join(d, 'merged', 'results_merged.json')  # chunked runs (scripts/merge_chunks.py)
    res = [merged] if os.path.exists(merged) else sorted(glob.glob(os.path.join(d, '**', 'results_*.json'), recursive=True))
    spd = os.path.join(d, 'speed.jsonl')
    acc = n = None
    if res:
        r = json.load(open(res[-1]))
        res_ = r['results']
        m = res_.get('group') or res_.get('minerva_math') or next(iter(res_.values()))
        # MATH (minerva) answers are scored with math_verify: instruct models answer in \\boxed{}, which
        # minerva's exact_match pattern ("Final Answer: ...") does not read.
        acc = m.get('math_verify,none', m.get('exact_match,flexible-extract', m.get('exact_match,strict-match', m.get('exact_match,none'))))
        if acc is None:  # single-task runs keep their metrics under the task name
            for task, mm in res_.items():
                acc = mm.get('exact_match,flexible-extract', mm.get('exact_match,strict-match', mm.get('exact_match,none')))
        ns = r.get('n-samples', {})
        n = sum(v.get('effective', 0) for v in ns.values()) if ns else None
    s = json.loads(open(spd).read().strip().splitlines()[-1]) if os.path.exists(spd) else {}
    rows[name] = dict(acc=acc, n=n, nfe=s.get('Average NFE'), tpn=s.get('Tokens per NFE'),
                      tps=s.get('Tokens per Second'), time=s.get('Total Time Taken'))

ref = rows.get(args.ref, {})
fmt = lambda v, f: '-' if v is None else format(v, f)
print(f"{'run':28s} {'n':>5s} {'acc':>6s} {'avgNFE':>7s} {'tok/NFE':>7s} {'TPS':>7s} {'NFE-speedup':>11s} {'TPS-speedup':>11s}")
for name, r in rows.items():
    sp_nfe = ref['nfe'] / r['nfe'] if ref.get('nfe') and r['nfe'] else None
    sp_tps = r['tps'] / ref['tps'] if ref.get('tps') and r['tps'] else None
    acc = None if r['acc'] is None else 100 * r['acc']
    print(f"{name:28s} {fmt(r['n'], 'd'):>5s} {fmt(acc, '.2f'):>6s} {fmt(r['nfe'], '.1f'):>7s} "
          f"{fmt(r['tpn'], '.2f'):>7s} {fmt(r['tps'], '.1f'):>7s} {fmt(sp_nfe, '.2f'):>11s} {fmt(sp_tps, '.2f'):>11s}")
