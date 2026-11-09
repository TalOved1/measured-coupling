"""Paired accuracy difference (ours - MF) with a 95% confidence interval, for every model and benchmark in the
main tables. Uses the same per-question results as the p-values (scripts/paired.py). For paired binary outcomes
with b = ours-only correct and c = MF-only correct out of n, the difference is (b - c) / n with standard error
sqrt(b + c - (b - c)^2 / n) / n (the paired Wald interval).
Usage: python scripts/paired_ci.py"""
import json, math, os, re, subprocess, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
PAIRS = [  # (model, benchmark, ours, MF): the runs behind Tables 1, 2 and 3
    ('LLaDA-8B', 'GSM8K', 'llada_gsm8k/full/fusedm_signed_k12_t0.9_b20', 'llada_gsm8k/full/mf_t0.85_b20'),
    ('LLaDA-8B', 'MATH', 'llada_math/full/fusedm_signed_k12_t0.9_b20', 'llada_math/full/mf_t0.85_b20'),
    ('LLaDA-8B', 'HumanEval', 'llada_humaneval/speed_all4/ours', 'llada_humaneval/speed_all4/mf_t0.85_b20'),
    ('LLaDA-8B', 'MBPP', 'llada_mbpp/speed_all4/ours', 'llada_mbpp/speed_all4/mf_t0.85_b20'),
    ('LLaDA-1.5', 'GSM8K', 'llada15_gsm8k/full/fusedm_signed_k12_t0.9_b20', 'llada15_gsm8k/full/mf_t0.85_b20'),
    ('LLaDA-1.5', 'MATH', 'llada15_math/full/fusedm_signed_k12_t0.9_b20', 'llada15_math/full/mf_t0.85_b20'),
    ('LLaDA-1.5', 'HumanEval', 'llada15_humaneval/speed_all3/ours_first', 'llada15_humaneval/speed_all3/mf_t0.85_b20'),
    ('LLaDA-1.5', 'MBPP', 'llada15_mbpp/speed_all2/ours_first', 'llada15_mbpp/speed_all2/mf_t0.85_b20'),
    ('Dream', 'GSM8K', 'dream_gsm8k/full/fusedm_signed_k20_t0.95_b20', 'dream_gsm8k/full/mf_t0.95_b20'),
    ('Dream', 'MATH', 'dream_math/full/fusedm_signed_k20_t0.95_b20', 'dream_math/full/mf_t0.95_b20'),
    ('Dream', 'HumanEval', 'dream_humaneval/speed_all2/ours_first', 'dream_humaneval/speed_all2/mf_t0.95_b20'),
    ('Dream', 'MBPP', 'dream_mbpp/speed_all2/ours_first', 'dream_mbpp/speed_all2/mf_t0.95_b20'),
]
py = sys.executable
out = []
for model, bench, a, b in PAIRS:
    txt = subprocess.run([py, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paired.py'), os.path.join(R, a), os.path.join(R, b)],
                         capture_output=True, text=True).stdout
    m = re.search(r'n=(\d+)\s+A=([\d.]+)\s+B=([\d.]+)\s+A-only=(\d+)\s+B-only=(\d+)\s+exact McNemar p=([\d.e-]+)', txt)
    n, accA, accB, bo, co, p = int(m[1]), float(m[2]), float(m[3]), int(m[4]), int(m[5]), float(m[6])
    d = (bo - co) / n
    se = math.sqrt(bo + co - (bo - co) ** 2 / n) / n
    lo, hi = 100 * (d - 1.96 * se), 100 * (d + 1.96 * se)
    out.append(dict(model=model, bench=bench, n=n, ours=accA, mf=accB, diff=round(100 * d, 2), lo=round(lo, 2), hi=round(hi, 2), p=p))
    print(f'{model:10s} {bench:9s} n={n:5d}  ours {accA:5.2f}  MF {accB:5.2f}  diff {100*d:+5.2f}  95% CI [{lo:+5.2f}, {hi:+5.2f}]  p={p:.3g}')
json.dump(out, open(os.path.join(R, 'paired_ci.json'), 'w'), indent=1)
