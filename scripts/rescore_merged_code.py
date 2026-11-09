"""Re-score a chunked MBPP run from its merged samples (merge_chunks.py output), one process, private metric cache.
Same scoring as rescore_code.py (MBPP: filtered response vs the test cases). Overwrites <run>/merged/ with
samples_mbpp_rescored.jsonl and results_merged.json, which summarize.py and paired.py read.
Usage: HF_ALLOW_CODE_EVAL=1 python scripts/rescore_merged_code.py <run_dir> [...]"""
import json, os, sys, tempfile
os.environ['HF_METRICS_CACHE'] = tempfile.mkdtemp(prefix='code_eval_')
import evaluate
code_eval = evaluate.load('code_eval')


def score(target, pred):
    res, _ = code_eval.compute(references=[target], predictions=[[pred]], k=[1])
    return float(res['pass@1'])


for d in sys.argv[1:]:
    src = os.path.join(d, 'merged', 'samples_mbpp_merged.jsonl')
    rows, old, new = [], [], []
    for line in open(src):
        r = json.loads(line)
        pred = r['filtered_resps'][0] if isinstance(r['filtered_resps'][0], str) else r['filtered_resps'][0][0]
        key = 'pass@1' if 'pass@1' in r else 'pass_at_1'
        s = score(r['target'], pred)
        old.append(float(r[key])); new.append(s)
        r['exact_match'] = s; r['filter'] = 'flexible-extract'
        rows.append(r)
    with open(os.path.join(d, 'merged', 'samples_mbpp_rescored.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    json.dump({'results': {'mbpp': {'exact_match,flexible-extract': sum(new) / len(new)}},
               'n-samples': {'mbpp': {'effective': len(new)}}}, open(os.path.join(d, 'merged', 'results_merged.json'), 'w'))
    print(f"{d}: n={len(new)} pass@1 logged={100 * sum(old) / len(old):.2f} rescored={100 * sum(new) / len(new):.2f} changed={sum(a != b for a, b in zip(old, new))}")
