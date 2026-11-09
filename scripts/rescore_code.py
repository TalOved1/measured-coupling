"""Re-score code runs (HumanEval / MBPP) from their logged samples, one process, private metric cache.
Needed because `evaluate`'s code_eval writes a temp file at a fixed path under HF_HOME/metrics, so
concurrent jobs on the shared filesystem could clobber each other (3 crashes; silent mix-ups possible).
HumanEval uses DAWN's sanitize post-processing (the paper-level numbers come from it).
Writes <run>/merged/samples_<task>_rescored.jsonl and results_merged.json (read by summarize/paired).
Usage: HF_ALLOW_CODE_EVAL=1 python scripts/rescore_code.py results/llada_mbpp/full ..."""
import glob, json, os, sys, tempfile
os.environ['HF_METRICS_CACHE'] = tempfile.mkdtemp(prefix='code_eval_')
import evaluate
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'measured_coupling', 'llada'))
from sanitize import sanitize  # DAWN's (from Dream) HumanEval post-processing

code_eval = evaluate.load('code_eval')


def score(reference, prediction):
    res, _ = code_eval.compute(references=[reference], predictions=[[prediction]], k=[1], num_workers=4, timeout=5.0)
    return float(res['pass@1'])


for root in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        files = [f for f in glob.glob(os.path.join(d, '**', 'samples_*.jsonl'), recursive=True) if '/merged/' not in f]
        if not files:
            continue
        task = os.path.basename(files[0]).split('_')[1]
        rows, old, new = [], [], []
        for line in open(sorted(files)[-1]):
            r = json.loads(line)
            if task == 'humaneval':
                # As DAWN's postprocess_code_humaneval.py: take the ```python block of the chat answer,
                # append it to the prompt, and sanitize down to the entry-point function.
                body = r['resps'][0][0].split('```python\n', 1)[-1].split('```')[0]
                pred = sanitize(r['doc']['prompt'] + '\n' + body, r['doc']['entry_point'])
            else:
                pred = r['filtered_resps'][0] if isinstance(r['filtered_resps'][0], str) else r['filtered_resps'][0][0]
            key = 'pass@1' if 'pass@1' in r else 'pass_at_1'
            s = score(r['target'], pred)
            old.append(float(r[key])); new.append(s)
            r['exact_match'] = s  # paired.py reads this field
            r['filter'] = 'flexible-extract'
            rows.append(r)
        out = os.path.join(d, 'merged')
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f'samples_{task}_rescored.jsonl'), 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        json.dump({'results': {task: {'exact_match,flexible-extract': sum(new) / len(new)}},
                   'n-samples': {task: {'effective': len(new)}}}, open(os.path.join(out, 'results_merged.json'), 'w'))
        changed = sum(a != b for a, b in zip(old, new))
        print(f"{d}: n={len(new)} pass@1 logged={100 * sum(old) / len(old):.2f} rescored={100 * sum(new) / len(new):.2f} changed={changed}")
