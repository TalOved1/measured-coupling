"""Run any experiment of the paper by its name in configs/runs.json (model, task, shots and decoder settings).
  python run.py --list [filter]          list run names (e.g. --list llada_gsm8k/full)
  python run.py <run_name> [--limit N]   run it; results go to results/<run_name>/
Example: python run.py llada_gsm8k/full/fusedm_signed_k12_t0.9_b20 --limit 20"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = json.load(open(os.path.join(HERE, 'configs', 'runs.json')))
args = sys.argv[1:]
if not args or args[0] in ('-h', '--help'):
    print(__doc__); sys.exit(0)
if args[0] == '--list':
    for name in sorted(RUNS):
        if len(args) < 2 or args[1] in name:
            print(name)
    sys.exit(0)
name, extra = args[0], args[1:]
if name not in RUNS:
    sys.exit(f'unknown run {name!r}; see: python run.py --list')
c = RUNS[name]
script = 'run_eval_dream.sh' if c['model'] == 'dream' else 'run_eval.sh'
task = c['task']
extra_tasks = ['--include_path', os.path.join(HERE, 'measured_coupling', 'llada', 'tasks')] if task in ('gsm8k_train', 'mbpp_train', 'mbpp_val') else []
if c.get('limit') and '--limit' not in extra:  # the paper's timing jobs used a subset (GSM8K 200, MATH 15 per subject)
    extra = ['--limit', str(int(c['limit']))] + extra
cmd = ['bash', os.path.join(HERE, 'scripts', script), name, task, str(c['num_fewshot'] or 0), c['model_args']] + extra_tasks + extra
print(' '.join(cmd)); sys.exit(subprocess.call(cmd))
