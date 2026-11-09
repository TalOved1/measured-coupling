# Measured Coupling for Parallel Decoding in Diffusion Language Models

Official implementation of the paper **Measured Coupling for Parallel Decoding in Diffusion Language Models**
by Tal Oved and Lior Wolf (Tel Aviv University). Paper: [arXiv:2611.08307](https://arxiv.org/abs/2611.08307).

Measured coupling replaces the similarity penalty of mean-field parallel decoding with the
measured effect of committing one token on another's confidence, using cheap probes that run inside the model's
main forward pass. It needs no training.

## Install

```bash
pip install -r requirements.txt   # install torch for your CUDA version first
```

Models (LLaDA-8B-Instruct, LLaDA-1.5, Dream-v0-Instruct-7B) and benchmarks download from the Hugging Face Hub
on first use. The paper used one A100 80GB; the model weights take about 15 GB and the probes add 0.1 to 0.3 GB.

## Run

Every experiment in the paper has a name in `configs/runs.json`. Run one by name:

```bash
python run.py --list llada_gsm8k/full                            # see the available runs
python run.py llada_gsm8k/full/fusedm_signed_k12_t0.9_b20        # our method, LLaDA-8B, GSM8K
python run.py llada_gsm8k/full/fusedm_signed_k12_t0.9_b20 --limit 20   # quick check on 20 questions
```

Results go to `results/<run name>/`: accuracy and per-question samples (lm-eval), `speed.jsonl` (time, tokens per
second, average NFE) and `nfe.jsonl` (steps per question).

Run names follow `<model>_<benchmark>/<group>/<setting>`:

| Setting | Method |
|---|---|
| `fusedm_signed_k12_t0.9_b20` | **Ours** on LLaDA (K = 12 probes, τ = 0.9, block 20) |
| `fusedm_signed_k20_t0.95_b20` | **Ours** on Dream (K = 20, τ = 0.95) |
| `mf_t0.85_b20` / `mf_t0.95_b20` | Mean-field decoding on LLaDA / Dream |
| `fastdllm_t0.9_b32`, `dawn_b32`, `baseline_b32` | Fast-dLLM, DAWN, one token per step |
| `ctrl_*` | Controls of Table 4 |
| `full_cached/*_k4` | Key-value cache results (Appendix D) |

To run a setting that is not in the list, call the scripts directly with your own decoder arguments:

```bash
bash scripts/run_eval.sh my_run gsm8k 5 \
  "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=20,mf=True,mf_tau=0.9,mf_R=2,mf_coupling=fusedm_signed,mf_probe_k=12"
```

`scripts/run_eval_dream.sh` does the same for Dream.

## Reproduce the tables

- **Accuracy and NFE:** run the names in `configs/runs.json`. Code benchmarks are scored with
  `python scripts/rescore_code.py results/<model>_<benchmark>/<group>` (HumanEval uses the DAWN sanitizer).
- **Paired comparison:** `python scripts/paired.py results/<run A> results/<run B>` prints the exact McNemar p-value
  and the paired 95% interval of the accuracy difference, as in the last column of Tables 1 to 3.
- **Time per answer:** run all methods of a benchmark in one job on the same GPU, one after another (the paper
  times ours first and last). Time per answer is `Total Time Taken` in `speed.jsonl` divided by the number of
  questions.
- **Long runs:** add `doc_range=a-b` to the decoder arguments to split a benchmark across jobs, then join them with
  `python scripts/merge_chunks.py results/<run>`.

## Outputs of the paper

`outputs/` holds the per-question results of every run behind a number in the paper, one gzipped JSON line per
question: `doc_id`, `subtask` (MATH subject), `correct`, `nfe` (steps for that question), and the generated
`output`. Per-question steps are missing for a few early runs, which predate that logging; their average steps are
in the paper.

## Code

- `measured_coupling/llada/`: evaluation harness for LLaDA (`eval_llada.py`), decoding loops (`generate.py`), our
  method (`mf_select.py`: fused probes, the measured coupling, the mean-field rule, the controls) and the fused
  probe attention kernel (`probe_attn_kernel.py`, Triton).
- `measured_coupling/dream/`: the same for Dream (`eval.py`, `dream_fused.py`).
- Tests that run on CPU: `python measured_coupling/llada/tests/test_controls.py`,
  `python measured_coupling/dream/tests/smoke_sampler.py`.

The harness builds on the [DAWN](https://github.com/lizhuo-luo/DAWN) code base. MIT License.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{oved2026measured,
  title         = {Measured Coupling for Parallel Decoding in Diffusion Language Models},
  author        = {Oved, Tal and Wolf, Lior},
  journal       = {arXiv preprint arXiv:2611.08307},
  year          = {2026},
  eprint        = {2611.08307},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.08307}
}
```

GitHub's "Cite this repository" button (from `CITATION.cff`) gives the same reference in other formats.
