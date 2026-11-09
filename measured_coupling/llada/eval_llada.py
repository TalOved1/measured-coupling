# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import os
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache, generate_klass, generate_mf_dual_cache, generate_fastdllm_dual_cache_graph, generate_mf_dual_cache_lean
import generate as _gen_module
from model.modeling_llada import LLaDAModelLM
import json
import time
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        use_cache=False,
        threshold=None,
        factor=None,
        save_dir=None,
        show_speed=False,
        dual_cache=False,
        dawn=False,
        klass=False,
        local_leap=False,
        outp_path=None,
        threshold_klass=0.6,
        kl_threshold=0.015,
        tau_sink=0.01,
        tau_edge=0.07,
        tau_induce=0.7,
        tau_low=0.7,
        relaxed_threshold=0.75,
        radius=4,
        doc_range=None,
        mf=False,
        mf_tau=0.85,
        mf_R=2,
        mf_pairwise=True,
        mf_uniform_D=False,
        mf_damping=1.0,
        mf_coupling='jsd',
        mf_lag_signed=True,
        mf_lag_ridge=1.0,
        mf_lag_forget=1.0,
        mf_lag_reset=False,
        mf_lag_log=False,
        mf_flip_rho=0.05,
        mf_flip_eps=None,
        mf_probe_k=8,
        mf_probe_graph=False,
        mf_plugin=None,
        mf_control=None,
        mf_dual_cache=False,
        cache_graph=False,
        cache_kernel='auto',
        mf_probe_select='top',
        mf_probe_window=0,
        mf_probe_attn='manual',
        mf_probe_every=1,
        plugin_theta=0.9,
        plugin_tau_low=0.7,
        plugin_tau_induce=0.7,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        # DAWN's local config class, so the hub repo's remote code is never executed.
        from model.configuration_llada import LLaDAConfig
        config = LLaDAConfig.from_pretrained(model_path)
        config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.is_instruct = True if 'instruct' in model_path.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        self.dawn = dawn
        self.outp_path = outp_path
        self.klass = klass
        self.local_leap = local_leap
        self.threshold_klass = threshold_klass
        self.kl_threshold = kl_threshold
        self.tau_sink = tau_sink
        self.tau_edge = tau_edge
        self.tau_induce = tau_induce
        self.tau_low = tau_low
        self.relaxed_threshold = relaxed_threshold
        self.radius = radius
        # doc_range='a-b': generate only for requests a <= idx < b (others return ''), so a long run can be
        # split into parallel jobs that see identical prompts (lm-eval's few-shot sampling is order dependent).
        self.doc_range = tuple(int(v) for v in str(doc_range).split('-')) if doc_range not in (None, 'None') else None
        self.mf_dual_cache = str(mf_dual_cache) in ('True', 'true', '1')
        self.cache_graph = str(cache_graph) in ('True', 'true', '1')  # cached decoders on the shared CUDA-graph step
        self.cache_kernel = str(cache_kernel)  # 'auto': three-source kernel with probes, first kernel without; '4': rotary-fused kernel for all
        self.mf_probe_select = str(mf_probe_select)
        self._graphs = {}
        self.mf_kwargs = dict(mf=mf, mf_tau=float(mf_tau), mf_R=int(mf_R), mf_pairwise=mf_pairwise,
                              mf_uniform_D=mf_uniform_D, mf_damping=float(mf_damping),
                              mf_coupling=mf_coupling, mf_probe_k=int(mf_probe_k),
                              mf_probe_graph=mf_probe_graph, mf_probe_window=int(mf_probe_window), mf_probe_attn=str(mf_probe_attn), mf_probe_every=int(mf_probe_every),
                              mf_plugin=None if mf_plugin in (None, 'None') else mf_plugin,
                              mf_control=None if mf_control in (None, 'None') else str(mf_control),
                              # plug-in rules (plugins.py): threshold = Fast-dLLM's rule; dawn = DAWN's schedule
                              mf_plugin_kw={'threshold': {'theta': float(plugin_theta)},
                                            'dawn': {'tau_low': float(plugin_tau_low), 'tau_induce': float(plugin_tau_induce)}}.get(mf_plugin))
        if mf_coupling == 'lagged':
            from lagged import LaggedCoupling
            self.mf_kwargs['mf_estimator'] = LaggedCoupling(ridge=float(mf_lag_ridge), forget=float(mf_lag_forget),
                                                            signed=mf_lag_signed, reset_each_question=mf_lag_reset,
                                                            log=mf_lag_log)
        elif mf_coupling == 'flip':
            from lagged import FlipCoupling
            # mf_flip_signed=True allows w_ij < 0 (support); default is conservative (w_ij >= 0).
            self.mf_kwargs['mf_estimator'] = FlipCoupling(ridge=float(mf_lag_ridge), signed=kwargs.pop('mf_flip_signed', False),
                                                          unary_only=kwargs.pop('mf_flip_unary_only', False),
                                                          reset_each_question=mf_lag_reset, log=mf_lag_log)
            self.mf_kwargs['mf_flip_rho'] = float(mf_flip_rho)
            self.mf_kwargs['mf_flip_eps'] = None if mf_flip_eps in (None, 'None') else float(mf_flip_eps)
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def _graph(self, K=None):
        """The shared CUDA-graph cached step (mf_select.CachedStepGraph), one per probe count; None if disabled."""
        if not self.cache_graph:
            return None
        import mf_select
        fused = self.mf_kwargs.get('mf_coupling', 'jsd').startswith('fusedm_')
        K = (self.mf_kwargs['mf_probe_k'] if fused else 0) if K is None else K
        if K not in self._graphs:
            # best kernel per case: the three-source kernel with probe rows, the first fused kernel without (K = 0)
            kern = 4 if self.cache_kernel == '4' else (3 if K else True)
            self._graphs[K] = mf_select.CachedStepGraph(self.model, K, kernel=kern)
        return self._graphs[K]

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        _gen_module.BLOCK_START_TIME[0] = 0.0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        
        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])
        
        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            if self.doc_range is not None:
                assert len(batch) == 1, 'doc_range needs batch_size 1'
                if not (self.doc_range[0] <= len(output) < self.doc_range[1]):
                    output.append('')
                    continue
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                if self.is_instruct:
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = self.tokenizer(user_input)['input_ids']
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)['input_ids']
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))
            
            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device), torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)
            
            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros((batched_input_ids.shape[0], 1, max_len+self.gen_length, max_len+self.gen_length), device=self.device, dtype=torch.bool)
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True


            stop_tokens = req.args[1]['until']
            input_ids = batched_input_ids
            if self.mf_kwargs.get('mf') and self.mf_dual_cache and self.cache_graph and self.mf_kwargs['mf_coupling'].startswith('fusedm_'):
                # measured coupling, cached: lean per-step code on the shared graph step with the three-source kernel
                generated_answer, nfe = generate_mf_dual_cache_lean(self.model, input_ids, gen_length=self.gen_length, block_length=self.block_length,
                                        mask_id=self.mask_id, mf_tau=self.mf_kwargs['mf_tau'], mf_R=self.mf_kwargs['mf_R'],
                                        mf_probe_k=self.mf_kwargs['mf_probe_k'], graph=self._graph())
            elif self.mf_kwargs.get('mf') and self.mf_dual_cache:
                generated_answer, nfe = generate_mf_dual_cache(self.model, input_ids, gen_length=self.gen_length, block_length=self.block_length,
                                        mask_id=self.mask_id, mf_tau=self.mf_kwargs['mf_tau'], mf_R=self.mf_kwargs['mf_R'],
                                        mf_coupling=self.mf_kwargs['mf_coupling'], mf_probe_k=self.mf_kwargs['mf_probe_k'],
                                        mf_pairwise=self.mf_kwargs['mf_pairwise'], graph=self._graph(), probe_select=self.mf_probe_select)
            elif self.use_cache and self.dual_cache and self.cache_graph:
                generated_answer, nfe = generate_fastdllm_dual_cache_graph(self.model, input_ids, gen_length=self.gen_length,
                                        block_length=self.block_length, mask_id=self.mask_id, threshold=self.threshold, graph=self._graph(0))
            elif self.use_cache:
                if self.dual_cache:
                    generated_answer, nfe = generate_with_dual_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
                else:
                    generated_answer, nfe = generate_with_prefix_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor)
            elif self.klass:
                generated_answer, nfe = generate_klass(self.model, input_ids, gen_length=self.gen_length, steps=self.steps, block_length=self.block_length, 
                                        temperature=0, mask_id=self.mask_id, conf_threshold=self.threshold_klass, kl_threshold=self.kl_threshold,)
            else:
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold, factor=self.factor, dawn=self.dawn, local_leap=self.local_leap, tau_sink=self.tau_sink, tau_edge=self.tau_edge, tau_induce=self.tau_induce, tau_low=self.tau_low, relaxed_threshold=self.relaxed_threshold, radius=self.radius, **self.mf_kwargs)

            # torch.cuda.empty_cache()
            
            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                batched_generated_answer = [self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) for i in range(len(generated_answer_ids))]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    generated_answer_i = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # output.append(generated_answer)
            # Per-question NFE, in request order (same order as lm-eval's doc_id for generate_until tasks).
            if self.outp_path:
                with open(os.path.join(os.path.dirname(self.outp_path), 'nfe.jsonl'), 'a', encoding='utf-8') as f:
                    for k, req in enumerate(batch):
                        import mf_select as _ms  # probe statistics of this question (stale probes, measured pairs)
                        _st = dict(_ms.PROBE_STATS); _ms.PROBE_STATS.update(probes=0, stale=0, pairs=0, measured_pairs=0)
                        f.write(json.dumps({'idx': len(output) + k, 'nfe': nfe, 'probe_stats': _st,
                                            'prompt_tail': req.args[0][-120:]}, ensure_ascii=False) + '\n')
            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer in batched_generated_answer:
                        f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                # print('question: ', question)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')
            # self.accelerator.wait_for_everyone()
        end_time = time.time()
        est = self.mf_kwargs.get('mf_estimator')
        if est is not None and self.outp_path:
            with open(os.path.join(os.path.dirname(self.outp_path), 'lagged_theta.json'), 'w') as f:
                json.dump(est.summary(), f, indent=1)
            if est.log is not None:
                n = est.log.save(os.path.join(os.path.dirname(self.outp_path), 'observations.pt'))
                print(f'saved {n} lagged observations')
            print('lagged estimator:', est.summary())
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")
            print(f"Tokens per NFE is {num_tokens / num_nfe}")
            print(f"Average NFE is {num_nfe / len(output)}")


            dirpath = os.path.dirname(self.outp_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            with open(self.outp_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {
                        'Total Number of Tokens': num_tokens.item(),
                        'Total Time Taken': end_time - start_time,
                        'Tokens per Second': num_tokens.item() / (end_time - start_time),
                        'Total NFE': num_nfe,
                        'Block Start Time': _gen_module.BLOCK_START_TIME[0],  # cached decoders: full passes at block starts
                        'Tokens per NFE': num_tokens.item() / num_nfe,
                        'Average NFE': num_nfe / len(output),
                    },
                    ensure_ascii=False
                ) + "\n")
            
        return output


if __name__ == "__main__":
    cli_evaluate()
    
