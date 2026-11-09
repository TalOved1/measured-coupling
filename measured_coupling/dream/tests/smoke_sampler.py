"""CPU smoke test of the Dream sampler with alg='mf' (MF with JSD, and our fused probes) on a tiny model.
Usage: python measured_coupling/dream/tests/smoke_sampler.py"""
import sys, os, types
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from model.generation_utils import DreamGenerationMixin

torch.manual_seed(0)
cfg = DreamConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=256, rope_theta=10000.0, pad_token_id=0,
                  mask_token_id=96, bos_token_id=1, eos_token_id=2)
model = DreamModel(cfg).eval()
model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
model._sample = types.MethodType(DreamGenerationMixin._sample, model)
ids = torch.randint(3, 90, (1, 30))
for coupling, tau in (('jsd', 0.3), ('fusedm_signed', 0.3), ('jsd', 0.999), ('fusedm_signed', 0.999)):
    out, nfe = model.diffusion_generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=50, steps=50,
                                        alg='mf', block_length=20, mf_coupling=coupling, mf_tau=tau, mf_probe_k=4,
                                        return_dict_in_generate=True, output_history=False, temperature=0.0)
    gen = out.sequences[0, 30:]
    assert (gen != 96).all(), 'masks left'
    print(coupling, 'nfe', nfe, 'ok')
