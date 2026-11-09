"""CPU check of dream_fused_forward on a tiny random Dream model.
1. Main logits equal the model's own (shifted) logits.
2. With every position recomputed (ppos = 1..L-1), each probe row equals an exact forward pass on the
   sequence with that row's tokens, so the probe approximation is exact there.
Usage: python measured_coupling/dream/tests/test_dream_fused.py"""
import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from dream_fused import dream_fused_forward

torch.manual_seed(0)
cfg = DreamConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=128, rope_theta=10000.0, pad_token_id=0)
model = DreamModel(cfg).eval().double()
L = 24
x = torch.randint(0, 97, (L,))


def exact(seq):
    out = model(seq.unsqueeze(0), 'full', None)
    out = out[0] if isinstance(out, tuple) else out
    lg = out.logits
    return torch.cat([lg[:, :1], lg[:, :-1]], 1)[0]


ref = exact(x)
ppos = torch.arange(10, 16)
rows = x[ppos].unsqueeze(0).repeat(3, 1)
rows[0, 0], rows[1, 2], rows[2, 5] = 5, 7, 11
main, probe = dream_fused_forward(model, x, rows, ppos)
e1 = (main[0] - ref).abs().max().item()
print('main logits max |diff|:', e1)

pall = torch.arange(1, L)
rows = x[pall].unsqueeze(0).repeat(2, 1)
rows[0, 3], rows[1, 9] = 5, 7
_, probe = dream_fused_forward(model, x, rows, pall)
e2 = 0
for r in range(2):
    seq = x.clone(); seq[pall] = rows[r]
    e2 = max(e2, (probe[r] - exact(seq)[pall]).abs().max().item())
print('probe logits (all positions recomputed) max |diff|:', e2)
assert e1 < 1e-8 and e2 < 1e-6, 'mismatch'  # probe scores use a float32 softmax; the test model is double
print('OK')
