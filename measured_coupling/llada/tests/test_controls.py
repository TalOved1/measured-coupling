"""CPU checks of apply_control (measured_coupling/llada/mf_select.py). Usage: python measured_coupling/llada/tests/test_controls.py"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mf_select import apply_control

torch.manual_seed(0)
m = 7
D = torch.randn(m, m); D.fill_diagonal_(0.0)
measured = torch.tensor([1, 0, 1, 1, 0, 0, 1], dtype=torch.bool)
D[:, ~measured] = 0.0  # unmeasured columns are zero, as in the decoder
mask = measured.unsqueeze(0).expand(m, m) & ~torch.eye(m, dtype=torch.bool)
for ctrl in ('shuffle', 'support', 'conflict', 'sym', 'bonus:0.5'):
    E = apply_control(D, measured, ctrl)
    assert torch.equal(E[~mask], D[~mask]), ctrl  # unmeasured entries and the diagonal untouched
E = apply_control(D, measured, 'shuffle')
assert torch.equal(E[mask].sort().values, D[mask].sort().values) and not torch.equal(E, D)
assert (apply_control(D, measured, 'support')[mask] <= 0).all()
assert (apply_control(D, measured, 'conflict')[mask] >= 0).all()
assert (apply_control(D, measured, 'bonus:0.5')[mask] == -0.5).all()
S = apply_control(D, measured, 'sym'); both = mask & mask.T
assert torch.allclose(S[both], ((D + D.T) / 2)[both])
print('OK')
