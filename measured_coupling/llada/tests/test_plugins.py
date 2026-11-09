"""CPU checks of the measured-coupling plug-ins (measured_coupling/llada/plugins.py).
Usage: python measured_coupling/llada/tests/test_plugins.py"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plugins import threshold_select, dawn_select

torch.manual_seed(0)
for _ in range(200):  # D = 0 reduces to the base rules
    m = torch.randint(2, 20, (1,)).item()
    p = torch.rand(m)
    Z = torch.zeros(m, m)
    base = p >= 0.9
    if not base.any(): base[p.argmax()] = True
    assert torch.equal(threshold_select(p, Z), base)
    # DAWN without edges: confident set, plus every candidate above tau_low (no conflicts to exclude)
    d = (p >= 0.9) | (p >= 0.7)
    if not d.any(): d[p.argmax()] = True
    assert torch.equal(dawn_select(p, Z), d)
# conflict drops a token, support adds one
p = torch.tensor([0.95, 0.93, 0.85])
D = torch.zeros(3, 3); D[1, 0] = 3.0             # committing 0 hurts 1 a lot
S = threshold_select(p, D); assert S.tolist() == [True, False, False], S
D = torch.zeros(3, 3); D[2, 0] = -2.0; D[2, 1] = -2.0  # 0 and 1 support 2
S = threshold_select(p, D); assert S.tolist() == [True, True, True], S
p = torch.tensor([0.95, 0.8, 0.75])
D = torch.zeros(3, 3); D[2, 1] = 3.0; D[1, 2] = 3.0    # 1 and 2 conflict: DAWN keeps one of them
S = dawn_select(p, D); assert S.tolist() == [True, True, False], S
print('OK')
