"""Measured coupling as a plug-in for other decoders.

Each rule takes the base method's confidences p_i (top-token probability) over the masked positions of the
block and the measured coupling D[i, j] = c_i - c_i^(j) (change of i's log-margin when j is revealed; 0 for
unprobed j). Committing j shifts i's confidence by the additive model of Proposition 1:
    p_i' = sigmoid(logit(p_i) - sum_{j in S} D[i, j]).
With D = 0 every rule reduces exactly to its base method, so differences come from the coupling only."""
import torch


def _logit(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def corrected(p, D, S):
    """p_i' given that the positions in the bool mask S are committed (i itself excluded)."""
    Dz = D.clone()
    Dz.fill_diagonal_(0.0)
    return torch.sigmoid(_logit(p) - Dz @ S.to(D.dtype))


def threshold_select(p, D, theta=0.9, R=2):
    """Confidence threshold (Fast-dLLM) with measured correction. Start from {p_i >= theta}; then R times,
    keep the positions whose corrected confidence given the others still passes theta. Conflicts drop tokens,
    support adds tokens just under the bar."""
    S = p >= theta
    for _ in range(R):
        S = corrected(p, D, S) >= theta
    if not S.any():
        S[p.argmax()] = True
    return S


def dawn_select(p, D, tau_low=0.7, tau_induce=0.7, conf=0.9):
    """DAWN's schedule with measured relations in place of attention:
    1. confident set A = {p_i >= conf} (as DAWN);
    2. induced set I: positions supported by A (net support from A is positive) whose corrected confidence
       given A reaches tau_induce (DAWN: attention edge to a confident anchor and p_i >= tau_induce);
    3. the rest with p_i >= tau_low, minus those in conflict with A or I, go through DAWN's greedy maximum
       independent set by confidence, on the measured conflict graph. A conflict edge (i, j) means revealing
       one would push the other below tau_low."""
    m = p.numel()
    Dz = D.clone()
    Dz.fill_diagonal_(0.0)
    A = p >= conf
    support = -(Dz @ A.to(D.dtype))  # > 0: net support from the confident set
    I = ~A & (support > 0) & (corrected(p, D, A) >= tau_induce)
    lp = _logit(p)
    # conflict[i, j]: revealing j pushes i below tau_low (only meaningful where p_i >= tau_low)
    hurt = torch.sigmoid(lp.unsqueeze(1) - Dz) < tau_low
    conflict = (hurt | hurt.T) & ~torch.eye(m, dtype=torch.bool, device=p.device)
    sel = A | I
    cand = (p >= tau_low) & ~sel & ~(conflict & sel.unsqueeze(0)).any(1)
    conf_c = torch.where(cand, p, torch.full_like(p, -1.0))
    while (conf_c > -1).any():
        k = int(conf_c.argmax())
        sel[k] = True
        conf_c[k] = -1.0
        conf_c[conflict[k]] = -1.0
    if not sel.any():
        sel[p.argmax()] = True
    return sel
