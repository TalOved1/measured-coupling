# Lagged coupling estimators: learn how committing tokens affects the other masked positions, from
# the decoder's own next forward pass, with zero extra forward passes.
#
# At step t the decoder proposes token v_i with log-margin c_i^t for every active masked i, and
# commits a set S_t. The forward pass of step t+1 (run anyway) re-predicts every still-masked i.
# Only S_t changed in between, so this pass shows exactly what revealing S_t did to i:
#   y_i = c_i^t - c_i^{t+1}            margin drop of the old token v_i       (LaggedCoupling)
#   z_i = 1[argmax_i^{t+1} != v_i]     "flip": committing i together with S_t
#                                      would have disagreed with the choice made given S_t   (FlipCoupling)
# Each committed j appears in a single step, so per-pair effects are not identifiable; both
# estimators fit a model shared across pairs on free features phi(i, j), summed over j in S_t.

import math
import torch

from mf_select import pairwise_jsd, log_margin

# Distance buckets are split by direction (j left / right of i): dependencies are not symmetric.
FEATURES = ['bias', 'L1', 'L2', 'L3_4', 'L5_8', 'R1', 'R2', 'R3_4', 'R5_8', 'jsd_sim', 'same_tok', 'unc_i', 'unc_i*adj']


def pair_features(pos, tok, probs):
    """pos: (m,) positions, tok: (m,) proposed tokens, probs: (m, V). Returns phi: (m, m, K) with
    phi[i, j] = features of the effect of committing j on i (diagonal is zero)."""
    m = pos.numel()
    d = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs().float()  # (m, m)
    left = pos.unsqueeze(0) < pos.unsqueeze(1)  # j left of i
    sim = 1.0 - pairwise_jsd(probs) / math.log(2.0) if m > 1 else torch.zeros(m, m, device=pos.device)
    same = (tok.unsqueeze(0) == tok.unsqueeze(1)).float()
    unc = (1.0 - probs.max(-1).values).unsqueeze(1).expand(m, m)  # uncertainty of the affected i
    buckets = [d == 1, d == 2, (d >= 3) & (d <= 4), (d >= 5) & (d <= 8)]
    dist = [(bk & left).float() for bk in buckets] + [(bk & ~left).float() for bk in buckets]
    phi = torch.stack([torch.ones_like(d)] + dist + [sim.float(), same, unc, unc * (d == 1).float()], dim=-1)
    phi[torch.arange(m), torch.arange(m)] = 0.0
    return phi


class LaggedCoupling:
    """Online estimator of Delta_ij = theta^T phi(i, j). One instance per decoding run (batch size 1).
    signed: keep negative couplings (j supports i); otherwise clamp D at 0 (conflicts only).
    reset_each_question: forget theta between questions (pure per-question zero-shot)."""

    def __init__(self, ridge=1.0, forget=1.0, signed=True, reset_each_question=False, min_obs=20, log=False):
        self.K = len(FEATURES)
        self.log = ObservationLog() if log else None
        self.ridge, self.forget, self.signed = ridge, forget, signed
        self.reset_each_question, self.min_obs = reset_each_question, min_obs
        self.reset_theta()
        self.prev = None

    def reset_theta(self):
        self.A = self.ridge * torch.eye(self.K, dtype=torch.float64)
        self.b = torch.zeros(self.K, dtype=torch.float64)
        self.theta = torch.zeros(self.K, dtype=torch.float64)
        self.n_obs = 0
        self.sq_err_before = 0.0  # sum of squared prediction errors before each update (honest online error)
        self.sq_y = 0.0

    def start_question(self):
        self.prev = None
        if self.log is not None:
            self.log.question += 1
        if self.reset_each_question:
            self.reset_theta()

    @torch.no_grad()
    def observe(self, logits_row, pos_now):
        """Use this step's forward pass to update theta from the previous step's commit.
        logits_row: (L, V) logits of the current forward; pos_now: (m',) active masked positions."""
        p = self.prev
        self.prev = None
        if p is None or p['S'].numel() == 0:
            return
        keep = ~p['commit']  # previous active positions that are still masked
        if keep.sum() == 0:
            return
        idx = keep.nonzero(as_tuple=True)[0]
        still = torch.isin(p['pos'][idx], pos_now)
        idx = idx[still]
        if idx.numel() == 0:
            return
        logp = torch.log_softmax(logits_row[p['pos'][idx]].float(), -1)
        c_new = log_margin(logp, p['tok'][idx])
        y = (p['c'][idx] - c_new).double().cpu()  # (n,)
        X = p['phi'][idx][:, p['S']].sum(1).double().cpu()  # (n, K): sum_{j in S} phi(i, j)
        if self.log is not None:
            self.log.add(p, idx, logp)
        pred = X @ self.theta
        self.sq_err_before += float(((y - pred) ** 2).sum())
        self.sq_y += float((y ** 2).sum())
        self.A = self.forget * self.A + X.T @ X
        self.b = self.forget * self.b + X.T @ y
        self.n_obs += y.numel()
        self.theta = torch.linalg.solve(self.A, self.b)

    def coupling(self, phi):
        """phi: (m, m, K) -> D (m, m) with D[i, j] = predicted penalty on i from committing j."""
        if self.n_obs < self.min_obs:
            return torch.zeros(phi.shape[:2], device=phi.device)
        D = phi @ self.theta.to(phi.device, phi.dtype)
        return D if self.signed else D.clamp_min(0.0)

    def record(self, pos, tok, c, commit, phi):
        """Remember this step so the next forward pass can be used as an observation."""
        self.prev = {'pos': pos, 'tok': tok, 'c': c, 'commit': commit, 'phi': phi,
                     'S': commit.nonzero(as_tuple=True)[0]}

    def summary(self):
        r2 = 1.0 - self.sq_err_before / self.sq_y if self.sq_y > 0 else float('nan')
        return {'n_obs': self.n_obs, 'online_R2': r2,
                'theta': dict(zip(FEATURES, [round(float(t), 4) for t in self.theta]))}


UNARY_FEATURES = ['bias', 'margin_i', 'unc_i']


def unary_features(c, probs):
    """c: (m,) log-margins, probs: (m, V). Returns u: (m, Ku)."""
    return torch.stack([torch.ones_like(c), c.float(), (1.0 - probs.max(-1).values).float()], dim=-1)


class ObservationLog:
    """Raw lagged observations, one row per (still-masked i, commit step): saved for offline analysis."""

    def __init__(self):
        self.rows = {k: [] for k in ['question', 'step', 'pos', 'n_S', 'c_old', 'c_new', 'flip', 'p_top_old', 'X', 'u']}
        self.question, self.step = -1, 0

    def add(self, p, idx, logp):
        n = idx.numel()
        new_top = logp.argmax(-1)
        c_new = log_margin(logp, p['tok'][idx])
        self.rows['question'].append(torch.full((n,), self.question))
        self.rows['step'].append(torch.full((n,), self.step))
        self.rows['pos'].append(p['pos'][idx].cpu())
        self.rows['n_S'].append(torch.full((n,), p['S'].numel()))
        self.rows['c_old'].append(p['c'][idx].float().cpu())
        self.rows['c_new'].append(c_new.float().cpu())
        self.rows['flip'].append((new_top != p['tok'][idx]).cpu())
        self.rows['p_top_old'].append(p['p_top'][idx].float().cpu() if 'p_top' in p else torch.full((n,), float('nan')))
        self.rows['X'].append(p['phi'][idx][:, p['S']].sum(1).float().cpu())
        u = p.get('u')
        self.rows['u'].append(u[idx].float().cpu() if u is not None else torch.full((n, len(UNARY_FEATURES)), float('nan')))
        self.step += 1

    def save(self, path):
        out = {k: torch.cat(v) if v else torch.empty(0) for k, v in self.rows.items()}
        out['pair_features'], out['unary_features'] = FEATURES, UNARY_FEATURES
        torch.save(out, path)
        return int(out['pos'].numel())


class FlipCoupling:
    """Online logistic model of the flip event (see header):
        logit P(flip_i | S) = theta_u^T u_i + sum_{j in S} theta_p^T phi(i, j)   =: a_i + sum_{j in S} w_ij
    fitted by ridge-regularized logistic regression (Newton / IRLS, warm-started) on all lagged
    observations so far. Shared across the questions of a run unless reset_each_question.
    signed=False clamps w_ij >= 0 (committing j can only raise i's flip odds; conservative).
    unary_only=True (ablation): pair features are zeroed, so theta_p stays 0 and W = 0; what remains is
    a calibrated per-token confidence rule."""

    def __init__(self, ridge=1.0, signed=False, reset_each_question=False, min_obs=200, min_flips=20,
                 refit_every=100, newton_steps=3, log=False, unary_only=False):
        self.unary_only = unary_only
        self.Ku, self.Kp = len(UNARY_FEATURES), len(FEATURES)
        self.ridge, self.signed, self.reset_each_question = ridge, signed, reset_each_question
        self.min_obs, self.min_flips, self.refit_every, self.newton_steps = min_obs, min_flips, refit_every, newton_steps
        self.log = ObservationLog() if log else None
        self.reset_theta()
        self.prev = None

    def reset_theta(self):
        self.theta = torch.zeros(self.Ku + self.Kp, dtype=torch.float64)
        self.Xs, self.zs = [], []
        self.n_obs = self.n_flips = self.since_fit = 0
        self.logloss_before = self.brier_before = 0.0  # scored before each update (honest online metrics)
        self.n_scored = 0

    @property
    def ready(self):
        return self.n_obs >= self.min_obs and self.n_flips >= self.min_flips

    def start_question(self):
        self.prev = None
        if self.log is not None:
            self.log.question += 1
        if self.reset_each_question:
            self.reset_theta()

    def _fit(self):
        X = torch.cat(self.Xs)
        z = torch.cat(self.zs)
        reg = self.ridge * torch.eye(X.shape[1], dtype=torch.float64)
        reg[0, 0] = 1e-6  # do not shrink the intercept
        th = self.theta
        for _ in range(self.newton_steps):
            p = torch.sigmoid(X @ th)
            g = X.T @ (p - z) + reg @ th
            H = (X * (p * (1 - p)).unsqueeze(1)).T @ X + reg
            th = th - torch.linalg.solve(H, g)
        self.theta = th
        self.since_fit = 0

    @torch.no_grad()
    def observe(self, logits_row, pos_now):
        p = self.prev
        self.prev = None
        if p is None or p['S'].numel() == 0:
            return
        idx = (~p['commit']).nonzero(as_tuple=True)[0]
        idx = idx[torch.isin(p['pos'][idx], pos_now)]
        if idx.numel() == 0:
            return
        logp = torch.log_softmax(logits_row[p['pos'][idx]].float(), -1)
        if self.log is not None:
            self.log.add(p, idx, logp)
        z = (logp.argmax(-1) != p['tok'][idx]).double().cpu()
        pair = p['phi'][idx][:, p['S']].sum(1)
        if self.unary_only:
            pair = torch.zeros_like(pair)  # zero columns + ridge keep theta_p at exactly 0
        X = torch.cat([p['u'][idx], pair], dim=-1).double().cpu()
        if self.ready:
            pr = torch.sigmoid(X @ self.theta).clamp(1e-6, 1 - 1e-6)
            self.logloss_before -= float((z * pr.log() + (1 - z) * (1 - pr).log()).sum())
            self.brier_before += float(((pr - z) ** 2).sum())
            self.n_scored += z.numel()
        self.Xs.append(X)
        self.zs.append(z)
        self.n_obs += z.numel()
        self.n_flips += int(z.sum())
        self.since_fit += z.numel()
        if self.n_obs >= self.min_obs and self.since_fit >= self.refit_every:
            self._fit()

    def logit_terms(self, u, phi):
        """Returns a (m,) unary flip log-odds and W (m, m) with W[i, j] = w_ij (diag 0)."""
        th = self.theta.to(phi.device, torch.float32)
        a = u.float() @ th[:self.Ku]
        W = phi.float() @ th[self.Ku:]
        if not self.signed:
            W = W.clamp_min(0.0)
        W.fill_diagonal_(0.0)
        return a, W

    def record(self, pos, tok, c, commit, phi, u):
        self.prev = {'pos': pos, 'tok': tok, 'c': c, 'commit': commit, 'phi': phi, 'u': u,
                     'S': commit.nonzero(as_tuple=True)[0]}

    def summary(self):
        b = self.n_flips / max(self.n_obs, 1)
        base_ll = -(b * math.log(max(b, 1e-12)) + (1 - b) * math.log(max(1 - b, 1e-12)))
        ns = max(self.n_scored, 1)
        return {'n_obs': self.n_obs, 'flip_rate': b, 'n_scored': self.n_scored,
                'online_logloss': self.logloss_before / ns, 'base_rate_logloss': base_ll,
                'online_brier': self.brier_before / ns, 'base_rate_brier': b * (1 - b),
                'theta_unary': dict(zip(UNARY_FEATURES, [round(float(t), 4) for t in self.theta[:self.Ku]])),
                'theta_pair': dict(zip(FEATURES, [round(float(t), 4) for t in self.theta[self.Ku:]]))}


def flip_mean_field_commit(a, W, rho=0.05, R=2, damping=1.0, eps=None, schedule='sequential'):
    """Decision rule for the flip model.
    Utility of committing S (log-odds form): U(S) = sum_{i in S} (gamma - a_i) - sum_{i != j in S} w_ij,
    gamma = logit(rho): commit i only when its flip probability given the rest of S is <= rho.
    Naive mean-field ELBO:  L(q) = sum_i q_i (gamma - a_i) - q^T W q + sum_i H(q_i).
    Since s^T W s = s^T W_sym s, its stationary points satisfy
        q_i = sigmoid(gamma - a_i - sum_j (w_ij + w_ji) q_j),   commit {i : q_i >= 1/2}.
    schedule='sequential': coordinate ascent (each update maximizes L exactly in q_i, so L rises
        monotonically and converges); visits positions in order of decreasing (gamma - a_i).
    schedule='parallel': synchronous updates as in the MF paper; can oscillate when couplings are
        large (no contraction once ||W + W^T|| > 4), e.g. two conflicting tokens switching together.
    eps (optional certificate): greedily drop the riskiest member until
        sum_{i in S} P(flip_i | S \\ {i}) <= eps  (expected wrong commits <= eps, so P(any) <= eps).
    Returns (commit (m,) bool, q (m,), flip prob of each member given the final S (m,))."""
    gamma = math.log(rho / (1 - rho))
    Ws = W + W.T
    h = gamma - a
    q = torch.sigmoid(h)
    if schedule == 'sequential':
        order = torch.argsort(h, descending=True).tolist()
        for _ in range(R):
            for i in order:
                q[i] = torch.sigmoid(h[i] - Ws[i] @ q)  # Ws[i, i] = 0
    else:
        for _ in range(R):
            q = (1 - damping) * q + damping * torch.sigmoid(h - Ws @ q)
    commit = q >= 0.5
    if not commit.any():
        commit[q.argmax()] = True

    def member_flip(cm):
        s = cm.float()
        return torch.sigmoid(a + W @ s) * s  # W diag is 0, so row i sums over S \ {i}

    pf = member_flip(commit)
    if eps is not None:
        while commit.sum() > 1 and pf.sum() > eps:
            commit[pf.argmax()] = False
            pf = member_flip(commit)
    return commit, q, pf
