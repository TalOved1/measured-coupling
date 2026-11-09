"""Fused probe pass for Dream (Qwen2-style, grouped-query attention, shifted logits).

Same idea as llada/mf_select.fused_forward for LLaDA: one forward pass computes the main logits and the
K probe rows, so every weight is read once. Each probe row reveals one candidate token and recomputes only
a few positions; every other position uses the main keys/values of the same layer.

Dream predicts the token at position i from the hidden state at i-1. So a probe row recomputes the masked
positions P and the positions just before them, R = P u (P-1). The probe logits for i in P are read at
the row slot of i-1."""
import math
import torch
import torch.nn.functional as F


def _rotate_half(t):
    t1, t2 = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2:]
    return torch.cat((-t2, t1), dim=-1)


def _rope(t, cos, sin):  # t (..., n, hd); cos/sin broadcastable (1, 1, n, hd)
    return t * cos + _rotate_half(t) * sin


def recompute_positions(ppos):
    """R = sorted(P u (P-1)) and, for each i in P, the index of i-1 in R."""
    R = torch.unique(torch.cat([ppos, ppos - 1]))  # sorted
    return R, torch.searchsorted(R, ppos - 1)


@torch.no_grad()
def dream_fused_forward(model, x, probe_rows, ppos):
    """x: (L,) tokens; ppos: (m,) masked positions (sorted); probe_rows: (K, m) tokens at ppos, one candidate
    revealed per row. Returns main logits (1, L, V), already shifted as in Dream's sampler, and probe logits
    (K, m, V) for the positions ppos (also shifted). Assumes batch 1 and no padding (Dream's "full" mask)."""
    M = model.model
    cfg = model.config
    L = x.numel()
    K, m = probe_rows.shape
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = cfg.hidden_size // nh
    g = nh // nkv
    R, prev_idx = recompute_positions(ppos)
    nR = R.numel()
    rows = x[R].unsqueeze(0).repeat(K, 1)
    rows[:, torch.searchsorted(R, ppos)] = probe_rows
    h = M.embed_tokens(torch.cat([x, rows.reshape(-1)]))  # (L + K*nR, d)
    pos_ids = torch.arange(L, device=x.device).unsqueeze(0)
    cos, sin = M.rotary_emb(h, pos_ids)  # (1, L, hd)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (1, 1, L, hd)
    cos_r, sin_r = cos[:, :, R], sin[:, :, R]
    excl = torch.zeros(L, dtype=torch.bool, device=x.device)
    excl[R] = True  # shared keys at recomputed positions come from each row's own keys
    scale = 1.0 / math.sqrt(hd)

    for layer in M.layers:
        at = layer.self_attn
        xn = layer.input_layernorm(h)
        q, k, v = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
        qm = _rope(q[:L].view(1, L, nh, hd).transpose(1, 2), cos, sin)
        km = _rope(k[:L].view(1, L, nkv, hd).transpose(1, 2), cos, sin)
        vm = v[:L].view(1, L, nkv, hd).transpose(1, 2)
        qp = _rope(q[L:].view(K, nR, nh, hd).transpose(1, 2), cos_r, sin_r)  # (K, nh, nR, hd)
        kp = _rope(k[L:].view(K, nR, nkv, hd).transpose(1, 2), cos_r, sin_r)  # (K, nkv, nR, hd)
        vp = v[L:].view(K, nR, nkv, hd).transpose(1, 2)
        att_m = F.scaled_dot_product_attention(qm, km.repeat_interleave(g, 1), vm.repeat_interleave(g, 1))
        # Probe queries grouped by their kv head, so the shared keys/values are read once per kv head.
        qg = qp.reshape(K, nkv, g * nR, hd)  # head index = kv * g + group, as in repeat_kv
        qflat = qg.transpose(0, 1).reshape(nkv, K * g * nR, hd)
        sc_sh = (torch.matmul(qflat, km[0].transpose(-1, -2)).float() * scale).view(nkv, K, g * nR, L).transpose(0, 1)
        sc_sh = sc_sh.masked_fill(excl, float('-inf'))
        sc_own = torch.matmul(qg, kp.transpose(-1, -2)).float() * scale  # (K, nkv, g*nR, nR)
        p = torch.softmax(torch.cat([sc_sh, sc_own], -1), -1).to(vm.dtype)
        p_sh = p[..., :L].transpose(0, 1).reshape(nkv, K * g * nR, L)
        att_p = torch.matmul(p_sh, vm[0]).view(nkv, K, g * nR, hd).transpose(0, 1) + torch.matmul(p[..., L:], vp)
        att_p = att_p.reshape(K, nh, nR, hd)
        att = torch.cat([att_m.transpose(1, 2).reshape(L, nh * hd), att_p.transpose(1, 2).reshape(K * nR, nh * hd)])
        h = h + at.o_proj(att)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    h = M.norm(h)
    logits = model.lm_head(h[:L]).unsqueeze(0)
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)  # Dream's shift
    hp = h[L:].view(K, nR, -1)[:, prev_idx]  # hidden state at i-1 for each i in ppos
    return logits, model.lm_head(hp)
