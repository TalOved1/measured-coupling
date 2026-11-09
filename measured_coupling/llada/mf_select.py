# Reimplementation of "Mean-Field Parallel Decoding for Discrete Diffusion Language Models"
# (Zoabi, Ali, Ringel, Wolf; arXiv 2606.15805). The official code is not public.
#
#   c_i   = log pi_i(top1) - log pi_i(top2)                  (unary commit score)
#   D~_ij = 1 - JSD(pi_i, pi_j) / ln 2,  D = D~ / max_{a!=b} D~_ab,  D_ii = 0
#   q^0   = sigmoid(c),   q^{r+1} = sigmoid(c - D q^r)        (R mean-field updates)
#   commit {i : q_i >= tau}  (at least the argmax-q position, as in DAWN's threshold rule)

import math
import torch
import torch.nn.functional as F


def pairwise_jsd(p: torch.Tensor, chunk: int = 8, eps: float = 1e-12) -> torch.Tensor:
    """p: (m, V) probabilities. Returns (m, m) JSD in nats, using JSD = H(M) - (H(p)+H(q))/2."""
    m = p.shape[0]
    h = -(p * torch.log(p.clamp_min(eps))).sum(-1)  # (m,)
    jsd = torch.empty(m, m, device=p.device, dtype=p.dtype)
    for s in range(0, m, chunk):
        mix = 0.5 * (p[s:s + chunk, None, :] + p[None, :, :])  # (c, m, V)
        h_mix = -(mix * torch.log(mix.clamp_min(eps))).sum(-1)  # (c, m)
        jsd[s:s + chunk] = h_mix - 0.5 * (h[s:s + chunk, None] + h[None, :])
    return jsd.clamp_(0.0, math.log(2.0))


def log_margin(logp: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    """logp: (..., V) log-probs, tok: (...) token ids. Returns log p(tok) - max_{v != tok} log p(v)."""
    lp_tok = logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    other = logp.scatter(-1, tok.unsqueeze(-1), float('-inf')).max(-1).values
    return lp_tok - other


@torch.no_grad()
def oracle_coupling(model, x, pos, tok, chunk=8):
    """Counterfactual coupling by direct conditioning (the oracle; |pos| extra forward rows).
    For each candidate j, reveal x_j := tok_j and re-run the model on the whole sequence.
    Returns
      margin: (m, m) with margin[j, i] = log-margin of tok_i at position i after revealing j
      tv:     (m, m) with tv[j, i] = TV(pi_i, pi_i | x_j = tok_j)   (DEMASK's D at y_j = top-1)
    x: (L,) one sequence; pos: (m,) masked positions; tok: (m,) proposed tokens."""
    m = pos.numel()
    margin = torch.empty(m, m, device=x.device)
    probs_cond = []
    for s in range(0, m, chunk):
        js = torch.arange(s, min(s + chunk, m), device=x.device)
        xb = x.unsqueeze(0).repeat(js.numel(), 1)
        xb[torch.arange(js.numel(), device=x.device), pos[js]] = tok[js]
        out = model(xb)
        out = out[0] if isinstance(out, tuple) else out  # DAWN's LLaDA returns (output, attn_scores)
        logits = out.logits[:, pos].float()  # (c, m, V)
        logp = F.log_softmax(logits, -1)
        margin[js] = log_margin(logp, tok.unsqueeze(0).expand(js.numel(), m))
        probs_cond.append(logp.exp())
    return margin, torch.cat(probs_cond, 0)  # second: (m_j, m_i, V) conditional distributions


@torch.no_grad()
def rotate_kv_cache(model, kv_cache):
    """Apply the rotary embedding to the cached keys once (the cache stores pre-rotation keys).
    Positions are fixed, so probes can then attend to these keys without re-rotating them per row."""
    rot = next(m for m in model.modules() if type(m).__name__ == 'RotaryEmbedding')
    out = []
    for k, v in kv_cache:
        L = k.shape[-2]
        with torch.autocast(k.device.type, enabled=False):
            sin, cos = rot.get_rotary_embedding(L, k.device)
            k_ = k.float() if rot.config.rope_full_precision else k
            out.append((rot.apply_rotary_pos_emb(sin.type_as(k_), cos.type_as(k_), k_).type_as(k), v))
    return out


@torch.no_grad()
def oracle_coupling_probe(model, x, pos, tok, block, kv_rot):
    """Fast block-local probes: same approximation as oracle_coupling_cached, computed with the
    model's probe attention path (shared rotated cache, no per-row cache copy, no re-rotation).
    kv_rot: rotate_kv_cache(model, past_key_values of the main pass)."""
    s, e = block
    m = pos.numel()
    rel = pos - s
    xb = x[s:e].unsqueeze(0).repeat(m, 1)
    xb[torch.arange(m, device=x.device), rel] = tok
    out = model(xb, past_key_values=kv_rot, use_cache=False, replace_position=('probe', s, e))
    out = out[0] if isinstance(out, tuple) else out
    logp = F.log_softmax(out.logits[:, rel].float(), -1)  # (m, m, V)
    return log_margin(logp, tok.unsqueeze(0).expand(m, m)), logp.exp()


def _probe_attn_mask(L, K, bl, s, e, device):
    """Boolean mask (K*bl, L + K*bl) for the probe queries in fused_forward(attn='sdpa'): True = may
    attend. Shared main keys outside the block [s, e) are visible to every probe row; the probe block
    keys are visible only to the queries of the same row (block-diagonal)."""
    m = torch.zeros(K * bl, L + K * bl, dtype=torch.bool, device=device)
    m[:, :L] = True
    m[:, s:e] = False
    row = torch.arange(K * bl, device=device) // bl
    m[:, L:] = row.unsqueeze(1) == row.unsqueeze(0)
    return m


try:
    import probe_attn_kernel as _PK  # fused Triton attention for cached steps (optional)
except Exception:  # Triton unavailable
    _PK = None

_FLEX = None


def _block_excl(L, s, e, device):
    m = torch.zeros(L, dtype=torch.bool, device=device); m[s:e] = True
    return m


def _shared_ctx_attention(q, k, v, excl, scale):
    """q: (1, nh, Q, hd) probe queries; k, v: (1, nh, L, hd) main keys/values; excl: (L,) keys to ignore.
    One fused attention call (flex_attention) over the shared keys; returns output (1, nh, Q, hd) and the
    log-sum-exp of the scores (1, nh, Q), for merging with the own-block part."""
    global _FLEX
    from torch.nn.attention.flex_attention import flex_attention
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=True)
    keep = ~excl

    def score_mod(score, b, h, qi, ki):
        return torch.where(keep[ki], score, -float('inf'))

    out, lse = _FLEX(q, k, v, score_mod=score_mod, scale=scale, return_lse=True)
    # flex_attention returns the log-sum-exp in base e of scale*q.k (same convention as sc_own above)
    return out, lse


@torch.no_grad()
def fused_forward(model, x, probe_rows, block, attn='manual'):
    """One forward pass that computes the main logits for x and the block-local probes together.
    x: (L,) tokens; probe_rows: (K, bl) block tokens of the K probe rows (one candidate revealed each);
    block: (s, e). All per-token layers (norms, projections, MLP, LM head) run once on the main and probe
    tokens concatenated, so every weight is read once for both. Attention is split: main tokens use the
    model's bidirectional attention; probe rows attend to the same layer's main keys/values outside the
    block plus their own block keys (the block-local probe approximation, now with no separate call).
    block may also be a LongTensor of positions (bl,): then each probe row recomputes only those positions
    (in practice the still-masked ones), and every other position, inside or outside the block, uses the
    main keys/values. This is the same approximation applied to the committed block positions too.
    Returns main logits (1, L, V) and probe logits (K, bl, V)."""
    M = model.model
    T, cfg = M.transformer, M.config
    ppos = block if torch.is_tensor(block) else None
    s, e = (None, None) if ppos is not None else block
    L, (K, bl) = x.numel(), probe_rows.shape
    d, nh = cfg.d_model, cfg.n_heads
    hd = d // nh
    assert cfg.block_group_size == 1 and not cfg.alibi and cfg.rope and cfg.effective_n_kv_heads == nh
    h = T.wte(torch.cat([x, probe_rows.reshape(-1)]))  # (L + K*bl, d)
    if cfg.input_emb_norm:
        h = h * (d ** 0.5)
    h = T.emb_drop(h)
    rot = T.blocks[0].rotary_emb
    with torch.autocast(x.device.type, enabled=False):
        sin, cos = rot.get_rotary_embedding(L, x.device)  # (1, 1, L, hd)
    if ppos is None:
        sin_b, cos_b = sin[:, :, s:e], cos[:, :, s:e]
    else:
        sin_b, cos_b = sin.index_select(2, ppos), cos.index_select(2, ppos)
    scale = 1.0 / math.sqrt(hd)
    full = torch.float32 if cfg.rope_full_precision else None
    mask = None  # built once per call (attn='sdpa'), shared by all layers
    if ppos is not None:  # shared keys to ignore: the recomputed positions (one boolean row, reused by all layers)
        excl = torch.zeros(L, dtype=torch.bool, device=x.device)
        excl[ppos] = True

    def rope(t, sn, cs):
        t_ = t.to(full) if full else t
        return rot.apply_rotary_pos_emb(sn.type_as(t_), cs.type_as(t_), t_).type_as(t)

    for blk in T.blocks:
        xn = blk.attn_norm(h)
        q, k, v = blk.q_proj(xn), blk.k_proj(xn), blk.v_proj(xn)
        qm = q[:L].view(1, L, nh, hd).transpose(1, 2); km = k[:L].view(1, L, nh, hd).transpose(1, 2)
        vm = v[:L].view(1, L, nh, hd).transpose(1, 2)
        qp = q[L:].view(K, bl, nh, hd).transpose(1, 2); kp = k[L:].view(K, bl, nh, hd).transpose(1, 2)
        vp = v[L:].view(K, bl, nh, hd).transpose(1, 2)
        with torch.autocast(x.device.type, enabled=False):
            qm, km = rope(qm, sin, cos), rope(km, sin, cos)
            qp, kp = rope(qp, sin_b, cos_b), rope(kp, sin_b, cos_b)
        att_m = F.scaled_dot_product_attention(qm, km, vm, is_causal=False)  # (1, nh, L, hd)
        qflat = qp.transpose(0, 1).reshape(1, nh, K * bl, hd)
        if attn == 'lse':
            # Shared-context probe attention with the fused efficient kernel (no compilation): all K*bl probe queries
            # against the main keys in one call, the recomputed positions removed by an additive key bias, and the
            # call's log-sum-exp merged with the exact own-block part. Same math as 'manual'.
            if mask is None:
                kb_ = torch.zeros(1, 1, 1, -(-L // 16) * 16, device=x.device, dtype=vm.dtype)
                kb_[..., :L][..., (excl if ppos is not None else _block_excl(L, s, e, x.device))] = float('-inf')
                mask = kb_[..., :L]
            a_sh, l_sh = _attn_lse(qflat, km, vm, scale, mask)
            a_sh = a_sh[0].view(nh, K, bl, hd).transpose(0, 1); l_sh = l_sh[0].view(nh, K, bl).transpose(0, 1)
            sc_own = torch.matmul(qp, kp.transpose(-1, -2)).float() * scale
            att_own = torch.matmul(torch.softmax(sc_own, -1).to(vp.dtype), vp)
            att_p = _merge(a_sh, l_sh, att_own, torch.logsumexp(sc_own, -1), vm.dtype)
        elif attn == 'shared':
            # Shared-context probe attention: every probe row sees the same main keys/values outside the
            # recomputed positions, so all K*bl probe queries go through ONE fused attention call over the
            # main keys (keys read once, no float32 score tensor), which also returns its log-sum-exp. The
            # tiny own-block part (each row against its own bl keys) is exact, and the two are merged by
            # their log-sum-exps, which gives the same softmax as attending to both at once.
            att_sh, lse_sh = _shared_ctx_attention(qflat, km, vm, excl if ppos is not None else _block_excl(L, s, e, x.device), scale)
            att_sh = att_sh.view(nh, K, bl, hd).transpose(0, 1); lse_sh = lse_sh.view(nh, K, bl).transpose(0, 1)  # (K, nh, bl[, hd])
            sc_own = torch.matmul(qp, kp.transpose(-1, -2)).float() * scale  # (K, nh, bl, bl)
            lse_own = torch.logsumexp(sc_own, -1)
            att_own = torch.matmul(torch.softmax(sc_own, -1).to(vp.dtype), vp)
            lse = torch.logaddexp(lse_sh, lse_own)
            att_p = (torch.exp(lse_sh - lse).unsqueeze(-1) * att_sh.float() + torch.exp(lse_own - lse).unsqueeze(-1) * att_own.float()).to(vm.dtype)
        elif attn == 'sdpa':
            # One fused attention call for all probe queries: keys = main keys + every probe row's
            # block keys, with a mask that keeps each row to its own block keys (same math).
            kflat = kp.transpose(0, 1).reshape(1, nh, K * bl, hd)
            vflat = vp.transpose(0, 1).reshape(1, nh, K * bl, hd)
            if mask is None:
                assert ppos is None, "attn='sdpa' supports block ranges only"
                mask = _probe_attn_mask(L, K, bl, s, e, x.device)
            att_p = F.scaled_dot_product_attention(qflat, torch.cat([km, kflat], 2), torch.cat([vm, vflat], 2),
                                                   attn_mask=mask).view(nh, K, bl, hd).transpose(0, 1)
        else:
            # Shared part as one matmul: all K*bl probe queries in one row, so the main keys/values are
            # read once (a broadcast batched matmul would copy them K times per layer).
            sc_sh = (torch.matmul(qflat, km.transpose(-1, -2)).float() * scale).view(nh, K, bl, L).transpose(0, 1)
            if ppos is None:
                sc_sh[..., s:e] = float('-inf')  # (K, nh, bl, L); block keys come from each row's own k
            else:
                sc_sh = sc_sh.masked_fill(excl, float('-inf'))  # recomputed positions come from each row's own k
            sc_own = torch.matmul(qp, kp.transpose(-1, -2)).float() * scale  # (K, nh, bl, bl)
            p = torch.softmax(torch.cat([sc_sh, sc_own], -1), -1).to(vm.dtype)
            p_sh = p[..., :L].transpose(0, 1).reshape(1, nh, K * bl, L)
            att_p = torch.matmul(p_sh, vm).view(nh, K, bl, hd).transpose(0, 1) + torch.matmul(p[..., L:], vp)  # (K, nh, bl, hd)
        att = torch.cat([att_m.transpose(1, 2).reshape(L, d), att_p.transpose(1, 2).reshape(K * bl, d)])
        h = h + blk.dropout(blk.attn_out(att))
        xf = blk.ff_norm(h)
        xf = blk.act(blk.ff_proj(xf)) * blk.up_proj(xf)
        h = h + blk.dropout(blk.ff_out(xf))
    h = T.ln_f(h)
    logits = F.linear(h, T.wte.weight, None) if cfg.weight_tying else T.ff_out(h)
    if cfg.scale_logits:
        logits.mul_(1 / math.sqrt(d))
    return logits[:L].unsqueeze(0), logits[L:].view(K, bl, -1)


@torch.no_grad()
def cached_probe_logits(model, x, probe_rows, ppos, kv_rot):
    """Probe rows against a dual cache, with the shared-context attention. x: (L,) tokens; ppos: (m,) still-masked
    positions of the block (sorted); probe_rows: (K, m) tokens at ppos with one candidate revealed per row;
    kv_rot: per-layer (k, v), (1, nh, L, hd), keys already rotated, holding the current state of the sequence
    (the block's keys/values as of this step). Each probe row recomputes only ppos; every other position, inside or
    outside the block, uses the cached keys/values (the same approximation as fused_forward with positions).
    Returns probe logits (K, m, V)."""
    M = model.model; T, cfg = M.transformer, M.config
    d, nh = cfg.d_model, cfg.n_heads; hd = d // nh
    K, m = probe_rows.shape
    L = kv_rot[0][0].shape[-2]
    h = T.wte(probe_rows.reshape(-1))
    if cfg.input_emb_norm:
        h = h * (d ** 0.5)
    rot = T.blocks[0].rotary_emb
    with torch.autocast(x.device.type, enabled=False):
        sin, cos = rot.get_rotary_embedding(L, x.device)
    sin_b, cos_b = sin.index_select(2, ppos), cos.index_select(2, ppos)
    full = torch.float32 if cfg.rope_full_precision else None
    excl = torch.zeros(L, dtype=torch.bool, device=x.device); excl[ppos] = True
    scale = 1.0 / math.sqrt(hd)

    def rope(t):
        t_ = t.to(full) if full else t
        return rot.apply_rotary_pos_emb(sin_b.type_as(t_), cos_b.type_as(t_), t_).type_as(t)

    for li, blk in enumerate(T.blocks):
        xn = blk.attn_norm(h)
        q, k, v = blk.q_proj(xn), blk.k_proj(xn), blk.v_proj(xn)
        q = rope(q.view(K, m, nh, hd).transpose(1, 2)); k = rope(k.view(K, m, nh, hd).transpose(1, 2))
        v = v.view(K, m, nh, hd).transpose(1, 2)
        k_sh, v_sh = kv_rot[li]
        att_sh, lse_sh = _shared_ctx_attention(q.transpose(0, 1).reshape(1, nh, K * m, hd), k_sh, v_sh, excl, scale)
        att_sh = att_sh.view(nh, K, m, hd).transpose(0, 1); lse_sh = lse_sh.view(nh, K, m).transpose(0, 1)
        sc_own = torch.matmul(q, k.transpose(-1, -2)).float() * scale
        lse_own = torch.logsumexp(sc_own, -1)
        att_own = torch.matmul(torch.softmax(sc_own, -1).to(v.dtype), v)
        lse = torch.logaddexp(lse_sh, lse_own)
        att = (torch.exp(lse_sh - lse).unsqueeze(-1) * att_sh.float() + torch.exp(lse_own - lse).unsqueeze(-1) * att_own.float()).to(v.dtype)
        h = h + blk.dropout(blk.attn_out(att.transpose(1, 2).reshape(K * m, d)))
        xf = blk.ff_norm(h)
        h = h + blk.dropout(blk.ff_out(blk.act(blk.ff_proj(xf)) * blk.up_proj(xf)))
    h = T.ln_f(h)
    lg = F.linear(h, T.wte.weight, None) if cfg.weight_tying else T.ff_out(h)
    if cfg.scale_logits:
        lg.mul_(1 / math.sqrt(d))
    return lg.view(K, m, -1)


def block_context(kv_rot, s, e):
    """Per-layer keys/values of the dual cache OUTSIDE the block [s, e), as contiguous tensors (1, nh, L - bl, hd),
    keys rotated. Built once per block: under the dual cache the context does not change within a block."""
    return [(torch.cat([k[:, :, :s], k[:, :, e:]], 2), torch.cat([v[:, :, :s], v[:, :, e:]], 2)) for k, v in kv_rot]


def _attn_lse(q, k, v, scale, bias=None):
    """Attention with its log-sum-exp (natural log of sum exp(scale q.k + bias)), via the fused efficient kernel:
    q (B, nh, Q, hd), k/v (B, nh, Lk, hd), optional additive bias broadcastable to (B, nh, Q, Lk) (used to drop
    padded keys). Returns out (B, nh, Q, hd) and lse (B, nh, Q) float32."""
    if bias is not None:
        bias = bias.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, bias, True, 0.0, False, scale=scale)
    return out, lse[..., :q.shape[-2]]


def _merge(a1, l1, a2, l2, dtype):
    l = torch.logaddexp(l1, l2)
    return (torch.exp(l1 - l).unsqueeze(-1) * a1.float() + torch.exp(l2 - l).unsqueeze(-1) * a2.float()).to(dtype)


@torch.no_grad()
def cached_step(model, x, s, e, ctx, probe_rows=None, ppos=None, probes_only=False):
    """One dual-cache step for the block [s, e), with optional probe rows, in ONE pass (every weight read once).
    ctx: block_context(...) of the current block (cache outside the block, keys rotated).
    Main block row: attends to ctx plus its own fresh block keys (the dual-cache step of Fast-dLLM).
    Probe rows (K, m): the still-masked positions ppos (sorted, inside the block) with one candidate revealed; each
    attends to ctx, to this step's keys of the committed block positions (from the main row), and to its own keys
    at ppos (the same approximation as fused_forward with positions).
    Attention over ctx is one unmasked fused call for all rows; the small in-block part is exact; they are merged
    by log-sum-exp. probes_only (block start, nothing committed yet): probe rows only, the main logits come from the
    full pass. Returns block logits (1, bl, V) or None, and probe logits (K, m, V) or None."""
    M = model.model; T, cfg = M.transformer, M.config
    d, nh = cfg.d_model, cfg.n_heads; hd = d // nh
    bl = e - s; K = 0 if probe_rows is None else probe_rows.shape[0]; m = 0 if K == 0 else probe_rows.shape[1]
    nb = 0 if probes_only else bl
    if K:
        comm = torch.ones(bl, dtype=torch.bool, device=x.device); comm[ppos - s] = False  # committed block slots
        assert not (probes_only and comm.any()), 'probes_only needs a fully masked block'
    toks = [x[s:e]] if nb else []
    if K:
        toks.append(probe_rows.reshape(-1))
    h = T.wte(torch.cat(toks))
    if cfg.input_emb_norm:
        h = h * (d ** 0.5)
    rot = T.blocks[0].rotary_emb
    with torch.autocast(x.device.type, enabled=False):
        sin, cos = rot.get_rotary_embedding(e, x.device)
    full = torch.float32 if cfg.rope_full_precision else None

    def rope(t_in, idx):
        t_ = t_in.to(full) if full else t_in
        sn, cs = sin.index_select(2, idx), cos.index_select(2, idx)
        return rot.apply_rotary_pos_emb(sn.type_as(t_), cs.type_as(t_), t_).type_as(t_in)

    bpos = torch.arange(s, e, device=x.device)
    scale = 1.0 / math.sqrt(hd)
    for li, blk in enumerate(T.blocks):
        xn = blk.attn_norm(h)
        q, k, v = blk.q_proj(xn), blk.k_proj(xn), blk.v_proj(xn)
        k_ctx, v_ctx = ctx[li]
        # shared part: every query of the step against the same context, one fused unmasked call
        qs = []
        if nb:
            qb = rope(q[:nb].view(1, nb, nh, hd).transpose(1, 2), bpos)
            kb = rope(k[:nb].view(1, nb, nh, hd).transpose(1, 2), bpos)
            vb = v[:nb].view(1, nb, nh, hd).transpose(1, 2)
            qs.append(qb[0])
        if K:
            qp = rope(q[nb:].view(K, m, nh, hd).transpose(1, 2), ppos)
            kp = rope(k[nb:].view(K, m, nh, hd).transpose(1, 2), ppos)
            vp = v[nb:].view(K, m, nh, hd).transpose(1, 2)
            qs.append(qp.transpose(0, 1).reshape(nh, K * m, hd))
        a_sh, l_sh = _attn_lse(torch.cat(qs, 1).unsqueeze(0), k_ctx, v_ctx, scale)
        a_sh, l_sh = a_sh[0], l_sh[0]  # (nh, nb + K*m, hd), (nh, nb + K*m)
        att = []
        if nb:  # main block row: own fresh block keys
            sc = torch.matmul(qb, kb.transpose(-1, -2)).float() * scale
            a_own = torch.matmul(torch.softmax(sc, -1).to(vb.dtype), vb)[0]
            att.append(_merge(a_sh[:, :nb], l_sh[:, :nb], a_own, torch.logsumexp(sc, -1)[0], vb.dtype).transpose(0, 1).reshape(nb, d))
        if K:  # probe rows: committed block keys of this step (from the main row) + own keys at ppos
            ks, vs = kp, vp
            if nb and comm.any():
                kc = kb[:, :, comm].expand(K, -1, -1, -1); vc = vb[:, :, comm].expand(K, -1, -1, -1)
                ks, vs = torch.cat([kc, kp], 2), torch.cat([vc, vp], 2)
            sc = torch.matmul(qp, ks.transpose(-1, -2)).float() * scale  # (K, nh, m, c + m)
            a_own = torch.matmul(torch.softmax(sc, -1).to(vs.dtype), vs)  # (K, nh, m, hd)
            a_p = a_sh[:, nb:].view(nh, K, m, hd).transpose(0, 1); l_p = l_sh[:, nb:].view(nh, K, m).transpose(0, 1)
            att.append(_merge(a_p, l_p, a_own, torch.logsumexp(sc, -1), vs.dtype).transpose(1, 2).reshape(K * m, d))
        h = h + blk.dropout(blk.attn_out(torch.cat(att) if len(att) > 1 else att[0]))
        xf = blk.ff_norm(h)
        h = h + blk.dropout(blk.ff_out(blk.act(blk.ff_proj(xf)) * blk.up_proj(xf)))
    h = T.ln_f(h)
    lg = F.linear(h, T.wte.weight, None) if cfg.weight_tying else T.ff_out(h)
    if cfg.scale_logits:
        lg.mul_(1 / math.sqrt(d))
    return (lg[:nb].unsqueeze(0) if nb else None), (lg[nb:].view(K, m, -1) if K else None)


@torch.no_grad()
def cached_step_depth(model, x, s, e, ctx, probe_rows, ppos, L0):
    """cached_step with depth-sparse probe rows (an approximation, for a fidelity test). Probe row r recomputes only
    its candidate position j_r in layers 0..L0-1, reading the main row's keys/values of the same layer at every other
    block position (committed or masked); from layer L0 on it recomputes all masked positions as cached_step does,
    starting from the main row's hidden states at L0 (its own state at j_r). L0 = 0 is cached_step exactly; L0 =
    n_layers recomputes only the candidate. Needs a main row (nothing to seed from at a block start).
    Returns block logits (1, bl, V) and probe logits (K, m, V)."""
    M = model.model; T, cfg = M.transformer, M.config
    d, nh = cfg.d_model, cfg.n_heads; hd = d // nh
    bl = e - s; K, m = probe_rows.shape
    comm = torch.ones(bl, dtype=torch.bool, device=x.device); comm[ppos - s] = False
    cj = (probe_rows != x[ppos].unsqueeze(0)).float().argmax(1)  # index in ppos of each row's candidate
    h = T.wte(torch.cat([x[s:e], probe_rows[torch.arange(K), cj]]))  # phase 1: main row + one token per probe
    if cfg.input_emb_norm:
        h = h * (d ** 0.5)
    rot = T.blocks[0].rotary_emb
    with torch.autocast(x.device.type, enabled=False):
        sin, cos = rot.get_rotary_embedding(e, x.device)
    full = torch.float32 if cfg.rope_full_precision else None

    def rope(t_in, idx):
        t_ = t_in.to(full) if full else t_in
        return rot.apply_rotary_pos_emb(sin.index_select(2, idx).type_as(t_), cos.index_select(2, idx).type_as(t_), t_).type_as(t_in)

    def rope_rows(t_in, idx):  # t_in (K, nh, n, hd), idx (K, n): a different position per row and token
        t_ = t_in.to(full) if full else t_in
        return rot.apply_rotary_pos_emb(sin[0, 0][idx].unsqueeze(1).type_as(t_), cos[0, 0][idx].unsqueeze(1).type_as(t_), t_).type_as(t_in)

    bpos = torch.arange(s, e, device=x.device)
    scale = 1.0 / math.sqrt(hd)
    wide = False
    for li, blk in enumerate(T.blocks):
        if li == L0:  # phase 2: grow each probe row to all masked positions, seeded from the main row's states
            hp = h[:bl][ppos - s].unsqueeze(0).repeat(K, 1, 1)                     # (K, m, d)
            hp[torch.arange(K), cj] = h[bl:bl + K]                                 # the candidate keeps its own state
            h = torch.cat([h[:bl], hp.reshape(K * m, d)]); wide = True
        n = m if wide else 1
        xn = blk.attn_norm(h)
        q, k, v = blk.q_proj(xn), blk.k_proj(xn), blk.v_proj(xn)
        k_ctx, v_ctx = ctx[li]
        qb = rope(q[:bl].view(1, bl, nh, hd).transpose(1, 2), bpos)
        kb = rope(k[:bl].view(1, bl, nh, hd).transpose(1, 2), bpos)
        vb = v[:bl].view(1, bl, nh, hd).transpose(1, 2)
        p_idx = ppos.unsqueeze(0).expand(K, -1) if wide else ppos[cj].unsqueeze(1)  # (K, n) positions of the probe tokens
        qp = rope_rows(q[bl:].view(K, n, nh, hd).transpose(1, 2), p_idx)
        kp = rope_rows(k[bl:].view(K, n, nh, hd).transpose(1, 2), p_idx)
        vp = v[bl:].view(K, n, nh, hd).transpose(1, 2)
        qs = torch.cat([qb[0], qp.transpose(0, 1).reshape(nh, K * n, hd)], 1).unsqueeze(0)
        a_sh, l_sh = _attn_lse(qs, k_ctx, v_ctx, scale); a_sh, l_sh = a_sh[0], l_sh[0]
        sc = torch.matmul(qb, kb.transpose(-1, -2)).float() * scale
        a_own = torch.matmul(torch.softmax(sc, -1).to(vb.dtype), vb)[0]
        att = [_merge(a_sh[:, :bl], l_sh[:, :bl], a_own, torch.logsumexp(sc, -1)[0], vb.dtype).transpose(0, 1).reshape(bl, d)]
        # in-block keys of a probe row: the main row's block keys except where the row has its own (masked slots in
        # phase 2, the candidate's slot in phase 1), plus its own keys
        vis = comm.unsqueeze(0).expand(K, -1).clone() if wide else torch.ones(K, bl, dtype=torch.bool, device=x.device)
        if not wide:
            vis[torch.arange(K), ppos[cj] - s] = False
        bias = torch.where(vis, 0.0, float('-inf')).view(K, 1, 1, bl)
        ks = torch.cat([kb.expand(K, -1, -1, -1), kp], 2); vs = torch.cat([vb.expand(K, -1, -1, -1), vp], 2)
        sc = torch.matmul(qp, ks.transpose(-1, -2)).float() * scale + torch.cat([bias.expand(-1, nh, n, -1), torch.zeros(K, nh, n, n, device=x.device)], -1)
        a_o = torch.matmul(torch.softmax(sc, -1).to(vs.dtype), vs)
        a_p = a_sh[:, bl:].view(nh, K, n, hd).transpose(0, 1); l_p = l_sh[:, bl:].view(nh, K, n).transpose(0, 1)
        att.append(_merge(a_p, l_p, a_o, torch.logsumexp(sc, -1), vs.dtype).transpose(1, 2).reshape(K * n, d))
        h = h + blk.dropout(blk.attn_out(torch.cat(att)))
        xf = blk.ff_norm(h)
        h = h + blk.dropout(blk.ff_out(blk.act(blk.ff_proj(xf)) * blk.up_proj(xf)))
    if not wide:  # L0 = n_layers: masked positions other than the candidate keep the main row's final states
        hp = h[:bl][ppos - s].unsqueeze(0).repeat(K, 1, 1); hp[torch.arange(K), cj] = h[bl:bl + K]
        h = torch.cat([h[:bl], hp.reshape(K * m, d)])
    h = T.ln_f(h)
    lg = F.linear(h, T.wte.weight, None) if cfg.weight_tying else T.ff_out(h)
    if cfg.scale_logits:
        lg.mul_(1 / math.sqrt(d))
    return lg[:bl].unsqueeze(0), lg[bl:].view(K, m, -1)


class CachedStepGraph:
    """cached_step replayed from CUDA graphs, to remove per-op launch overhead (the step is ~100 small kernels per
    layer at batch size 1). Same math as cached_step: the main block row attends to the context outside the block
    plus its own fresh block keys; probe rows attend to the context, to this step's keys of the committed block
    positions, and to their own keys at the still-masked positions. Shapes are padded so that a few graphs cover
    every step: probe rows to K, masked positions to a multiple of 4 (up to the block length). Padding is excluded by
    additive -inf biases, and padded outputs are dropped. One instance per model; graphs are keyed by
    (context length, block length, K, padded m). Used for every cached decoder (Fast-dLLM, MF, ours), so cached
    comparisons differ only in the decoding rule."""

    def __init__(self, model, K, pad=256, kernel=False, mp_step=4, window=0, rule=None):
        self.model, self.K, self.pad, self.kernel = model, K, pad, kernel
        self.rule = rule  # (tau, R, mask_id): run the measured-coupling commit rule inside the graph (rule_step)
        self.window = window  # > 0: short probe rows, each recomputing only the masked positions within +-window of its candidate
        self.mp_step = mp_step  # masked positions are padded to a multiple of this (fewer graph shapes when larger)
        M = model.model; self.T, self.cfg = M.transformer, M.config
        self.d, self.nh = self.cfg.d_model, self.cfg.n_heads; self.hd = self.d // self.nh
        self.rot = self.T.blocks[0].rotary_emb
        self.graphs = {}
        self.pool = None
        self._ctxbuf = {}  # (context length, block length) -> static per-layer context buffers, shared across graphs

    def _forward(self, st):
        """The step on static buffers (captured into a graph). st: dict of static tensors."""
        T, cfg, d, nh, hd, K = self.T, self.cfg, self.d, self.nh, self.hd, self.K
        bl, mp = st['bl'], st['mp']
        h = T.wte(st['toks'])  # (bl + K*mp,)
        if cfg.input_emb_norm:
            h = h * (d ** 0.5)
        scale = 1.0 / math.sqrt(hd)
        full = torch.float32 if cfg.rope_full_precision else None

        def rope(t_in, sn, cs):
            t_ = t_in.to(full) if full else t_in
            return self.rot.apply_rotary_pos_emb(sn.type_as(t_), cs.type_as(t_), t_).type_as(t_in)

        for li, blk in enumerate(T.blocks):
            xn = blk.attn_norm(h)
            q, k, v = blk.q_proj(xn), blk.k_proj(xn), blk.v_proj(xn)
            if self.kernel == 4:  # rotary fused into the kernel, output already token-major: no copies per layer
                att = _PK.probe_attention4(q, k, v, st['k_ctx'][li][0], st['v_ctx'][li][0], st['ctx_bias_f'], st['pos_b'], st['pos_p'],
                                           st['cos_t'], st['sin_t'], st['committed'], st['pvalid'], bl, K, mp, scale)
                h = h + blk.attn_out(att)
                xf = blk.ff_norm(h)
                h = h + blk.ff_out(blk.act(blk.ff_proj(xf)) * blk.up_proj(xf))
                continue
            qb = rope(q[:bl].view(1, bl, nh, hd).transpose(1, 2), st['sin_b'], st['cos_b'])
            kb = rope(k[:bl].view(1, bl, nh, hd).transpose(1, 2), st['sin_b'], st['cos_b'])
            vb = v[:bl].view(1, bl, nh, hd).transpose(1, 2)
            att = []
            qs = [qb[0]]
            if K:
                qp = rope(q[bl:].view(K, mp, nh, hd).transpose(1, 2), st['sin_p'], st['cos_p'])
                kp = rope(k[bl:].view(K, mp, nh, hd).transpose(1, 2), st['sin_p'], st['cos_p'])
                vp = v[bl:].view(K, mp, nh, hd).transpose(1, 2)
                qs.append(qp.transpose(0, 1).reshape(nh, K * mp, hd))
            if self.kernel == 3:  # three-source Triton kernel: block row and probe rows in one launch, no key copies
                ob, op = _PK.probe_attention3(qb[0], qp if K else None, st['k_ctx'][li][0], st['v_ctx'][li][0], st['ctx_bias_f'],
                                              kb[0], vb[0], kp if K else None, vp if K else None, st['committed'], st['pvalid'], scale)
                att = ob.transpose(0, 1).reshape(bl, d)
                if K:
                    att = torch.cat([att, op.transpose(1, 2).reshape(K * mp, d)])
                h = h + blk.attn_out(att)
                xf = blk.ff_norm(h)
                h = h + blk.ff_out(blk.act(blk.ff_proj(xf)) * blk.up_proj(xf))
                continue
            if self.kernel:  # fused Triton attention (probe_attn_kernel.py): shared context + own keys, one pass
                ksh, vsh = st['k_ctx'][li][0], st['v_ctx'][li][0]
                att.append(_PK.probe_attention(qb, ksh, vsh, st['ctx_bias_f'], kb, vb, st['bias_b'], scale)[0].transpose(0, 1).reshape(bl, d))
                if K:
                    ks = torch.cat([kb.expand(K, -1, -1, -1), kp], 2); vs = torch.cat([vb.expand(K, -1, -1, -1), vp], 2)
                    att.append(_PK.probe_attention(qp, ksh, vsh, st['ctx_bias_f'], ks, vs, st['bias_p_f'], scale).transpose(1, 2).reshape(K * mp, d))
                h = h + blk.attn_out(torch.cat(att) if K else att[0])
                xf = blk.ff_norm(h)
                h = h + blk.ff_out(blk.act(blk.ff_proj(xf)) * blk.up_proj(xf))
                continue
            a_sh, l_sh = _attn_lse(torch.cat(qs, 1).unsqueeze(0), st['k_ctx'][li], st['v_ctx'][li], scale, st['ctx_bias'])
            a_sh, l_sh = a_sh[0], l_sh[0]
            sc = torch.matmul(qb, kb.transpose(-1, -2)).float() * scale
            a_own = torch.matmul(torch.softmax(sc, -1).to(vb.dtype), vb)[0]
            att.append(_merge(a_sh[:, :bl], l_sh[:, :bl], a_own, torch.logsumexp(sc, -1)[0], vb.dtype).transpose(0, 1).reshape(bl, d))
            if K:
                # keys a probe row sees inside the block: the committed slots of this step's block row (bias 0 on
                # committed slots, -inf elsewhere) and its own keys at the masked positions (-inf on padding)
                ks = torch.cat([kb.expand(K, -1, -1, -1), kp], 2); vs = torch.cat([vb.expand(K, -1, -1, -1), vp], 2)
                sc = torch.matmul(qp, ks.transpose(-1, -2)).float() * scale + st['bias_p']  # (K, nh, mp, bl + mp)
                a_own = torch.matmul(torch.softmax(sc, -1).to(vs.dtype), vs)
                a_p = a_sh[:, bl:].view(nh, K, mp, hd).transpose(0, 1); l_p = l_sh[:, bl:].view(nh, K, mp).transpose(0, 1)
                att.append(_merge(a_p, l_p, a_own, torch.logsumexp(sc, -1), vs.dtype).transpose(1, 2).reshape(K * mp, d))
            h = h + blk.attn_out(torch.cat(att) if K else att[0])
            xf = blk.ff_norm(h)
            h = h + blk.ff_out(blk.act(blk.ff_proj(xf)) * blk.up_proj(xf))
        h = T.ln_f(h)
        lg = F.linear(h, T.wte.weight, None) if cfg.weight_tying else T.ff_out(h)
        if cfg.scale_logits:
            lg.mul_(1 / math.sqrt(d))
        st['out'].copy_(lg)
        if self.rule is not None:
            tau, R_, mask_id = self.rule
            new_tok, v, c, left = graph_commit_rule(lg[:bl], lg[bl:].view(K, mp, -1), st['toks'][:bl], st['slot_pos'], st['slot_valid'],
                                                    st['cand'], st['cand_tok'], st['cand_valid'], tau, R_, mask_id)
            st['new_tok'].copy_(new_tok); st['v_out'].copy_(v); st['c_out'].copy_(c); st['left'].copy_(left)

    def _build(self, Lc, bl, mp, like):
        dev = like.device; K, nh, hd, d = self.K, self.nh, self.hd, self.d
        V = self.T.wte.weight.shape[0]
        st = {'bl': bl, 'mp': mp,
              'toks': torch.zeros(bl + K * mp, dtype=torch.long, device=dev),
              'sin_b': torch.zeros(1, 1, bl, hd, device=dev, dtype=torch.float32), 'cos_b': torch.zeros(1, 1, bl, hd, device=dev, dtype=torch.float32),
              'sin_p': torch.zeros(K if self.window else 1, 1, mp, hd, device=dev, dtype=torch.float32),
              'cos_p': torch.zeros(K if self.window else 1, 1, mp, hd, device=dev, dtype=torch.float32),
              'bias_p': torch.zeros(1, 1, 1, bl + mp, device=dev, dtype=torch.float32),
              'k_ctx': self._ctxbuf[(Lc, bl)][0], 'v_ctx': self._ctxbuf[(Lc, bl)][1],  # shared by every padded m
              'ctx_bias': self._ctxbuf[(Lc, bl)][2], 'ctx_bias_f': self._ctxbuf[(Lc, bl)][3],
              'bias_b': torch.zeros(1, bl, device=dev, dtype=torch.float32),  # main row sees all its own block keys
              'committed': torch.zeros((K, bl) if self.window else (bl,), device=dev, dtype=torch.int32),  # block slots read from the main row
              'pvalid': torch.zeros((K, max(mp, 1)) if self.window else (max(mp, 1),), device=dev, dtype=torch.int32),  # real recomputed slots
              'bias_p_f': torch.zeros(K if K else 1, bl + mp, device=dev, dtype=torch.float32),
              'out': torch.zeros(bl + K * mp, V, device=dev, dtype=like.dtype),
              'pos_b': torch.zeros(bl, device=dev, dtype=torch.int32), 'pos_p': torch.zeros(max(mp, 1), device=dev, dtype=torch.int32)}
        if self.kernel == 4:  # rotary tables for the fused kernel (float32, (Lmax, hd)), shared by all graphs
            st['cos_t'], st['sin_t'] = self._tables(dev)
        if self.rule is not None:  # inputs and outputs of the in-graph commit rule
            st.update({'slot_pos': torch.full((mp,), bl, device=dev, dtype=torch.long), 'slot_valid': torch.zeros(mp, device=dev, dtype=torch.bool),
                       'cand': torch.zeros(K, device=dev, dtype=torch.long), 'cand_tok': torch.zeros(K, device=dev, dtype=torch.long),
                       'cand_valid': torch.zeros(K, device=dev, dtype=torch.bool),
                       'new_tok': torch.zeros(bl, device=dev, dtype=torch.long), 'v_out': torch.zeros(mp, device=dev, dtype=torch.long),
                       'c_out': torch.zeros(mp, device=dev, dtype=torch.float32), 'left': torch.zeros((), device=dev, dtype=torch.long)})
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_), torch.no_grad():
            for _ in range(2):
                self._forward(st)  # warm-up (allocations, kernel selection) outside the graph
        torch.cuda.current_stream().wait_stream(s_)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool), torch.no_grad():
            self._forward(st)
        self.pool = g.pool()
        st['graph'] = g
        return st

    def _tables(self, dev, Lmax=4096):
        """Rotary cos/sin as (Lmax, hd) float32 tables (the model's own get_rotary_embedding), built once."""
        if getattr(self, '_cs', None) is None:
            with torch.autocast(dev.type, enabled=False):
                sin, cos = self.rot.get_rotary_embedding(Lmax, dev)
            self._cs = (cos[0, 0].float().contiguous(), sin[0, 0].float().contiguous())
        return self._cs

    @torch.no_grad()
    def set_context(self, ctx, s, e):
        """Bind the block's context (block_context(...)); copied into the static buffers once per block."""
        n, bl = ctx[0][0].shape[-2], e - s
        Lc = -(-n // self.pad) * self.pad  # padded context length: graphs are reused across prompts of similar length
        if (Lc, bl) not in self._ctxbuf:
            sh = list(ctx[0][0].shape); sh[-2] = Lc
            self._ctxbuf[(Lc, bl)] = ([ctx[0][0].new_zeros(sh) for _ in ctx], [ctx[0][1].new_zeros(sh) for _ in ctx],
                                      torch.zeros(1, 1, 1, Lc, device=ctx[0][0].device, dtype=ctx[0][0].dtype),
                                      torch.zeros(Lc, device=ctx[0][0].device, dtype=torch.float32))
        kbuf, vbuf, bias, bias_f = self._ctxbuf[(Lc, bl)]
        bias_f.zero_(); bias_f[n:] = float('-inf')
        for li, (kc, vc) in enumerate(ctx):
            kbuf[li][:, :, :n].copy_(kc); vbuf[li][:, :, :n].copy_(vc)
        bias.zero_(); bias[..., n:] = float('-inf')  # padded keys never attended
        self._ctx, self._key = ctx, (Lc, bl)

    @torch.no_grad()
    def step(self, x, s, e, probe_rows=None, ppos=None):
        """Same inputs and outputs as cached_step(model, x, s, e, ctx, probe_rows, ppos) with the bound context.
        probe_rows may have fewer than K rows (padded with copies of row 0, outputs dropped)."""
        Lc, bl = self._key
        m = 0 if ppos is None else ppos.numel()
        if self.window and self.K and m:
            return self._step_windowed(x, s, e, probe_rows, ppos)
        mp = (max(self.mp_step, -(-m // self.mp_step) * self.mp_step) if self.K else 4)  # padded masked count (graph shape)
        key = (Lc, bl, mp)
        if key not in self.graphs:
            self.graphs[key] = self._build(Lc, bl, mp, self._ctx[0][0])
        st = self.graphs[key]
        if self.kernel == 4:  # only what the fused kernel reads; no host syncs
            return self._step4(st, x, s, e, probe_rows, ppos, m, mp)
        with torch.autocast(x.device.type, enabled=False):
            sin, cos = self.rot.get_rotary_embedding(e, x.device)
        st['sin_b'].copy_(sin[:, :, s:e]); st['cos_b'].copy_(cos[:, :, s:e])
        st['toks'][:bl].copy_(x[s:e])
        nP = 0
        if self.K and m:
            idx = torch.cat([ppos, ppos[-1:].expand(mp - m)])  # padded positions: rotary of the last one (masked out)
            st['sin_p'].copy_(sin.index_select(2, idx)); st['cos_p'].copy_(cos.index_select(2, idx))
            nP = probe_rows.shape[0]
            rows = torch.cat([probe_rows, probe_rows[:1].expand(self.K - nP, -1)]) if nP < self.K else probe_rows
            rows = torch.cat([rows, rows[:, -1:].expand(-1, mp - m)], 1)
            st['toks'][bl:].copy_(rows.reshape(-1))
            bias = torch.full((bl + mp,), float('-inf'), device=x.device)
            comm = torch.ones(bl, dtype=torch.bool, device=x.device); comm[ppos - s] = False
            bias[:bl][comm] = 0.0; bias[bl:bl + m] = 0.0
            st['bias_p'].copy_(bias.view(1, 1, 1, -1))
            st['bias_p_f'].copy_(bias.view(1, -1).expand(self.K, -1))
            st['committed'].copy_(comm.to(torch.int32)); pv = torch.zeros(mp, dtype=torch.int32, device=x.device); pv[:m] = 1
            st['pvalid'].copy_(pv)
        st['graph'].replay()
        out = st['out']
        blk_logits = out[:bl].unsqueeze(0).clone()
        pl = out[bl:].view(self.K, mp, -1)[:nP, :m].clone() if nP else None
        return blk_logits, pl

    def _step4(self, st, x, s, e, probe_rows, ppos, m, mp):
        """Input preparation of step() for kernel 4: block tokens and positions, probe rows padded to (K, mp), the
        committed-slot and valid-slot masks. Padding (rows, slots) is excluded inside the kernel; padded outputs dropped."""
        bl = e - s; dev = x.device
        st['toks'][:bl].copy_(x[s:e])
        st['pos_b'].copy_(torch.arange(s, e, device=dev, dtype=torch.int32))
        nP = 0
        if self.K and m:
            nP = probe_rows.shape[0]
            rows = torch.cat([probe_rows, probe_rows[:1].expand(self.K - nP, -1)]) if nP < self.K else probe_rows
            rows = torch.cat([rows, rows[:, -1:].expand(-1, mp - m)], 1)
            st['toks'][bl:].copy_(rows.reshape(-1))
            st['pos_p'].copy_(torch.cat([ppos, ppos[-1:].expand(mp - m)]).to(torch.int32))
            comm = torch.ones(bl, dtype=torch.int32, device=dev); comm[ppos - s] = 0
            st['committed'].copy_(comm)
            pv = torch.zeros(mp, dtype=torch.int32, device=dev); pv[:m] = 1
            st['pvalid'].copy_(pv)
        st['graph'].replay()
        out = st['out']
        return out[:bl].unsqueeze(0).clone(), (out[bl:].view(self.K, mp, -1)[:nP, :m].clone() if nP else None)

    @torch.no_grad()
    def rule_step(self, x, s, e, masked, sel, cand_tok):
        """One cached step with the commit rule inside the graph (no per-op launches for the rule). x: (L,) tokens;
        masked: (m,) block indices of the masked slots (sorted); sel: (kk,) slot indices of the probe candidates;
        cand_tok: (kk,) their revealed tokens (the previous step's proposals). Same decisions as the lean decoder.
        Returns new block tokens (bl,), slot proposals v (m,), slot margins c (m,), and masked slots left (python int)."""
        Lc, bl = self._key
        m = masked.numel(); kk = sel.numel()
        mp = max(self.mp_step, -(-m // self.mp_step) * self.mp_step)
        key = (Lc, bl, mp)
        if key not in self.graphs:
            self.graphs[key] = self._build(Lc, bl, mp, self._ctx[0][0])
        st = self.graphs[key]
        blk = x[s:e]
        pos = masked + s
        with torch.autocast(x.device.type, enabled=False):
            sin, cos = self.rot.get_rotary_embedding(e, x.device)
        st['sin_b'].copy_(sin[:, :, s:e]); st['cos_b'].copy_(cos[:, :, s:e])
        st['toks'][:bl].copy_(blk)
        idx = torch.cat([pos, pos[-1:].expand(mp - m)])
        st['sin_p'].copy_(sin.index_select(2, idx)); st['cos_p'].copy_(cos.index_select(2, idx))
        rows = blk[masked].unsqueeze(0).repeat(self.K, 1)
        rows[torch.arange(kk, device=x.device), sel] = cand_tok
        rows = torch.cat([rows, rows[:, -1:].expand(-1, mp - m)], 1)
        st['toks'][bl:].copy_(rows.reshape(-1))
        comm = torch.ones(bl, dtype=torch.bool, device=x.device); comm[masked] = False
        st['committed'].copy_(comm.to(torch.int32)); pv = torch.zeros(mp, dtype=torch.int32, device=x.device); pv[:m] = 1
        st['pvalid'].copy_(pv)
        sp = torch.full((mp,), bl, dtype=torch.long, device=x.device); sp[:m] = masked; st['slot_pos'].copy_(sp)
        sv = torch.zeros(mp, dtype=torch.bool, device=x.device); sv[:m] = True; st['slot_valid'].copy_(sv)
        ca = torch.zeros(self.K, dtype=torch.long, device=x.device); ca[:kk] = sel; st['cand'].copy_(ca)
        ct = torch.zeros(self.K, dtype=torch.long, device=x.device); ct[:kk] = cand_tok; st['cand_tok'].copy_(ct)
        cv = torch.zeros(self.K, dtype=torch.bool, device=x.device); cv[:kk] = True; st['cand_valid'].copy_(cv)
        st['graph'].replay()
        return st['new_tok'], st['v_out'][:m], st['c_out'][:m], st['out'][:bl]

    @torch.no_grad()
    def _step_windowed(self, x, s, e, probe_rows, ppos):
        """Short probe rows: row r recomputes only the masked positions within +-window of its candidate (padded to a
        fixed width W = min(2*window+1, block)); every other block position, committed or masked, comes from this
        step's main-row keys (as fused_forward_windowed). probe_rows: (n, m) full rows as for step(); each row's
        candidate is the position where it differs from the block. Returns block logits (1, bl, V) and probe logits
        (n, m, V) in the full layout, with the main logits outside each window (Delta = 0 there)."""
        Lc, bl = self._key
        m = ppos.numel(); nP = probe_rows.shape[0]
        W = min(2 * self.window + 1, bl)
        key = (Lc, bl, W)
        if key not in self.graphs:
            self.graphs[key] = self._build(Lc, bl, W, self._ctx[0][0])
        st = self.graphs[key]
        with torch.autocast(x.device.type, enabled=False):
            sin, cos = self.rot.get_rotary_embedding(e, x.device)
        st['sin_b'].copy_(sin[:, :, s:e]); st['cos_b'].copy_(cos[:, :, s:e])
        st['toks'][:bl].copy_(x[s:e])
        cand = (probe_rows != x[ppos].unsqueeze(0)).float().argmax(1)  # index in ppos of each row's candidate
        Pj = ppos[cand]
        wpos, wvalid, widx = probe_windows(ppos, Pj, self.window)       # (nP, W') with W' <= W
        Wp = wpos.shape[1]
        if Wp < W:  # pad to the graph width with the row's last position (masked out)
            wpos = torch.cat([wpos, wpos[:, -1:].expand(-1, W - Wp)], 1)
            widx = torch.cat([widx, widx[:, -1:].expand(-1, W - Wp)], 1)
            wvalid = torch.cat([wvalid, torch.zeros(nP, W - Wp, dtype=torch.bool, device=x.device)], 1)
        rows = torch.gather(probe_rows, 1, widx)                      # tokens at each row's window
        if nP < self.K:  # pad rows (outputs dropped)
            pad = self.K - nP
            rows = torch.cat([rows, rows[:1].expand(pad, -1)]); wpos = torch.cat([wpos, wpos[:1].expand(pad, -1)])
            wvalid = torch.cat([wvalid, torch.zeros(pad, W, dtype=torch.bool, device=x.device)])
        st['toks'][bl:].copy_(rows.reshape(-1))
        st['sin_p'].copy_(sin[0, 0][wpos].unsqueeze(1)); st['cos_p'].copy_(cos[0, 0][wpos].unsqueeze(1))
        # block slots read from the main row: committed ones, and masked ones outside the row's window
        own = torch.zeros(self.K, bl, dtype=torch.bool, device=x.device)
        own.scatter_(1, (wpos - s).clamp(0, bl - 1), wvalid)
        st['committed'].copy_((~own).to(torch.int32))
        st['pvalid'].copy_(wvalid.to(torch.int32))
        st['graph'].replay()
        out = st['out']
        blk_logits = out[:bl].unsqueeze(0).clone()
        pl_w = out[bl:].view(self.K, W, -1)[:nP]
        fill = blk_logits[0][ppos - s].unsqueeze(0).expand(nP, -1, -1)
        pl = scatter_window_logits(pl_w, widx[:nP], wvalid[:nP], m, fill)
        return blk_logits, pl


def probe_windows(pos, Pj, w):
    """Windowed probes: for each probed candidate Pj[r], the still-masked positions within +-w of it.
    pos: (m,) still-masked positions (sorted); returns wpos (K, W) padded with each row's last real position,
    valid (K, W) bool (False on padding), and idx (K, W) the index of each wpos entry in pos."""
    near = (pos.unsqueeze(0) - Pj.unsqueeze(1)).abs() <= w  # (K, m)
    W = int(near.sum(1).max())
    order = torch.argsort((~near).to(torch.int8), dim=1, stable=True)[:, :W]  # windowed positions first, in order
    valid = torch.gather(near, 1, order)
    last = order[torch.arange(order.shape[0]), valid.sum(1) - 1].unsqueeze(1)
    idx = torch.where(valid, order, last)
    return pos[idx], valid, idx


@torch.no_grad()
def fused_forward_windowed(model, x, probe_rows_w, wpos, wvalid):
    """fused_forward with windowed probe rows (mf_select.probe_windows): probe row r recomputes only the positions
    wpos[r] (padded entries marked False in wvalid), with its candidate revealed; every other position uses the main
    keys/values of the same layer. probe_rows_w: (K, W) tokens at wpos. Same main logits as fused_forward.
    Returns main logits (1, L, V) and probe logits (K, W, V) (entries on padding are meaningless)."""
    M = model.model
    T, cfg = M.transformer, M.config
    L, (K, W) = x.numel(), probe_rows_w.shape
    d, nh = cfg.d_model, cfg.n_heads
    hd = d // nh
    h = T.wte(torch.cat([x, probe_rows_w.reshape(-1)]))
    if cfg.input_emb_norm:
        h = h * (d ** 0.5)
    rot = T.blocks[0].rotary_emb
    with torch.autocast(x.device.type, enabled=False):
        sin, cos = rot.get_rotary_embedding(L, x.device)
    sin_p = sin[0, 0][wpos].unsqueeze(1); cos_p = cos[0, 0][wpos].unsqueeze(1)  # (K, 1, W, hd)
    scale = 1.0 / math.sqrt(hd)
    full = torch.float32 if cfg.rope_full_precision else None
    excl = torch.zeros(K, L, dtype=torch.bool, device=x.device)
    excl.scatter_(1, wpos, True)  # each row's recomputed positions come from its own keys
    excl = excl.view(K, 1, 1, L)
    own_bias = torch.where(wvalid, 0.0, float('-inf')).view(K, 1, 1, W)  # padding keys never attended

    def rope(t_in, sn, cs):
        t_ = t_in.to(full) if full else t_in
        return rot.apply_rotary_pos_emb(sn.type_as(t_), cs.type_as(t_), t_).type_as(t_in)

    for blk in T.blocks:
        xn = blk.attn_norm(h)
        q, k, v = blk.q_proj(xn), blk.k_proj(xn), blk.v_proj(xn)
        qm = q[:L].view(1, L, nh, hd).transpose(1, 2); km = k[:L].view(1, L, nh, hd).transpose(1, 2)
        vm = v[:L].view(1, L, nh, hd).transpose(1, 2)
        qp = q[L:].view(K, W, nh, hd).transpose(1, 2); kp = k[L:].view(K, W, nh, hd).transpose(1, 2)
        vp = v[L:].view(K, W, nh, hd).transpose(1, 2)
        with torch.autocast(x.device.type, enabled=False):
            qm, km = rope(qm, sin, cos), rope(km, sin, cos)
            qp, kp = rope(qp, sin_p, cos_p), rope(kp, sin_p, cos_p)
        att_m = F.scaled_dot_product_attention(qm, km, vm, is_causal=False)
        qflat = qp.transpose(0, 1).reshape(1, nh, K * W, hd)
        sc_sh = (torch.matmul(qflat, km.transpose(-1, -2)).float() * scale).view(nh, K, W, L).transpose(0, 1)
        sc_sh = sc_sh.masked_fill(excl, float('-inf'))
        sc_own = torch.matmul(qp, kp.transpose(-1, -2)).float() * scale + own_bias  # (K, nh, W, W)
        p_ = torch.softmax(torch.cat([sc_sh, sc_own], -1), -1).to(vm.dtype)
        p_sh = p_[..., :L].transpose(0, 1).reshape(1, nh, K * W, L)
        att_p = torch.matmul(p_sh, vm).view(nh, K, W, hd).transpose(0, 1) + torch.matmul(p_[..., L:], vp)
        att = torch.cat([att_m.transpose(1, 2).reshape(L, d), att_p.transpose(1, 2).reshape(K * W, d)])
        h = h + blk.dropout(blk.attn_out(att))
        xf = blk.ff_norm(h)
        h = h + blk.dropout(blk.ff_out(blk.act(blk.ff_proj(xf)) * blk.up_proj(xf)))
    h = T.ln_f(h)
    logits = F.linear(h, T.wte.weight, None) if cfg.weight_tying else T.ff_out(h)
    if cfg.scale_logits:
        logits.mul_(1 / math.sqrt(d))
    return logits[:L].unsqueeze(0), logits[L:].view(K, W, -1)


def scatter_window_logits(pl_w, widx, wvalid, m, fill):
    """Probe logits (K, W, V) of windowed rows back to the full (K, m, V) layout of masked positions; entries
    outside a row's window get `fill` (K, m, V) values (the main logits there, so Delta = 0)."""
    out = fill.clone()
    K, W, V = pl_w.shape
    r = torch.arange(K, device=pl_w.device).unsqueeze(1).expand(K, W)
    out[r[wvalid], widx[wvalid]] = pl_w[wvalid]
    return out


def graph_commit_rule(blk_logits, probe_logits, blk_tok, slot_pos, slot_valid, cand, cand_tok, cand_valid, tau, R, mask_id):
    """Measured-coupling commit rule on static, padded tensors (so it can live inside a CUDA graph; no host syncs).
    blk_logits (bl, V) main-row logits of the block; probe_logits (K, mp, V) at the padded masked slots;
    blk_tok (bl,) current block tokens; slot_pos (mp,) block index of each masked slot (padding: index bl, a scratch slot);
    slot_valid (mp,) bool real slot; cand (K,) slot index of each probe's candidate; cand_tok (K,) its revealed token;
    cand_valid (K,) bool real probe row. Same math as generate_mf_dual_cache_lean: margins from logits (exact),
    stale probes dropped, Delta on probed columns, R mean-field updates from sigmoid(c), commit q >= tau and argmax.
    Returns new block tokens (bl,), slot proposals v (mp,), slot margins c (mp,) (-inf on padding), and the number
    of masked slots left after the commit (0-dim long tensor)."""
    bl = blk_tok.shape[0]
    sp = slot_pos.clamp_max(bl - 1)                           # padding reads a real row; its results are masked out
    lgm = blk_logits[sp].float()                               # (mp, V)
    v = lgm.argmax(-1)                                         # same tie-break as the lean rule (first max)
    top2 = lgm.topk(2, -1)
    c = top2.values[:, 0] - top2.values[:, 1]                  # margin of the argmax token = top1 - top2 (0 on a tie)
    c = torch.where(slot_valid, c, torch.full_like(c, float('-inf')))
    pl = probe_logits.float()                                  # (K, mp, V)
    l_v = pl.gather(-1, v.view(1, -1, 1).expand(pl.shape[0], -1, 1)).squeeze(-1)  # (K, mp) logit of v_i
    pt2 = pl.topk(2, -1)
    other = torch.where(pt2.indices[..., 0] == v.unsqueeze(0), pt2.values[..., 1], pt2.values[..., 0])
    mP = l_v - other                                           # (K, mp) margin of v_i with candidate j revealed
    keep = cand_valid & (cand_tok == v[cand])                  # stale probes dropped
    mp = c.shape[0]
    cz = torch.where(slot_valid, c, torch.zeros_like(c))
    Dcol = torch.where(keep.unsqueeze(1) & slot_valid.unsqueeze(0), cz.unsqueeze(0) - mP, torch.zeros_like(mP))  # (K, mp): D[i, cand_k] = c_i - mP[k, i]
    D = torch.zeros(mp, mp, device=c.device, dtype=c.dtype)
    D.index_add_(1, cand, Dcol.T)                              # columns at the candidates' slots
    D = D * (1 - torch.eye(mp, device=c.device, dtype=c.dtype))
    D = D * slot_valid.unsqueeze(0) * slot_valid.unsqueeze(1)
    q = torch.sigmoid(cz)
    for _ in range(R):
        q = torch.sigmoid(cz - D @ q)
    q = torch.where(slot_valid, q, torch.full_like(q, -1.0))
    commit = (q >= tau) & slot_valid
    commit = commit | (torch.arange(mp, device=c.device) == q.argmax())  # always commit the most confident
    commit = commit & slot_valid
    ext = torch.cat([blk_tok, blk_tok[:1]])                  # one scratch slot at index bl for padded writes
    ext.scatter_(0, slot_pos, torch.where(commit, v, ext[slot_pos]))
    new_tok = ext[:bl]
    left = (new_tok == mask_id).sum()
    return new_tok, v, c, left


class ProbeGraphRunner:
    """Fast probes replayed from CUDA graphs, to remove the per-call launch overhead.
    One graph per (padded sequence length, block length), with K probe rows (unused rows are padding).
    The shared cache is rotated straight into static buffers; padding keys and block keys are excluded
    through a mask, so results equal the eager probe path up to floating-point noise."""

    def __init__(self, model, K, bucket=128):
        self.model, self.K, self.bucket = model, K, bucket
        self.rot = next(m for m in model.modules() if type(m).__name__ == 'RotaryEmbedding')
        self.graphs = {}

    def _build(self, Lb, bl, like):
        dev = like[0][0].device
        st = {'xb': torch.zeros(self.K, bl, dtype=torch.long, device=dev),
              'pos_idx': torch.zeros(bl, dtype=torch.long, device=dev),
              'excl': torch.ones(Lb, dtype=torch.bool, device=dev),
              'kv': [tuple(torch.zeros(1, t.shape[1], Lb, t.shape[3], dtype=t.dtype, device=dev) for t in layer)
                     for layer in like]}

        def fwd():
            out = self.model(st['xb'], past_key_values=st['kv'], use_cache=False,
                             replace_position=('probe_t', st['pos_idx'], st['excl']))
            return (out[0] if isinstance(out, tuple) else out).logits

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm-up: allocator, rotary/attention-bias caches
                fwd()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            st['logits'] = fwd()
        st['graph'] = g
        return st

    @torch.no_grad()
    def run(self, x, pos, tok, P, block, kv_cache):
        s, e = block
        L, bl = x.numel(), e - s
        Lb = -(-L // self.bucket) * self.bucket
        key = (Lb, bl)
        if key not in self.graphs:
            self.graphs[key] = self._build(Lb, bl, kv_cache)
        st = self.graphs[key]
        sin, cos = self.rot.get_rotary_embedding(L, x.device)
        for (k, v), (k_st, v_st) in zip(kv_cache, st['kv']):
            with torch.autocast(k.device.type, enabled=False):
                k_ = k.float() if self.rot.config.rope_full_precision else k
                k_st[:, :, :L].copy_(self.rot.apply_rotary_pos_emb(sin.type_as(k_), cos.type_as(k_), k_))
            v_st[:, :, :L].copy_(v)
        st['excl'].fill_(False)
        st['excl'][s:e] = True
        st['excl'][L:] = True
        st['pos_idx'].copy_(torch.arange(s, e, device=x.device))
        n, rel = P.numel(), pos - s
        rows = x[s:e].unsqueeze(0).repeat(self.K, 1)
        rows[torch.arange(n, device=x.device), rel[P]] = tok[P]
        st['xb'].copy_(rows)
        st['graph'].replay()
        logp = F.log_softmax(st['logits'][:n, rel].float(), -1)  # (n, m, V)
        return log_margin(logp, tok.unsqueeze(0).expand(n, -1))


_GRAPH_RUNNERS = {}


@torch.no_grad()
def _probe_columns(model, x, pos, tok, P, block, kv_cache, use_graph=False, K=8):
    if use_graph:
        runner = _GRAPH_RUNNERS.get((id(model), K))
        if runner is None:
            runner = _GRAPH_RUNNERS[(id(model), K)] = ProbeGraphRunner(model, K)
        return runner.run(x, pos, tok, P, block, kv_cache)
    return _probe_columns_eager(model, x, pos, tok, P, block, kv_cache)


@torch.no_grad()
def _probe_columns_eager(model, x, pos, tok, P, block, kv_cache):
    """Margins of all candidates `pos` after revealing each probed candidate pos[P] (one fast probe
    row per member of P). Returns (|P|, m)."""
    s, e = block
    kv_rot = rotate_kv_cache(model, kv_cache)
    rel = pos - s
    xb = x[s:e].unsqueeze(0).repeat(P.numel(), 1)
    xb[torch.arange(P.numel(), device=x.device), rel[P]] = tok[P]
    out = model(xb, past_key_values=kv_rot, use_cache=False, replace_position=('probe', s, e))
    out = out[0] if isinstance(out, tuple) else out
    logp = F.log_softmax(out.logits[:, rel].float(), -1)  # (|P|, m, V)
    return log_margin(logp, tok.unsqueeze(0).expand(P.numel(), -1))


@torch.no_grad()
def oracle_coupling_cached(model, x, pos, tok, block, kv_cache, chunk=32):
    """Block-local approximation of oracle_coupling (cheap probes).
    Each probe recomputes only the current block x[s:e] (with x_j := tok_j revealed); every other
    position reuses the keys/values of the main forward pass (where j was still masked), as in
    Fast-dLLM's dual cache. A probe costs ~|block| tokens instead of the whole sequence.
    kv_cache: past_key_values of the main forward on x (not modified: each probe batch gets a copy,
    because the model writes block keys/values into the cache in place)."""
    s, e = block
    m = pos.numel()
    margin = torch.empty(m, m, device=x.device)
    probs_cond = []
    rel = pos - s
    for c0 in range(0, m, chunk):
        js = torch.arange(c0, min(c0 + chunk, m), device=x.device)
        n = js.numel()
        xb = x[s:e].unsqueeze(0).repeat(n, 1)
        xb[torch.arange(n, device=x.device), rel[js]] = tok[js]
        cache = [tuple(t.expand(n, *t.shape[1:]).clone() for t in layer) for layer in kv_cache]
        out = model(xb, past_key_values=cache, use_cache=True, replace_position=(s, e))  # slice fast path
        out = out[0] if isinstance(out, tuple) else out
        logp = F.log_softmax(out.logits[:, rel].float(), -1)  # (n, m, V)
        margin[js] = log_margin(logp, tok.unsqueeze(0).expand(n, m))
        probs_cond.append(logp.exp())
        del cache
    return margin, torch.cat(probs_cond, 0)


def mean_field_commit(probs: torch.Tensor, tau: float = 0.85, R: int = 2, use_pairwise: bool = True,
                      uniform_D: bool = False, damping: float = 1.0, D: torch.Tensor = None):
    """probs: (m, V) predictive distributions of the active masked positions.
    D: optional precomputed coupling with D[i, j] = penalty on i from committing j (diag ignored).
    Returns (commit mask (m,) bool, q (m,), c (m,), D (m, m))."""
    m = probs.shape[0]
    top2 = probs.topk(2, dim=-1).values
    c = torch.log(top2[:, 0].clamp_min(1e-12)) - torch.log(top2[:, 1].clamp_min(1e-12))
    if D is not None:
        D = D.to(probs.dtype).clone()
        D.fill_diagonal_(0.0)
    elif m == 1 or not use_pairwise:
        D = torch.zeros(m, m, device=probs.device, dtype=probs.dtype)
    elif uniform_D:
        D = torch.ones(m, m, device=probs.device, dtype=probs.dtype)
        D.fill_diagonal_(0.0)
    else:
        D = 1.0 - pairwise_jsd(probs) / math.log(2.0)
        D.fill_diagonal_(0.0)
        D = D / D.max().clamp_min(1e-12)
    q = torch.sigmoid(c)
    for _ in range(R):
        q = (1 - damping) * q + damping * torch.sigmoid(c - D @ q)
    commit = q >= tau
    commit[q.argmax()] = True
    return commit, q, c, D


def coupling_from_oracle(kind, c, probs, margin, probs_cond):
    """Turn oracle measurements into D[i, j] (penalty on i from committing j).
    kind: 'oracle_margin'  D_ij = max(0, c_i - c_i^(j))  (drop in i's log-margin; lambda = 1, no tuning)
          'oracle_signed'  D_ij = c_i - c_i^(j)          (negative = j supports i)
          'oracle_tv'      D_ij = TV(pi_i, pi_i | x_j)    (DEMASK's coupling, true not learned)"""
    if kind == 'oracle_margin':
        return (c.unsqueeze(1) - margin.T).clamp_min(0.0)
    if kind == 'oracle_signed':
        return c.unsqueeze(1) - margin.T
    if kind == 'oracle_tv':
        return 0.5 * (probs_cond - probs.unsqueeze(0)).abs().sum(-1).T
    raise ValueError(kind)


PROBE_STATS = {'probes': 0, 'stale': 0, 'pairs': 0, 'measured_pairs': 0}  # per run, reset by the caller
_CTRL_GEN = None


def apply_control(D, measured, control):
    """Controls for the measured coupling (review step 2). Only the measured columns are touched, so every
    control has the same coverage as the deployed method.
    'shuffle'     permute the measured off-diagonal entries at random (same values, random pairs)
    'support'     keep support only, D <- min(D, 0)
    'conflict'    keep conflict only, D <- max(D, 0)
    'sym'         (D + D^T) / 2 on the measured block
    'bonus:<b>'   D_ij = -b on every measured pair (a constant support bonus)
    'csup:<b>'    measured conflicts kept (D > 0), every measured support (D < 0) replaced by -b: isolates whether
                  pair-specific support magnitudes matter beyond accurate conflicts plus a generic support"""
    global _CTRL_GEN
    m = D.shape[0]
    mask = measured.unsqueeze(0).expand(m, m) & ~torch.eye(m, dtype=torch.bool, device=D.device)
    if not mask.any():
        return D
    D = D.clone()
    if control == 'shuffle':
        if _CTRL_GEN is None:
            _CTRL_GEN = torch.Generator(device='cpu').manual_seed(0)
        vals = D[mask]
        D[mask] = vals[torch.randperm(vals.numel(), generator=_CTRL_GEN).to(D.device)]
    elif control == 'support':
        D[mask] = D[mask].clamp_max(0.0)
    elif control == 'conflict':
        D[mask] = D[mask].clamp_min(0.0)
    elif control == 'sym':
        both = mask & mask.T
        S = (D + D.T) / 2
        D[both] = S[both]
    elif control.startswith('bonus:'):
        D[mask] = -float(control.split(':', 1)[1])
    elif control.startswith('csup:'):
        sup = mask & (D < 0)
        D[sup] = -float(control.split(':', 1)[1])
    else:
        raise ValueError(control)
    return D


def get_transfer_index_mf(logits, mask_index, x, tau=0.85, R=2, use_pairwise=True, uniform_D=False,
                          damping=1.0, coupling='jsd', model=None, estimator=None, trace=None,
                          flip_rho=0.05, flip_eps=None, kv_cache=None, block=None, probe_k=8, probe_graph=False, probe_info=None,
                          plugin=None, plugin_kw=None, control=None):
    """Drop-in replacement for DAWN's get_transfer_index (greedy, temperature 0).
    mask_index must already be restricted to the current block.
    coupling: 'jsd' (the paper), one of the oracle kinds (needs model; costs extra forward rows),
              or 'lagged' (needs estimator: lagged.LaggedCoupling; no extra forward passes),
              or 'flip' (needs estimator: lagged.FlipCoupling; commit iff predicted flip prob <= flip_rho;
              plain threshold tau until the estimator has enough observations)."""
    x0 = torch.argmax(logits, dim=-1)
    x0 = torch.where(mask_index, x0, x)
    transfer_index = torch.zeros_like(mask_index)
    for b in range(x.shape[0]):
        pos = mask_index[b].nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            continue
        probs = F.softmax(logits[b, pos].float(), dim=-1)
        if coupling == 'flip':
            from lagged import pair_features, unary_features, flip_mean_field_commit
            assert x.shape[0] == 1, 'flip coupling keeps per-sequence state; use batch size 1'
            estimator.observe(logits[b], pos)
            top2 = probs.topk(2, dim=-1).values
            c0 = torch.log(top2[:, 0].clamp_min(1e-12)) - torch.log(top2[:, 1].clamp_min(1e-12))
            phi = pair_features(pos, x0[b, pos], probs)
            u = unary_features(c0, probs)
            if estimator.ready:
                a, W = estimator.logit_terms(u, phi)
                commit, q, _ = flip_mean_field_commit(a, W, rho=flip_rho, R=R, damping=damping, eps=flip_eps)
            else:  # warm-up: the plain threshold rule
                q = torch.sigmoid(c0)
                commit = q >= tau
                commit[q.argmax()] = True
            transfer_index[b, pos[commit]] = True
            estimator.record(pos, x0[b, pos], c0, commit, phi, u)
            if trace is not None:
                trace.append({'pos': pos.tolist(), 'c': c0.tolist(), 'q': q.tolist(),
                              'commit': commit.tolist(), 'tokens': x0[b, pos].tolist()})
            continue
        D = phi = None
        if plugin is not None and coupling == 'jsd':  # plug-in without probes: D = 0, the base rule
            p_top = probs.max(-1).values
            import plugins
            rule = {'threshold': plugins.threshold_select, 'dawn': plugins.dawn_select}[plugin]
            transfer_index[b, pos[rule(p_top, torch.zeros(pos.numel(), pos.numel(), device=probs.device), **(plugin_kw or {}))]] = True
            continue
        if coupling == 'lagged':
            from lagged import pair_features
            assert x.shape[0] == 1, 'lagged coupling keeps per-sequence state; use batch size 1'
            estimator.observe(logits[b], pos)
            phi = pair_features(pos, x0[b, pos], probs)
            D = estimator.coupling(phi)
        elif coupling != 'jsd' and pos.numel() > 1:
            top2 = probs.topk(2, dim=-1).values
            c0 = torch.log(top2[:, 0].clamp_min(1e-12)) - torch.log(top2[:, 1].clamp_min(1e-12))
            if coupling.startswith('cached_'):  # cheap block-local probes, same formulas as the oracle
                margin, probs_cond = oracle_coupling_cached(model, x[b], pos, x0[b, pos], block, kv_cache)
                kind = 'oracle_' + coupling[len('cached_'):]
            elif coupling.startswith(('fused_', 'fusedm_')):
                # Probes computed inside the main pass (fused_forward). A probe whose revealed token is
                # no longer the current proposal is dropped (Delta = 0 for that column).
                Pj, tok_rev, probe_logits = probe_info
                m = pos.numel()
                jidx = (pos.unsqueeze(0) == Pj.unsqueeze(1)).float().argmax(1)  # index of each Pj in pos
                valid = tok_rev == x0[b, Pj]
                cols = slice(None) if coupling.startswith('fusedm_') else (pos - block[0])  # fusedm rows hold only pos
                lpP = F.log_softmax(probe_logits[:, cols].float(), -1)  # (|P|, m, V)
                mP = log_margin(lpP, x0[b, pos].unsqueeze(0).expand(Pj.numel(), -1))
                margin = c0.unsqueeze(0).repeat(m, 1)
                margin[jidx[valid]] = mP[valid]
                probs_cond = None
                kind = 'oracle_' + coupling.split('_', 1)[1]
                measured = torch.zeros(m, dtype=torch.bool, device=probs.device)
                measured[jidx[valid]] = True  # columns j of Delta that were measured this step
                PROBE_STATS['probes'] += Pj.numel(); PROBE_STATS['stale'] += int((~valid).sum())
                PROBE_STATS['pairs'] += m * (m - 1); PROBE_STATS['measured_pairs'] += int(measured.sum()) * (m - 1)
            elif coupling.startswith('probe_'):
                # Fast probes with a budget: probe only the probe_k most confident candidates (their
                # q_j dominate sum_j Delta_ij q_j); unprobed columns get Delta = 0.
                m = pos.numel()
                P = torch.topk(c0, min(probe_k, m)).indices
                mP = _probe_columns(model, x[b], pos, x0[b, pos], P, block, kv_cache, use_graph=probe_graph, K=probe_k)
                margin = c0.unsqueeze(0).repeat(m, 1)  # rows j not probed: no change (Delta = 0)
                margin[P] = mP
                probs_cond = None
                kind = 'oracle_' + coupling[len('probe_'):]
            else:
                margin, probs_cond = oracle_coupling(model, x[b], pos, x0[b, pos])
                kind = coupling
            D = coupling_from_oracle(kind, c0, probs, margin, probs_cond)
            if control is not None and coupling.startswith(('fused_', 'fusedm_')):
                D = apply_control(D, measured, control)
        if plugin is not None:
            # Measured coupling plugged into another decoder's rule (plugins.py); D = 0 gives the base rule.
            import plugins
            p_top = probs.max(-1).values
            if D is None:
                D = torch.zeros(pos.numel(), pos.numel(), device=probs.device)
            rule = {'threshold': plugins.threshold_select, 'dawn': plugins.dawn_select}[plugin]
            commit = rule(p_top, D.float(), **(plugin_kw or {}))
            transfer_index[b, pos[commit]] = True
            continue
        commit, q, c, D = mean_field_commit(probs, tau=tau, R=R, use_pairwise=use_pairwise,
                                            uniform_D=uniform_D, damping=damping, D=D)
        transfer_index[b, pos[commit]] = True
        if coupling == 'lagged':
            estimator.record(pos, x0[b, pos], c, commit, phi)
        if trace is not None:
            trace.append({'pos': pos.tolist(), 'c': c.tolist(), 'q': q.tolist(),
                          'commit': commit.tolist(), 'tokens': x0[b, pos].tolist()})
    return x0, transfer_index
