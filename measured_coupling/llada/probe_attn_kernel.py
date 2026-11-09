"""Fused attention for one cached decoding step with probe rows (Triton), exact.

Every query of a step (the main block row and the K probe rows) attends to
  (a) the SHARED context: the cached keys/values outside the block, the same for every row, and
  (b) its row's OWN in-block keys/values: for the main row, its fresh block keys; for a probe row, this step's keys
      of the committed block positions plus its own keys at the still-masked positions (padding masked out).
mf_select.cached_step computes this as two attention calls and a log-sum-exp merge in float32. This kernel does it in
one pass per query tile with an online softmax over (a) then (b), the standard flash-attention recurrence, so it is
the same softmax up to floating-point rounding (no approximation).

Layout: q (R, nh, n, hd) with R rows (row 0 = main block row when present), shared k/v (nh, Lc, hd) with an additive
key bias (Lc,) (0 or -inf, for padded context), own k/v (R, nh, no, hd) with an additive bias (R, no).
Returns out (R, nh, n, hd) in q's dtype."""
import torch
import triton
import triton.language as tl


@triton.jit
def _probe_attn_fwd(Q, KS, VS, BS, KO, VO, BO, OUT, scale,
                    n, Lc, no,
                    sq_r, sq_h, sq_n, sks_h, sks_l, sko_r, sko_h, sko_l, sbo_r, so_r, so_h, so_n,
                    HD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_O: tl.constexpr):
    pid_m = tl.program_id(0)  # query tile within the row
    r = tl.program_id(1)  # row (0 = main block row when present)
    h = tl.program_id(2)  # head
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + r * sq_r + h * sq_h + offs_m[:, None] * sq_n + offs_d[None, :], mask=offs_m[:, None] < n, other=0.0)
    m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HD], tl.float32)
    qs = (q * scale).to(q.dtype)
    # (a) shared context, the same keys for every row
    for start in range(0, Lc, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        kmask = offs_n < Lc
        k = tl.load(KS + h * sks_h + offs_n[:, None] * sks_l + offs_d[None, :], mask=kmask[:, None], other=0.0)
        v = tl.load(VS + h * sks_h + offs_n[:, None] * sks_l + offs_d[None, :], mask=kmask[:, None], other=0.0)
        b = tl.load(BS + offs_n, mask=kmask, other=float('-inf'))
        s = tl.dot(qs, tl.trans(k)) + b[None, :]
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    # (b) this row's own in-block keys
    offs_o = tl.arange(0, BLOCK_O)
    omask = offs_o < no
    k = tl.load(KO + r * sko_r + h * sko_h + offs_o[:, None] * sko_l + offs_d[None, :], mask=omask[:, None], other=0.0)
    v = tl.load(VO + r * sko_r + h * sko_h + offs_o[:, None] * sko_l + offs_d[None, :], mask=omask[:, None], other=0.0)
    b = tl.load(BO + r * sbo_r + offs_o, mask=omask, other=float('-inf'))
    s = tl.dot(qs, tl.trans(k)) + b[None, :]
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    acc = acc / l_i[:, None]
    tl.store(OUT + r * so_r + h * so_h + offs_m[:, None] * so_n + offs_d[None, :], acc.to(OUT.dtype.element_ty), mask=offs_m[:, None] < n)


def probe_attention(q, k_sh, v_sh, b_sh, k_own, v_own, b_own, scale):
    """q (R, nh, n, hd); k_sh/v_sh (nh, Lc, hd) contiguous in the last dim; b_sh (Lc,) float32;
    k_own/v_own (R, nh, no, hd); b_own (R, no) float32. Returns (R, nh, n, hd)."""
    R, nh, n, hd = q.shape
    Lc, no = k_sh.shape[1], k_own.shape[2]
    out = torch.empty_like(q)
    BLOCK_M = 16 if n <= 16 else 32 if n <= 32 else 64
    BLOCK_N = 64
    BLOCK_O = max(16, triton.next_power_of_2(no))
    grid = (triton.cdiv(n, BLOCK_M), R, nh)
    _probe_attn_fwd[grid](q, k_sh, v_sh, b_sh, k_own, v_own, b_own, out, scale, n, Lc, no,
                          q.stride(0), q.stride(1), q.stride(2), k_sh.stride(0), k_sh.stride(1),
                          k_own.stride(0), k_own.stride(1), k_own.stride(2), b_own.stride(0),
                          out.stride(0), out.stride(1), out.stride(2),
                          HD=hd, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_O=BLOCK_O, num_warps=4, num_stages=2)
    return out


def reference(q, k_sh, v_sh, b_sh, k_own, v_own, b_own, scale):
    """Same attention in float32 PyTorch: softmax over [shared context ; own keys] with the biases."""
    R = q.shape[0]
    ks = torch.cat([k_sh.unsqueeze(0).expand(R, -1, -1, -1), k_own], 2).float()
    vs = torch.cat([v_sh.unsqueeze(0).expand(R, -1, -1, -1), v_own], 2).float()
    b = torch.cat([b_sh.view(1, 1, 1, -1).expand(R, 1, 1, -1), b_own.view(R, 1, 1, -1)], -1)
    s = q.float() @ ks.transpose(-1, -2) * scale + b
    return (torch.softmax(s, -1) @ vs).to(q.dtype)


@triton.jit
def _probe_attn3_fwd(QB, QP, KS, VS, BS, KB, VB, KP, VP, CM, PM, OB, OP, scale,
                     bl, mp, Lc,
                     sqb_h, sqb_n, sqp_r, sqp_h, sqp_n, sks_h, sks_l, skb_h, skb_l, skp_r, skp_h, skp_l, scm_r, spm_r,
                     HD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_P: tl.constexpr):
    """Program (tile, row, head). Row 0 is the main block row (bl queries); rows 1..K are probe rows (mp queries).
    Keys, in one online softmax: (a) shared context KS/VS with bias BS; (b) the block row's fresh block keys KB/VB,
    all bl for the main row, and for probe rows only committed slots (CM = 1); (c) for probe rows only, the row's own
    keys KP/VP at its masked positions (PM = 1 for real, 0 for padding). Nothing is copied or concatenated."""
    pid_m = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    offs_d = tl.arange(0, HD)
    is_main = r == 0
    n = tl.where(is_main, bl, mp)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    qmask = offs_m < n
    if r == 0:
        q = tl.load(QB + h * sqb_h + offs_m[:, None] * sqb_n + offs_d[None, :], mask=qmask[:, None], other=0.0)
    else:
        q = tl.load(QP + (r - 1) * sqp_r + h * sqp_h + offs_m[:, None] * sqp_n + offs_d[None, :], mask=qmask[:, None], other=0.0)
    qs = (q * scale).to(q.dtype)
    m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HD], tl.float32)
    for start in range(0, Lc, BLOCK_N):  # (a) shared context
        offs_n = start + tl.arange(0, BLOCK_N)
        km = offs_n < Lc
        k = tl.load(KS + h * sks_h + offs_n[:, None] * sks_l + offs_d[None, :], mask=km[:, None], other=0.0)
        v = tl.load(VS + h * sks_h + offs_n[:, None] * sks_l + offs_d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(qs, tl.trans(k)) + tl.load(BS + offs_n, mask=km, other=float('-inf'))[None, :]
        m_new = tl.maximum(m_i, tl.max(s, 1)); alpha = tl.exp(m_i - m_new); p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v); m_i = m_new
    # (b) block keys of this step: all slots for the main row, committed slots for probe rows
    offs_b = tl.arange(0, BLOCK_B)
    bm = offs_b < bl
    k = tl.load(KB + h * skb_h + offs_b[:, None] * skb_l + offs_d[None, :], mask=bm[:, None], other=0.0)
    v = tl.load(VB + h * skb_h + offs_b[:, None] * skb_l + offs_d[None, :], mask=bm[:, None], other=0.0)
    cm = tl.load(CM + tl.maximum(r - 1, 0) * scm_r + offs_b, mask=bm, other=0)  # per probe row (row stride 0: shared)
    keep = bm & ((cm != 0) | is_main)
    s = tl.dot(qs, tl.trans(k)) + tl.where(keep, 0.0, float('-inf'))[None, :]
    m_new = tl.maximum(m_i, tl.max(s, 1)); alpha = tl.exp(m_i - m_new); p = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v); m_i = m_new
    if r > 0:  # (c) the probe row's own keys at its masked positions
        offs_p = tl.arange(0, BLOCK_P)
        pmk = offs_p < mp
        kq = tl.load(KP + (r - 1) * skp_r + h * skp_h + offs_p[:, None] * skp_l + offs_d[None, :], mask=pmk[:, None], other=0.0)
        vq = tl.load(VP + (r - 1) * skp_r + h * skp_h + offs_p[:, None] * skp_l + offs_d[None, :], mask=pmk[:, None], other=0.0)
        pm = tl.load(PM + (r - 1) * spm_r + offs_p, mask=pmk, other=0)
        sq = tl.dot(qs, tl.trans(kq)) + tl.where(pmk & (pm != 0), 0.0, float('-inf'))[None, :]
        mq = tl.maximum(m_i, tl.max(sq, 1)); aq = tl.exp(m_i - mq); pq = tl.exp(sq - mq[:, None])
        l_i = l_i * aq + tl.sum(pq, 1); acc = acc * aq[:, None] + tl.dot(pq.to(vq.dtype), vq); m_i = mq
    acc = acc / l_i[:, None]
    if r == 0:
        tl.store(OB + h * sqb_h + offs_m[:, None] * sqb_n + offs_d[None, :], acc.to(OB.dtype.element_ty), mask=qmask[:, None])
    else:
        tl.store(OP + (r - 1) * sqp_r + h * sqp_h + offs_m[:, None] * sqp_n + offs_d[None, :], acc.to(OP.dtype.element_ty), mask=qmask[:, None])


def probe_attention3(qb, qp, k_sh, v_sh, b_sh, kb, vb, kp, vp, committed, pvalid, scale):
    """One launch for the main block row and all probe rows, no key copies.
    qb (nh, bl, hd) main row queries; qp (K, nh, mp, hd) probe queries (or None); k_sh/v_sh (nh, Lc, hd); b_sh (Lc,)
    float32; kb/vb (nh, bl, hd) this step's block keys/values; kp/vp (K, nh, mp, hd) probe own keys/values;
    committed (bl,) or (K, bl) int32: block slots a probe row reads from the main row's keys (1 = visible);
    pvalid (mp,) or (K, mp) int32: 1 = real recomputed position of the row (0 = padding).
    Returns ob (nh, bl, hd) and op (K, nh, mp, hd) or None."""
    nh, bl, hd = qb.shape
    K = 0 if qp is None else qp.shape[0]
    mp = 0 if qp is None else qp.shape[2]
    Lc = k_sh.shape[1]
    ob = torch.empty_like(qb)
    op = torch.empty_like(qp) if K else qb  # dummy pointer when no probes
    BLOCK_M = 16 if max(bl, mp) <= 16 else 32 if max(bl, mp) <= 32 else 64
    grid = (triton.cdiv(max(bl, mp), BLOCK_M), 1 + K, nh)
    qp_, kp_, vp_ = (qp, kp, vp) if K else (qb.unsqueeze(0), kb.unsqueeze(0), vb.unsqueeze(0))
    pv = pvalid if K else committed
    _probe_attn3_fwd[grid](qb, qp_, k_sh, v_sh, b_sh, kb, vb, kp_, vp_, committed, pv, ob, op, scale, bl, max(mp, 1), Lc,
                           qb.stride(0), qb.stride(1), qp_.stride(0), qp_.stride(1), qp_.stride(2),
                           k_sh.stride(0), k_sh.stride(1), kb.stride(0), kb.stride(1),
                           kp_.stride(0), kp_.stride(1), kp_.stride(2),
                           committed.stride(0) if committed.dim() == 2 else 0, pv.stride(0) if pv.dim() == 2 else 0,
                           HD=hd, BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_B=max(16, triton.next_power_of_2(bl)),
                           BLOCK_P=max(16, triton.next_power_of_2(max(mp, 1))), num_warps=4, num_stages=2)
    return ob, (op if K else None)


@triton.jit
def _probe_attn4_fwd(Q, KK, VV, KS, VS, BS, POSB, POSP, COS, SIN, CM, PM, OUT, scale,
                     bl, mp, Lc, s_tok, sks_h, sks_l, s_cs,
                     HD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_P: tl.constexpr):
    """_probe_attn3_fwd with rotary applied on load and the output written in token-major layout. Q/KK/VV are the
    projections of every token of the step, (n_tot, nh*HD): the bl block tokens, then K probe rows of mp tokens.
    Queries and in-block keys are rotated here (float32, then rounded to the input dtype, as the model's rotary with
    rope_full_precision); context keys come rotated from the cache. POSB (bl,) and POSP (mp,) are rotary positions;
    COS/SIN (Lmax, HD) float32 tables. OUT (n_tot, nh*HD) is what attn_out reads, so nothing is copied after."""
    pid_m = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    HALF: tl.constexpr = HD // 2
    offs_d = tl.arange(0, HD)
    offs_dp = (offs_d + HALF) % HD                     # rotate_half partner
    sgn = tl.where(offs_d < HALF, -1.0, 1.0)
    is_main = r == 0
    n = tl.where(is_main, bl, mp)
    base = tl.where(is_main, 0, bl + (r - 1) * mp)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    qmask = offs_m < n
    if r == 0:
        pq = tl.load(POSB + offs_m, mask=qmask, other=0)
    else:
        pq = tl.load(POSP + offs_m, mask=qmask, other=0)
    qa = Q + (base + offs_m)[:, None] * s_tok + h * HD
    q = tl.load(qa + offs_d[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
    q2 = tl.load(qa + offs_dp[None, :], mask=qmask[:, None], other=0.0).to(tl.float32)
    q = (q * tl.load(COS + pq[:, None] * s_cs + offs_d[None, :]) + q2 * sgn[None, :] * tl.load(SIN + pq[:, None] * s_cs + offs_d[None, :])).to(Q.dtype.element_ty)
    qs = (q * scale).to(Q.dtype.element_ty)
    m_i = tl.full([BLOCK_M], float('-inf'), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HD], tl.float32)
    for start in range(0, Lc, BLOCK_N):  # (a) shared context (already rotated)
        offs_n = start + tl.arange(0, BLOCK_N)
        km = offs_n < Lc
        k = tl.load(KS + h * sks_h + offs_n[:, None] * sks_l + offs_d[None, :], mask=km[:, None], other=0.0)
        v = tl.load(VS + h * sks_h + offs_n[:, None] * sks_l + offs_d[None, :], mask=km[:, None], other=0.0)
        s = tl.dot(qs, tl.trans(k)) + tl.load(BS + offs_n, mask=km, other=float('-inf'))[None, :]
        m_new = tl.maximum(m_i, tl.max(s, 1)); alpha = tl.exp(m_i - m_new); p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v); m_i = m_new
    # (b) this step's block keys (rotated here): all slots for the main row, committed slots for probe rows
    offs_b = tl.arange(0, BLOCK_B)
    bm = offs_b < bl
    pb = tl.load(POSB + offs_b, mask=bm, other=0)
    ka = KK + offs_b[:, None] * s_tok + h * HD
    kx = tl.load(ka + offs_d[None, :], mask=bm[:, None], other=0.0).to(tl.float32)
    kx2 = tl.load(ka + offs_dp[None, :], mask=bm[:, None], other=0.0).to(tl.float32)
    k = (kx * tl.load(COS + pb[:, None] * s_cs + offs_d[None, :]) + kx2 * sgn[None, :] * tl.load(SIN + pb[:, None] * s_cs + offs_d[None, :])).to(KK.dtype.element_ty)
    v = tl.load(VV + offs_b[:, None] * s_tok + h * HD + offs_d[None, :], mask=bm[:, None], other=0.0)
    cm = tl.load(CM + offs_b, mask=bm, other=0)
    keep = bm & ((cm != 0) | is_main)
    s = tl.dot(qs, tl.trans(k)) + tl.where(keep, 0.0, float('-inf'))[None, :]
    m_new = tl.maximum(m_i, tl.max(s, 1)); alpha = tl.exp(m_i - m_new); p = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1); acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v); m_i = m_new
    if r > 0:  # (c) the probe row's own keys at its masked positions (rotated here)
        offs_p = tl.arange(0, BLOCK_P)
        pmk = offs_p < mp
        pp = tl.load(POSP + offs_p, mask=pmk, other=0)
        kpa = KK + (base + offs_p)[:, None] * s_tok + h * HD
        kq = tl.load(kpa + offs_d[None, :], mask=pmk[:, None], other=0.0).to(tl.float32)
        kq2 = tl.load(kpa + offs_dp[None, :], mask=pmk[:, None], other=0.0).to(tl.float32)
        kq = (kq * tl.load(COS + pp[:, None] * s_cs + offs_d[None, :]) + kq2 * sgn[None, :] * tl.load(SIN + pp[:, None] * s_cs + offs_d[None, :])).to(KK.dtype.element_ty)
        vq = tl.load(VV + (base + offs_p)[:, None] * s_tok + h * HD + offs_d[None, :], mask=pmk[:, None], other=0.0)
        pm = tl.load(PM + offs_p, mask=pmk, other=0)
        sq = tl.dot(qs, tl.trans(kq)) + tl.where(pmk & (pm != 0), 0.0, float('-inf'))[None, :]
        mq = tl.maximum(m_i, tl.max(sq, 1)); aq = tl.exp(m_i - mq); pq_ = tl.exp(sq - mq[:, None])
        l_i = l_i * aq + tl.sum(pq_, 1); acc = acc * aq[:, None] + tl.dot(pq_.to(vq.dtype), vq); m_i = mq
    acc = acc / l_i[:, None]
    tl.store(OUT + (base + offs_m)[:, None] * s_tok + h * HD + offs_d[None, :], acc.to(OUT.dtype.element_ty), mask=qmask[:, None])


def probe_attention4(q, k, v, k_sh, v_sh, b_sh, pos_b, pos_p, cos, sin, committed, pvalid, bl, K, mp, scale, out=None):
    """probe_attention3 on the raw projections, with rotary fused and a token-major output.
    q/k/v (bl + K*mp, nh*hd) projections of the step's tokens (block row first, then K probe rows of mp tokens);
    k_sh/v_sh (nh, Lc, hd) rotated context; b_sh (Lc,) float32 key bias; pos_b (bl,) / pos_p (mp,) int32 rotary
    positions; cos/sin (Lmax, hd) float32; committed (bl,) int32 block slots visible to probe rows; pvalid (mp,) int32.
    Returns (bl + K*mp, nh*hd), the attention output before attn_out."""
    nh, Lc, hd = k_sh.shape
    out = torch.empty_like(q) if out is None else out
    BLOCK_M = 16 if max(bl, mp) <= 16 else 32 if max(bl, mp) <= 32 else 64
    grid = (triton.cdiv(max(bl, mp if K else 1), BLOCK_M), 1 + K, nh)
    pp = pos_p if K else pos_b
    pv = pvalid if K else committed
    _probe_attn4_fwd[grid](q, k, v, k_sh, v_sh, b_sh, pos_b, pp, cos, sin, committed, pv, out, scale,
                           bl, max(mp, 1), Lc, q.stride(0), k_sh.stride(0), k_sh.stride(1), cos.stride(0),
                           HD=hd, BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_B=max(16, triton.next_power_of_2(bl)),
                           BLOCK_P=max(16, triton.next_power_of_2(max(mp, 1))), num_warps=4, num_stages=2)
    return out
