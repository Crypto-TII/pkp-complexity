
from math import log2, lgamma, log, log1p, comb, factorial
from functools import lru_cache

# =========================
# Numerics: stable log base-2 utilities
# =========================
LN2 = log(2.0)

def check_number_of_solutions(n, m, q, ell):
    return log2(factorial(n)) - log2(q) * m * ell

def minimal_w(n,k,q):
    '''
    Return the minimum weight of a (n,k) code over Fq
    :param n: length
    :param k: dimension
    :param q: modulo
    :return: w, minimum Hamming weight
    '''
    w = 1
    s = 0
    while True:
        s += comb(n,w)*(q-1)**(w-1)
        if s > q**(n-k):
            return w
        w = w+1





def logsumexp2(values):
    """Stable log2(sum_i 2^{values[i]}). Ignores -inf terms."""
    vals = [v for v in values if v != float('-inf')]
    if not vals:
        return float('-inf')
    m = max(vals)
    return m + log2(sum(2 ** (v - m) for v in vals))

# =========================
# Basic combinatorial logs
# =========================
@lru_cache(maxsize=None)
def log2_factorial(n: int) -> float:
    """log2(n!) via lgamma for stability and speed."""
    if n < 0:
        return float('-inf')
    return lgamma(n + 1) / LN2

@lru_cache(maxsize=None)
def log2_binomial(n: int, k: int) -> float:
    """log2(C(n,k))."""
    if k < 0 or k > n:
        return float('-inf')
    return log2_factorial(n) - log2_factorial(k) - log2_factorial(n - k)

def log2_binomial_minus1(n: int, k: int) -> float:
    """log2(C(n,k) - 1) with safe edge cases."""
    if k == 0 or k == n:
        return float('-inf')  # C(n,0)-1 = 0 or C(n,n)-1 = 0
    lb = log2_binomial(n, k)
    return lb + log1p(-2 ** (-lb)) / LN2

@lru_cache(maxsize=None)
def log2_q_pow_minus_one(q: int, t: int) -> float:
    """log2(q^t - 1) computed without huge ints."""
    if t <= 0:
        return float('-inf')
    return t * log2(q) + log1p(-q ** (-t)) / LN2

@lru_cache(maxsize=None)
def gauss_binomial_log2(n: int, k: int, q: int) -> float:
    """
    log2([n choose k]_q) via product of (q^{n-i}-1)/(q^{k-i}-1) in logs.
    """
    if k < 0 or k > n:
        return float('-inf')
    s = 0.0
    for i in range(k):
        s += log2_q_pow_minus_one(q, n - i) - log2_q_pow_minus_one(q, k - i)
    return s

# =========================
# Provided COL_REP^{(u)}(w,d,q)
# =========================
@lru_cache(maxsize=None)
def _compute_list_of_repetitions_cached(w: int, q: int, d: int):
    """
    Your COL_REP(w,d,q) function, cached. Returns a tuple for caching.
    """
    if q**d - 1 >= w:
        return tuple(1 for _ in range(w))
    N = w // (q**d - 1)
    L = [N for _ in range(q**d - 1)]
    if sum(L) == w:
        return tuple(L)
    else:
        for i in range(q**d - 1):
            L[i] += 1
            if sum(L) == w:
                return tuple(L)

def compute_list_of_repetitions(w: int, q: int, d: int):
    return list(_compute_list_of_repetitions_cached(w, q, d))

# =========================
# Validity gate from your formula
# =========================



# =========================
# N_swaps^{(u)}(w,d,q)
# =========================
@lru_cache(maxsize=None)
def N_swaps_u_log2(w: int, d: int, q: int) -> float:
    """
    log2 N_swaps^{(u)}(w,d,q) = log2( ∏ u_i! ) where (u_i) = COL_REP^{(u)}(w,d,q).
    """
    L = _compute_list_of_repetitions_cached(w, q, d)
    return sum(log2_factorial(u) for u in L)

# =========================
# Exact DP for T_pkp1 using COL_REP^{(u)}(n,r,q) via your function
# =========================
@lru_cache(maxsize=None)
def T_pkp1_log2(n: int, r: int, q: int, l: int) -> float:
    """
    T_pkp1(n,r,q,l) = sum_{(mu) in J} (n-l)! / ∏ mu_i!,
    computed exactly via DP with multiplicity bounds COL_REP^{(u)}(n,r,q),
    using your compute_list_of_repetitions signature as COL_REP(n,r,q) ≡ compute_list_of_repetitions(n,q,r).
    """
    L = _compute_list_of_repetitions_cached(n, q, r)  # multiplicities, sum = n
    k = n - l
    if k < 0 or k > n:
        return float('inf')

    dp = [0] * (k + 1)
    dp[0] = 1

    for m_i in L:
        new = [0] * (k + 1)
        if m_i == 1:
            # Fast path: new[r] = dp[r] + r * dp[r-1]
            for r_used in range(k + 1):
                v = dp[r_used]
                if r_used > 0:
                    v += dp[r_used - 1] * r_used
                new[r_used] = v
        else:
            for r_used in range(k + 1):
                max_t = min(m_i, r_used)
                s = 0
                for t in range(max_t + 1):
                    s += dp[r_used - t] * comb(r_used, t)
                new[r_used] = s
        dp = new

    return log2(dp[k]) if dp[k] > 0 else float('-inf')

# =========================
# Inner SBC^(1) and SBC^* (with inner tilde r*)
# =========================
def inner_sbc1_log2(n_in: int, r_in: int, q: int, l_in: int,
                    w1: int, w2: int, d_in: int, rt_in: int, T_ISD_log: float, ALT: bool):
    """
    Implements Proposition 'complsbc1' in log2.
    """
    w = w1 + w2
    Nw1 = T_pkp1_log2(n_in, l_in, q, n_in - w1)
    Nw2 = T_pkp1_log2(n_in, l_in, q, n_in - w2)

    Tk_star = logsumexp2([
        Nw1,
        Nw2,
        Nw1 + Nw2 - d_in * l_in * log2(q)
    ])

    term_cap=T_pkp1_log2(n_in,l_in,q,n_in-w1-w2)-d_in * l_in * log2(q)

    if ALT:
        Nsw_log = N_swaps_u_log2(w, l_in, q)
        K_log_zero=term_cap+Nsw_log
        K_log = max(term_cap, 0.0)+Nsw_log
        if w>n_in-(l_in+1)+rt_in:
            prob = min(log2_factorial(l_in - rt_in + 1) - (r_in - 1) * (l_in + 1 - rt_in) * log2(q), 0.0)
            L_log=max(K_log+prob,0)
            L_log_zero=K_log_zero+prob
            Tsbc0=logsumexp2([T_ISD_log, Tk_star, K_log_zero])
            T_sbc1=logsumexp2([T_ISD_log, Tk_star, K_log])
            return T_sbc1, Tsbc0, L_log_zero, L_log

        TL_zero = K_log_zero+log2_factorial(n_in-w1-w2)-log2_factorial(l_in+1-rt_in)
        TL=K_log + log2_factorial(n_in - w1 - w2) - log2_factorial(l_in + 1 - rt_in)
        prob=min(log2_factorial(l_in-rt_in+1)-(r_in-1)*(l_in+1-rt_in)*log2(q),0.0)

        L_log_zero = TL_zero +prob-rt_in*(r_in-1)*log2(q)
        L_log = max(TL +prob-rt_in*(r_in-1)*log2(q), 0)

        return logsumexp2([T_ISD_log, Tk_star, TL]), logsumexp2([T_ISD_log, Tk_star, TL_zero]), L_log_zero, L_log


    else:
        K_log_zero = term_cap
        K_log = max(K_log_zero, 0)
        if n_in<r_in:
            if rt_in==r_in:
                rt_in-=1
            r_in-=1

        # This check is not needed if T_ISD_star is not used

        if w>n_in-r_in+rt_in:
            prob = min(
                log2_factorial(r_in - rt_in) - (l_in) * (r_in - rt_in) * log2(q) - N_swaps_u_log2(r_in - rt_in, l_in,
                                                                                                  q), 0.0)
            L_log=max(K_log+prob,0)
            L_log_zero=K_log_zero+prob
            Tsbc0=logsumexp2([T_ISD_log, Tk_star, K_log_zero])
            T_sbc1=logsumexp2([T_ISD_log, Tk_star, K_log])
            return T_sbc1, Tsbc0, L_log, L_log_zero

        N_n_r_w_rt = T_pkp1_log2(n_in, l_in, q, r_in + w - rt_in)
        N_n_r_plus_rt = T_pkp1_log2(n_in - w, l_in, q, r_in - rt_in)

        TL = logsumexp2([
            N_n_r_w_rt,
            K_log,
            N_n_r_w_rt + K_log - l_in * (rt_in - d_in) * log2(q)
        ])
        TL_zero = logsumexp2([
            N_n_r_w_rt,
            K_log_zero,
            N_n_r_w_rt + K_log_zero - l_in * (rt_in - d_in) * log2(q)
        ])

        L_log = N_n_r_plus_rt + K_log - l_in * (rt_in - d_in) * log2(q)
        L_log_zero = N_n_r_plus_rt + K_log_zero - l_in * (rt_in - d_in) * log2(q)

        Tsbc0=logsumexp2([T_ISD_log, Tk_star, TL_zero, L_log_zero])
        T_sbc1 = logsumexp2([T_ISD_log, Tk_star, TL, L_log])
        prob=min(log2_factorial(r_in - rt_in) - (l_in) * (r_in - rt_in) * log2(q)-N_swaps_u_log2(r_in-rt_in,l_in,q),0.0)
        L_log_zero = L_log_zero + prob
        L_log = L_log + prob
    return T_sbc1, Tsbc0, L_log, L_log_zero

def inner_sbc_star_logs(n_in: int, r_in: int, q: int, l_in: int,
                        w1: int, w2: int, d_in: int, rt_in: int, isd: float, ALT: bool):
    """
    Returns (T_sbc^* (log2), T_sbc^*^(0) (log2)).
    Uses Corollary definition of N_sol in log2.
    """
    if ALT:
        return  inner_sbc1_log2(n_in, r_in, q, l_in, w1, w2, d_in, rt_in, isd, ALT)
    else:
        T1, T0, L_log, L_log_zero= inner_sbc1_log2(n_in, r_in, q, l_in, w1, w2, d_in, rt_in, isd, ALT)

        # N_sol = max( min( n!/q^{l r}, |L| * N_swaps^{(u)}(w, d, q) ), N_swaps^{(u)}(w, d, q) )
        Nsw_log = N_swaps_u_log2(n_in, l_in, q)
        #cap = log2_factorial(n_in) - l_in * (r_in-1) * log2(q)
        #cap=inf
        L_times_swaps = L_log + Nsw_log
        L_times_swaps_zero=L_log_zero + Nsw_log


        N_sol = max(L_times_swaps, Nsw_log)
        # Zero-solution variant (analogy with pkp2^(0)): drop the outer max with N_swaps
        N_sol_zero = L_times_swaps_zero


        #T_sbc_star = logsumexp2([T1, N_sol])
        #T_sbc_star_zero = logsumexp2([T0, N_sol_zero])
        T_sbc_star=T1
        T_sbc_star_zero=T0

        return T_sbc_star, T_sbc_star_zero, N_sol_zero, N_sol

# =========================
# Outer blocks for RECURSIVE ALGORITHM
# =========================
@lru_cache(maxsize=None)
def T_K3_log2(n: int, r: int, q: int, l: int, ls: int,
              w: int, d: int,
              w1s: int, w2s: int, ds: int, rts: int, isd: float, ALT_int: bool) -> [float, int, int]:
    """
    T_K^{(3)}(n,w,d,q,l,w1*,w2*,d*) using inner SBC^* with inner tilde r* = rts.
    """
    # Inner problem parameters:
    n_in = w
    l_in = d
    r_in=ls+1
    #if l_in >= r_in - 1 - ds:
    T_star, T_star_zero, l1, l2 = inner_sbc_star_logs(n_in, r_in, q, l_in, w1s, w2s, ds, rts, isd, ALT_int)
    return logsumexp2([log2_binomial_minus1(n, w) + T_star_zero, T_star]), l1, l2


def K_outer_log2(n: int, r: int, q: int, l: int, ls: int,
                 w: int, d: int,
                 w1s: int, w2s: int, ds: int, rts: int, isd: float, ALT_int: bool) -> float:
    """
    |K| = max{ min{ n! q^{-d l} / (n-w)!, T_K^{(3)} }, N_swaps^{(u)}(w,d,q) }
    """
    Tk3,l1,l2 = T_K3_log2(n, r, q, l, ls, w, d, w1s, w2s, ds, rts, isd, ALT_int)
    K_size=logsumexp2([log2_binomial_minus1(n,w)+l1,l2])
    return K_size

def T_L_outer_log2(n: int, r: int, q: int, l: int,
                   w: int, d: int, rt: int,
                   w1s: int, w2s: int, ds: int, rts: int, isd, K_log) -> float:
    """
    T_L (outer) = n!/(r+w-rt)! + |K| + (n!*|K|*q^{-l(rt-d)})/(r+w-rt)!
    """
    term1 = log2_factorial(n) - log2_factorial(r + w - rt)
    term2 = K_log
    term3 = log2_factorial(n) + K_log - l * (rt - d) * log2(q) - log2_factorial(r + w - rt)
    return logsumexp2([term1, term2, term3])

def T_L_outer_log2_alt(n: int, r: int, q: int, l: int,
                   w: int, d: int, rt: int,
                   w1s: int, w2s: int, ds: int, rts: int, isd: float, K_log: float) -> float:
    """
    T_L (outer) = n!/(r+w-rt)! + |K| + (n!*|K|*q^{-l(rt-d)})/(r+w-rt)!
    """
    if n-w<l+1-rt:
        return K_log
    TL=K_log + log2_factorial(n-w) - log2_factorial(l+1-rt)
    return TL

def L_outer_log2(n: int, r: int, q: int, l: int,
                 w: int, d: int, rt: int,
                 w1s: int, w2s: int, ds: int, rts: int, isd: float, K_log: float) -> float:
    """
    |L| (outer) = (n-w)! * |K| * q^{-l(rt-d)} / (r - rt)!
    """

    return (log2_factorial(n - w)
            + K_log
            - l * (rt - d) * log2(q)
            - log2_factorial(r - rt))

def T_doublesbc_log2(n: int, r: int, q: int, l: int, ls:int,
                     w: int, d: int, rt: int,
                     w1s: int, w2s: int, ds: int, rts: int,
                     T_ISD_log2: float , isd: float, ALT_ext: bool, ALT_int: bool) -> float:
    """
    T_doublesbc = T_ISD^{(d)} + T_K^{(3)} + T_L + |L|  (all log2, combined via log-sum-exp)
    """

    Tk3, _, _ = T_K3_log2(n, r, q, l, ls, w, d, w1s, w2s, ds, rts, isd, ALT_int)
    K_log = K_outer_log2(n, r, q, l, ls, w, d, w1s, w2s, ds, rts, isd, ALT_int)

    if ALT_ext:
        TL = T_L_outer_log2_alt(n, r, q, l, w, d, rt, w1s, w2s, ds, rts, isd, K_log)
        return logsumexp2([T_ISD_log2, Tk3, TL])
    else:
        TL = T_L_outer_log2(n, r, q, l, w, d, rt, w1s, w2s, ds, rts, isd, K_log)
        Lg= L_outer_log2(n, r, q, l, w, d, rt, w1s, w2s, ds, rts, isd, K_log)
    return logsumexp2([T_ISD_log2, Tk3, TL, Lg])
