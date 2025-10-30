
from math import log2, log, factorial, log1p, lgamma, comb, exp
from sympy import factorint
from functools import lru_cache

def is_prime_power(n):
    if n <= 1:
        return False
    factors = factorint(n)
    # prime power ⇔ exactly one prime factor
    return len(factors) == 1


def findq(n,m,ell):
    q=2
    while True:
        if is_prime_power(q):
            if check_number_of_solutions(n,m,q,ell)<0:
                return q
        q+=1


LN2=log(2)
def log2_q_pow_minus_one(q: int, t: int) -> float:
    """log2(q^t - 1) computed without huge ints."""
    if t <= 0:
        return float('-inf')
    return t * log2(q) + log1p(-q ** (-t)) / LN2

def logsumexp2(vals):
    vals = [v for v in vals if v != float('-inf')]
    if not vals:
        return float('-inf')
    m = max(vals)
    return m + log2(sum(2 ** (v - m) for v in vals))

@lru_cache(maxsize=None)
def log2_factorial(n: int) -> float:
    if n < 0:
        return float('-inf')
    return lgamma(n + 1) / LN2

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


def log2_factorial(n: int) -> float:
    """log2(n!) via lgamma for stability and speed."""
    if n < 0:
        return float('-inf')
    return lgamma(n + 1) / LN2

def log2_binomial(n: int, k: int) -> float:
    """log2(C(n,k))."""
    if k < 0 or k > n:
        return float('-inf')
    return log2_factorial(n) - log2_factorial(k) - log2_factorial(n - k)

def N_check_log(n, k, w, d, q):
    """Logarithm of N_check_k(w,d) to avoid overflow."""
    if w < d:
        return float('-inf')
    # invalid
    log_val = log2_binomial(n, w)
    log_val += (w - d) * log2_q_pow_minus_one(q, d)
    log_val += gauss_binomial_log2(k, d, q)
    log_val -= gauss_binomial_log2(n, d, q)
    return log_val



def T_ISD(n, k, w, d, q):
    """Compute T_ISD^(d)(n,k,w)."""
    if n-w<k-d:
        return float("inf")
    denom=log2_binomial(w,d)+log2_binomial(n-w,k-d)+N_check_log(n,k,w,d,q)-log2_binomial(n,k)
    #if denom == -float("inf"):
    #    return float("inf")
    return logsumexp2([3*log2(k),log2_binomial(k,d)])-denom

def T_ISD_star(n, k, w, d, q):
    """Compute T_ISD^(d)(ren,k,w)."""
    if n-w<k-d:
        return 2*log2(n)
    denom=log2_binomial(w,d)+log2_binomial(n-w,k-d)+N_check_log(n,k,w,d,q)-log2_binomial(n,k)
    #if denom == -float("inf"):
    #    return float("inf")
    return logsumexp2([3*log2(k),log2_binomial(k,d)])-denom

def logsumexp2(values):
    """Stable log2(sum_i 2^{values[i]}). Ignores -inf terms."""
    vals = [v for v in values if v != float('-inf')]
    if not vals:
        return float('-inf')
    m = max(vals)
    s = 0.0
    for v in vals:
        # v - m <= 0, so 2**(v-m) is safe
        s += 2 ** (v - m)
    return m + log2(s)

def log_fact(n):
    return lgamma(n + 1.0) / log(2.0)

def compute_m_vector(n_tilde, r_tilde, q):
    if q**r_tilde-1>=n_tilde:
        return [1 for _ in range(n_tilde)]
    N = n_tilde//(q**r_tilde-1)
    m = [N for _ in range(q**r_tilde-1)]
    if sum(m)==n_tilde:
        return m
    else:
        for i in range(q**r_tilde-1):
            m[i] += 1
            if sum(m)==n_tilde:
                return m

def Compute_Nrank(ell: int, w: int, d: int, q: int) -> int:
    """
    Expected number of matrices X in F_q^{ell x w} with rank < (w - d),
    i.e., the count of such matrices.

    Uses the standard formula for the number of m x n matrices over F_q
    of exact rank r:
        N_{m,n}(r) = prod_{i=0}^{r-1} (q^m - q^i) * prod_{i=0}^{r-1} (q^n - q^i)
                      / prod_{i=0}^{r-1} (q^r - q^i)

    Then sums for r = 0, 1, ..., min(w-d-1, ell, w).

    Args:
        ell (int): number of rows
        w   (int): number of columns
        d   (int): parameter; threshold is rank < (w - d)
        q   (int): field size (prime power)

    Returns:
        int: total count of ell x w matrices over F_q with rank < (w - d)
    """
    import math

    def count_rank_exact(m: int, n: int, r: int, q: int) -> int:
        if r < 0 or r > min(m, n):
            return 0
        if r == 0:
            return 1  # only the zero matrix
        num = 1
        for i in range(r):
            num *= (q**m - q**i)
        for i in range(r):
            num *= (q**n - q**i)
        den = 1
        for i in range(r):
            den *= (q**r - q**i)
        return num // den  # integer

    r_max = min(max(w - d - 1, -1), ell, w)
    if r_max < 0:
        return 0

    total = 0
    for r in range(0, r_max + 1):
        total += count_rank_exact(ell, w, r, q)
    return total

def check_number_of_solutions(n, m, q, ell):
    return log2(factorial(n)) - log2(q) * m * ell

def log_prob_full_rank(q: int, n: int) -> float:


    lnq = log(q)
    # accumulate natural log
    log_prob = 0.0
    for i in range(n):
        # a = (i-n) * ln(q)  -> e^a = q^{i-n} in (0,1)
        a = (i - n) * lnq
        # stable computation of log(1 - exp(a))
        # when a is very negative, exp(a) underflows -> treated as 0 and log(1)=0 contribution.
        term = log1p(-exp(a))
        log_prob += term

    return log_prob
    # convert natural log to requested base

def coeff(n,l,w,q):
    prob_not_max_rank=1-exp(log_prob_full_rank(q,l+1))
    if n-w>l+1:
        N=comb(n-w,l+1)
    else:
        N=comb(w,l+1-n+w)
    try:
        return log1p(-(prob_not_max_rank**N))
    except:
        print(n,l,w,q)
        return -float("inf")
