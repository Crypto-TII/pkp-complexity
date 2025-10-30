from math import log2
import argparse

from utils import T_ISD, N_check_log, log2_factorial, logsumexp2
from concurrent.futures import ProcessPoolExecutor

def SBC_complexity(n, r, w1, w2, r_tilde, d, l, q):
    """Compute SBC complexity (not logged)."""
    w = w1 + w2
    if w > n:
        return float("inf")
    Tisd = T_ISD(n, r, w, d, q)
    if Tisd==float("inf"):
        return float("inf")
    # |K|
    K_size=max(log2_factorial(n)-(d*l*log2(q))-log2_factorial(n-w),0)

    # T_K
    t1=log2_factorial(n)-log2_factorial(n-w1)
    t2=log2_factorial(n)-log2_factorial(n-w2)
    t3=t1+t2-(d*l*log2(q))
    T_K=logsumexp2([t1,t2,t3])

    t1 = log2_factorial(n) - log2_factorial(r + w - r_tilde)
    t2 = t1 + K_size - (l * (r_tilde - d) * log2(q))
    T_L=logsumexp2([t1,K_size,t2])

    L_size=log2_factorial(n-w)+K_size-(l*(r_tilde - d)*log2(q))-log2_factorial(r-r_tilde)

    return logsumexp2([Tisd , T_K , T_L , L_size])

def process_task(task):
    n, r, l, q, d = task
    best_val = 100000000
    best_params = (0, 0, 0, 0)
    low=d
    high =r
    for w1 in range(1, n + 1):
        for w2 in range(1, n + 1 - w1):
            w = w1 + w2
            if N_check_log(n, r, w, d, q) <= 0:
                continue
            for r_tilde in range(low, high + 1):
                log_val = SBC_complexity(n, r, w1, w2, r_tilde, d, l, q)
                if log_val == float("inf"):
                    continue
                if log_val < best_val:
                    best_val = log_val
                    best_params = (d, w1, w2, r_tilde)
    return (best_val, best_params)

def sbc_complexity(n, r, l, q, parallel=False, debug=False):
    """Search over d, w1, w2, r_tilde to minimize SBC complexity."""
    best_val = float("inf")
    best_params = None

    tasks = [(n, r, l, q, d) for d in range(1, r+1)]
    if parallel:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_task, tasks))
    else:
        results = list(map(process_task, tasks))

    best_val = 100000
    for result in results:
        val, params = result
        if val < best_val:
            best_val = val
            best_params = params

    if debug:
        print(
            f"SBC for n: {n}, r: {r}, ell: {l}, q {q}\n"
            f"Optimal parameters     : d={best_params[0]}, w1={best_params[1]}, w2={best_params[2]}, r_tilde={best_params[3]}\n"
            f"T_total   (log2)       : {best_val:.2f}\n"
            f"===================================================="
        )


    return best_val, best_params

def main():
    parser = argparse.ArgumentParser(description="Compute the complexity of SBC.")
    parser.add_argument("n", type=int, help="Code length n")
    parser.add_argument("r", type=int, help="Dimension r")
    parser.add_argument("l", type=int, help="Parameter l")
    parser.add_argument("q", type=int, help="Field size q")
    args = parser.parse_args()

    best_val, params = sbc_complexity(args.n, args.r+1, args.l, args.q, debug=True)

    if not params:
        print("No valid parameters found.")

if __name__ == "__main__":
    main()
