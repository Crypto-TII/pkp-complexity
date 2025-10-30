

from utils import N_check_log, T_ISD
from recursive_algorithm_utils import *
import argparse
from concurrent.futures import ProcessPoolExecutor
from math import inf

def process_task_doublesbc(task):
    (d, w, n, r, q, l, ALT_ext, ALT_int) = task
    best = None
    best_val = float('inf')
    best_Tk3 = best_TL = best_L = best_K = None

    if ALT_ext:
        RR=l+1
        LOW=0
        LL=r-1
    else:
        RR=r
        LOW=d
        LL=l
    if n==w and r-1==d:
        check=True
        tisd=-inf
    elif N_check_log(n, r, w, d, q) > 0:
        check=True
        tisd = T_ISD(n, r, w, d, q)
    else:
        check=False
    if check:
        if w > d:
            if tisd != inf:
                    # inner search over w* and its splits
                    ls=min(w-d+1,l)
                    min_wstar = minimal_w(w, ls + 1, q)
                    for wstar in range(min_wstar, w+1):
                        for ds in range(1, ls+2):
                            if wstar == w and ls  == ds:
                                checkint = False
                                tisdst = -inf
                            elif N_check_log(w, ls + 1, wstar, ds, q) > 0:
                                checkint = False
                                tisdst = T_ISD(w, min(ls + 1, w), wstar, ds, q)
                            else:
                                checkint = True
                            if checkint:
                                continue
                            tisdst = T_ISD(w,min(ls+1,w), wstar, ds, q)
                            if tisdst != inf:
                                for w1s in range(1, wstar // 2 + 1):
                                    w2s = wstar - w1s

                                    if ALT_int:
                                        low=0
                                        high=d+1
                                    else:
                                        low=ds
                                        high=ls+1
                                    for rts in range(low, high+1):
                                        for rt in range(LOW, RR+1):
                                            total = T_doublesbc_log2(n, r, q, l, ls, w, d, rt, w1s, w2s, ds, rts, tisd, tisdst, ALT_ext, ALT_int)
                                            if total < best_val:
                                                best_val = total
                                                best = (w, d, rt, w1s, w2s, ds, rts, ls)
                                                # Cache intermediate values for reporting
                                                best_Tk3 = T_K3_log2(n, r, q, l, ls, w, d, w1s, w2s, ds, rts, tisdst,  ALT_int)
                                                best_K = K_outer_log2(n, r, q, l, ls, w, d, w1s, w2s, ds, rts, tisdst,  ALT_int)
                                                if ALT_ext:
                                                    best_TL, best_L = T_L_outer_log2_alt(n, r, q, l, w, d, rt, w1s, w2s, ds, rts, tisdst, best_K), 0
                                                else:
                                                    best_TL = T_L_outer_log2(n, RR, q, LL, w, d, rt, w1s, w2s, ds, rts,
                                                                             tisdst, best_K)
                                                    best_L = L_outer_log2(n, RR, q, LL, w, d, rt, w1s, w2s, ds, rts, tisdst,
                                                                          best_K)


    return best_val, best, best_Tk3, best_TL, best_L, best_K


def double_sbc_complexity2(n, r, ell, q, ALT_ext, ALT_int, parallel=False, debug=False):
    best_L = []
    min_val=inf
    best=None
    w_min = minimal_w(n, r, q)
    tasks = [
        (d, w, n, r, q, ell, ALT_ext, ALT_int)
        for d in range(1,r+1)
        for w in range (w_min,n+1)
    ]

    if parallel:
        # Parallelize the tasks
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_task_doublesbc, tasks))
    else:
        results = list(map(process_task_doublesbc , tasks))

    for result in results:
        try:
            val, best1 ,best_Tk31, best_TL1, best_L1, best_K1= result
            if val<min_val:
                min_val=val
                best=best1
                best_Tk3=best_Tk31
                best_TL=best_TL1
                best_L=best_L1
                best_K=best_K1
        except:
            pass
    if  best is not None:
        if debug:
            print(
                f"\nRec-dual-SBC for n: {n}, r: {r}, ell: {ell}, q: {q}\n"
                f"Best (w,d,r_tilde, w1s, w2s, ds, rts) = {best}\n"
                f"T_K^(3)   (log2): {best_Tk3}\n"
                f"K         (log2): {best_K}\n"
                f"T_L       (log2): {best_TL}\n"
                f"|L|       (log2): {best_L}\n"
                f"T_total   (log2): {min_val}\n"
                f"===================================================="
            )
    return min_val

def recursive_algorithm_complexity(n, r, ell, q, debug=False):
    term1 = double_sbc_complexity2(n, r, ell, q, True, True, parallel=False, debug=debug)
    term2 = double_sbc_complexity2(n, r, ell, q, True, False, parallel=False, debug=debug)
    term3 = double_sbc_complexity2(n, r, ell, q, False, True, parallel=False, debug=debug)
    term4 = double_sbc_complexity2(n, r, ell, q, False, False, parallel=False, debug=debug)
    #return term4
    return min(term1, term2, term3, term4)

def main():
    parser = argparse.ArgumentParser(description="Compute the complexity of our recursive algorithm.")
    parser.add_argument("n", type=int, help="Code length n")
    parser.add_argument("r", type=int, help="Dimension r")
    parser.add_argument("l", type=int, help="Parameter l")
    parser.add_argument("q", type=int, help="Field size q")
    args = parser.parse_args()

    T = recursive_algorithm_complexity(args.n, args.r + 1, args.l, args.q)
    print(f"T = {T}")

if __name__=="__main__":
    main()


