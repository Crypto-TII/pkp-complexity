import argparse

from kmp import kmp_complexity
from sbc import sbc_complexity
from recursive_algorithm import recursive_algorithm_complexity
from utils import findq
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from tqdm import tqdm

def plot_time(T_kmp, T_sbc, T_recursive, all_ell, n, r):
    """
    Plot timing lists against index ell ranging from min_l to n-r.

    Args:
        T_kmp, T_sbc, T_recursive (list of float):
            Lists of values (all must have the same length).
        all_ (list of int): Valies for the x-axis.
        n (int): Code length parameter.
        r (int): Dimension/offset parameter.
    """

    plt.plot(all_ell, T_kmp, label="KMP", marker="o")
    plt.plot(all_ell, T_sbc, label="SBC", marker="o")
    plt.plot(all_ell, T_recursive, color="purple", label="Recursive", marker="o")

    plt.xlabel(r"$\ell$")
    plt.ylabel("Values")
    plt.title("Comparison of Algorithms")
    plt.legend()
    plt.grid(True)

    filename = True
    if filename:
        plt.savefig(f"plot_n_{n}_r_{r}.png", bbox_inches="tight")
    else:
        plt.show()

def process_task(l, n, r):
        q = findq(n, r, l)
        if q == 2:
            return 0.0,0.0,0.0
        if l > r:
            _, kmp = kmp_complexity(n, l + 1, r, q, debug=False)
            sbc, _ = sbc_complexity(n, l + 1, r, q, parallel=False, debug=False)
        else:
            _, kmp = kmp_complexity(n, r + 1, l, q, debug=False)
            sbc, _ = sbc_complexity(n, r + 1, l, q, parallel=False, debug=False)

        tmp1 = recursive_algorithm_complexity(n, r + 1, l, q, debug=False)
        tmp2 = recursive_algorithm_complexity(n, l + 1, r, q, debug=False)
        recursive_algorithm = round(min(tmp1, tmp2), 3)

        return kmp, sbc, recursive_algorithm

def main(n, r, min_l):
    
    all_ell = list(range(min_l, n - r))  # keep your original half-open range

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(process_task, all_ell, repeat(n), repeat(r)), total=len(all_ell))
        )

    # Unpack results preserving order
    T_kmp, T_sbc, T_recursive_algorithm = map(list, zip(*results))

    T_kmp = [round(x,3) for x in T_kmp if x != 0.0]
    T_sbc = [round(x,3) for x in T_sbc if x != 0.0]
    T_recursive_algorithm = [round(x,3) for x in T_recursive_algorithm if x != 0.0]

    all_ell = [i for i in range(1, len(T_kmp)+1)]

    print("\n\n")
    print(f"all_ell {all_ell}")
    print(f"T_kmp {T_kmp}")
    print(f"T_sbc {T_sbc}")
    print(f"T_recursive_algorithm {T_recursive_algorithm}")

    plot_time(T_kmp, T_sbc, T_recursive_algorithm, all_ell, n, r)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the behavior of the algorithms.')
    parser.add_argument('--n', type=int, default=60, help='Length of the linear code.')
    parser.add_argument('--r', type=int, default=30, help='Dimension of parity-check (not extended yet).')
    parser.add_argument('--min_l', type=int, default=1, help='Minimum dimension of subcode.')

    # Parse the command-line arguments
    args = parser.parse_args()

    assert (args.min_l < args.n-args.r), "l cannot be larger than n-r"

    main(args.n, args.r, args.min_l)

