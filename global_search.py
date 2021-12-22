import numpy as np
from scipy.optimize import basinhopping, differential_evolution, shgo, dual_annealing
from src import IntegratedCycle
from time import time
import pickle
import pandas as pd


def worker2(x):
    pmax, pr1, pr2, pr3, f1, f2, f3 = x
    pmax *= 0.98e6
    pmax += 8.22e6
    model = IntegratedCycle(pmax, pr1, pr2, pr3, f1, f2, f3)
    try:
        model.solve()
        eta_FL = model.eta_FL
        LCOE, LCOE_perc_change, cost_terms = model.LCOE()
        penalty, pen_terms = model.penalty()
    except ValueError as ve:
        eta_FL = 0
        LCOE = np.inf
        LCOE_perc_change = np.inf
        penalty = np.inf
    # print('{:12.4f}{:12.4f}{:12.4f}{:12.4f}{:12.4f}{:12.4f}{:12.4f}               {:12.4f}{:12.4f}{:12.4f}{:12.4f}'.format(pmax, pr1, pr2, pr3, f1, f2, f3, LCOE, LCOE_perc_change, eta_FL, penalty))
    # print(pmax, pr1, pr2, pr3, f1, f2, f3, '\t\t', LCOE, LCOE_perc_change, eta_FL, penalty)
    return LCOE + penalty


def worker3(x):
    pmax, pr1, pr2, pr3, f1, f2, f3 = x
    model = IntegratedCycle(pmax * 0.98e6 + 8.22e6, pr1, pr2, pr3, f1, f2, f3)
    try:
        model.solve()
        eta_FL = model.eta_FL
        LCOE, LCOE_perc_change = model.LCOE()
        penalty = model.penalty()
    except ValueError as ve:
        eta_FL = 0
        LCOE = np.inf
        LCOE_perc_change = np.inf
        penalty = np.inf
    print('{:12.4f}{:12.4f}{:12.4f}{:12.4f}{:12.4f}{:12.4f}{:12.4f}               {:12.4f}{:12.4f}{:12.4f}{:12.4f}'.format(pmax, pr1, pr2, pr3, f1, f2, f3, LCOE, LCOE_perc_change, eta_FL, penalty))
    return pmax, pr1, pr2, pr3, f1, f2, f3, LCOE, LCOE_perc_change, eta_FL, penalty


if __name__ == '__main__':
    # x0 = [0.35, 0.18, 0.32, 0.39, 0.13, 0.13, 0.17]  # approx means from low LCOE box plots
    # x0 = [0.00121059, 0.66859519, 0.27470807, 0.15685518, 0.30904858, 0.17769498, 0.17881965]  # approx means from low LCOE box plots
    # res = basinhopping(func=worker2, x0=x0, niter=10, disp=True, niter_success=5)
    # bounds = [(0, 1)] * 7

    # Dual Annealing Point:
    # Exec Time: 1820 sec
    #      fun: 77.30089199215124
    #  message: ['Maximum number of iteration reached']
    #     nfev: 15211
    #     nhev: 0
    #      nit: 1000
    #     njev: 0
    #   status: 0
    #  success: True
    #        x: array([2.16137040e-04, 2.25562653e-01, 1.50004312e-01, 3.02050525e-01,
    #        1.86440208e-04, 7.45553522e-02, 5.70058948e-02])
    #
    # Outputs: LCOE: 77.3009    LCOE % Change: -18.9646      ETA: 0.3028      Penalty: 0.0000
    # baseline = np.array([2.16137040e-04, 2.25562653e-01, 1.50004312e-01, 3.02050525e-01, 1.86440208e-04, 7.45553522e-02, 5.70058948e-02])
    # worker2(baseline)
    # exit()

    # DIFFERENTIAL EVOLUTION
    # Execution Time: 1675.110895395279 sec
    #      fun: 77.394807358319
    #  message: 'Optimization terminated successfully.'
    #     nfev: 13650
    #      nit: 38
    #  success: True
    #        x: array([0.00395981, 0.09750805, 0.33170696, 0.46669163, 0.00211851,
    #        0.05917469, 0.07701761])
    # baseline = np.array([0.00395981, 0.09750805, 0.33170696, 0.46669163, 0.00211851, 0.05917469, 0.07701761])
    # worker2(baseline)
    # exit()

    # baseline = np.array([1.24466080e-04, 3.31720291e-01, 1.70881551e-01, 1.89183662e-01, 8.14953555e-04, 1.55046972e-04, 5.98168405e-02])  # Dual Annealing v2
    # baseline = np.array([4.25223426e-04, 1.12703637e-01, 2.09448156e-01, 4.32956702e-01, 1.40615927e-03, 5.64105033e-02, 5.68420877e-02])  # DiffEv v2
    # worker2(baseline)
    # exit()

    bounds = [(0, 1),
              (0.0001, 0.9999),
              (0.0001, 0.9999),
              (0.0001, 0.9999),
              (0.0001, 0.9999),
              (0.0001, 0.9999),
              (0.0001, 0.9999)]
    start = time()
    res = dual_annealing(func=worker2, bounds=bounds)
    # res = differential_evolution(func=worker2, bounds=bounds, disp=True, maxiter=200, popsize=50, mutation=0.5,
    #                              recombination=0.7, polish=False)
    stop = time()
    print('Dual Annealing:')
    print('Execution Time:', stop - start)
    print(res)
    print('\n\n')

    start = time()
    # res = dual_annealing(func=worker2, bounds=bounds)
    res = differential_evolution(func=worker2, bounds=bounds, disp=True, maxiter=200, popsize=50, mutation=0.5,
                                 recombination=0.7, polish=False)
    stop = time()
    print('Differential Evolution:')
    print('Execution Time:', stop - start)
    print(res)

    # print(worker2([8.50849025e-06, 7.11039505e-02, 5.43629019e-01, 4.77225272e-01, 3.52177475e-02, 3.64198849e-03, 1.47246084e-01]))

    # minima = []

    # for i in range(10):
    #     start = time()
    #     res = differential_evolution(func=worker2, bounds=bounds, disp=True, maxiter=200, popsize=50, mutation=0.5, recombination=0.7, polish=False)
    #     stop = time()
    #     print(res)
    #
    #     print('Exec Time:  {:.3f}  sec'.format(stop - start))
    #
    #     # print('\n\nValidation:')
    #     # print('Surrogate:')
    #     # start = time()
    #     # # worker2([0.00121059, 0.66859519, 0.27470807, 0.15685518, 0.30904858, 0.17769498, 0.17881965], use_surrogate=True, tune=False)
    #     # worker2(res.x, use_surrogate=True, tune=False)
    #     # stop = time()
    #     # print(stop - start)
    #     #
    #     # print('Surrogate (w/ tune):')
    #     # start = time()
    #     # # worker2([0.00121059, 0.66859519, 0.27470807, 0.15685518, 0.30904858, 0.17769498, 0.17881965], use_surrogate=True, tune=True)
    #     # worker2(res.x, use_surrogate=True, tune=True)
    #     # stop = time()
    #     # print(stop - start)
    #     #
    #     print('Optimal Output:')
    #     # start = time()
    #     # # worker2([0.00121059, 0.66859519, 0.27470807, 0.15685518, 0.30904858, 0.17769498, 0.17881965], use_surrogate=False, tune=False)
    #     minima.append(worker3(res.x))
    #     print('\n\n\n')
    #     # stop = time()
    #     # print(stop - start)
