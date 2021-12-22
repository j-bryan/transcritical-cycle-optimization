import numpy as np
import CoolProp.CoolProp as CP
from time import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


def euler(f, y0, t_span, N):
    t = np.linspace(t_span[0], t_span[1], N + 1)
    y_dims = (N + 1,) if (isinstance(y0, float) or len(y0) == 1) else (N + 1, len(y0))
    y = np.zeros(y_dims)
    y[0] = y0
    for i in range(N):
        h = t[i + 1] - t[i]
        y[i + 1] = y[i] + f(t[i], y[i]) * h
    return y.T


def heun(f, y0, t_span, N):
    t = np.linspace(t_span[0], t_span[1], N + 1)
    y_dims = (N + 1,) if (isinstance(y0, float) or len(y0) == 1) else (N + 1, len(y0))
    y = np.zeros(y_dims)
    y[0] = y0
    for i in range(N):
        h = t[i + 1] - t[i]
        fi = f(t[i], y[i])
        y_guess = y[i] + fi * h
        y[i + 1] = y[i] + 0.5 * (fi + f(t[i + 1], y_guess)) * h
    return y.T


def perr(y_true, y_pred):
    return (y_true - y_pred) / y_true * 100


class PrimaryHX:
    T_p = 310 + 273.15  # hot side inlet
    T_s = 301 + 273.15  # cold side outlet
    tube_material = 'Inconel_718'  # tube material
    RelRough = 0.035  # tube roughness
    d_o = 0.015875  # [m] "outside diameter of steam generator tube. Data from Kevin Drost email"
    d_i = 0.013335  # [m] "inside diameter of steam generator tube.  Data from Kevin Drost email"
    A_primary_flow_min = 1.597932  # [m^2] "minimum flow area between tubes for primary side.  Data from Kevin Drost email"
    N_t = 1380  # Total number of tubes. Data from drawing #: NP12-01-A011-M-SA-2689-S02
    r_c = 0.95885254  # [m] "Averge tube column radius. Data from drawing #: NP12-01-A011-M-SA-2689-S02"
    R_foul_dprime = 5e-6  # [m^2-K/W] "Fouling factor for tube walls. Value selected to match temp profile to NuScale Analysis"
    d_c = 2 * r_c  # average tube column diameter
    k_t = 15.095  # [W/m-K] Thermal Resistance Analogy
    L = 24.4812312  # [m] "Average total tube length. Data from drawing #: NP12-01-A011-M-SA-2689-S02"
    P_primary = 12.8e6  # [Pa]

    def __init__(self, P_max, m_dot_H, m_dot_C, calc_temps=True, rtol=1e-6, maxiter=100):
        self.m_dot_H = m_dot_H
        self.m_dot_C = m_dot_C
        self.m_dot_H_tube = self.m_dot_H / self.N_t
        self.m_dot_C_tube = self.m_dot_C / self.N_t
        self.P_max = P_max

        self.i_H = None
        self.i_C = None
        self.T_H = None
        self.T_C = None
        self.x = None
        self.Q_dot_in = None
        self.DeltaT_SGmin = None

        self.p_fld = CP.AbstractState('REFPROP', 'Water')
        self.s_fld = CP.AbstractState('REFPROP', 'Methanol')

        self.h_H = 9850
        self.h_H_old = 9850
        self.h_C = 7750
        self.h_C_old = 7750

        self.rtol = rtol
        self.maxiter = int(maxiter)
        self.calc_temps = calc_temps

        self.Delta_x = self.L / 50

    def solve(self):
        self.p_fld.update(CP.PT_INPUTS, self.P_primary, self.T_p)
        i_h_in = self.p_fld.hmass()
        self.s_fld.update(CP.PT_INPUTS, self.P_max, self.T_s)
        i_C_out = self.s_fld.hmass()

        # ====== RK45 method =====
        # soln = solve_ivp(self.diff_step, t_span=(0, self.L), y0=[i_h_in, i_C_out], method='RK45', max_step=self.L/5)
        # self.x = soln.t
        # self.i_H, self.i_C = soln.y

        self.T_H = np.zeros(51)
        self.T_C = np.zeros(51)
        self.T_H[0] = self.T_p
        self.T_C[0] = self.T_s

        for i in range(50):
            self.T_H[i + 1], self.T_C[i + 1] = self.step(self.T_H[i], self.T_C[i])
        plt.plot(self.T_H)
        plt.plot(self.T_C)

        T_H_out1 = self.T_H[-1]

        print('\n\n\n\n\n\n')

        # ===== Euler method =====
        self.i_H, self.i_C = euler(self.diff_step, t_span=(0, self.L), y0=[i_h_in, i_C_out], N=50)
        self.x = np.linspace(0, self.L, len(self.i_H))

        if self.calc_temps:
            self.T_H = np.zeros(self.i_H.shape)
            self.T_C = np.zeros(self.i_C.shape)
            for i in range(len(self.i_H)):
                self.p_fld.update(CP.HmassP_INPUTS, self.i_H[i], self.P_primary)
                self.T_H[i] = self.p_fld.T()
                self.s_fld.update(CP.HmassP_INPUTS, self.i_C[i], self.P_max)
                self.T_C[i] = self.s_fld.T()
            self.DeltaT_SGmin = np.min(self.T_H - self.T_C)
            plt.plot(self.T_H)
            plt.plot(self.T_C)
            T_H_out2 = self.T_H[-1]
            print('Rel Error: ', abs(T_H_out1 - T_H_out2) / T_H_out1 * 100, '%')

        plt.show()

        self.Q_dot_in = self.m_dot_H * (i_h_in - self.i_H[-1])

    def step(self, T_H, T_C):
        """
        Takes values from one side of a small section of heat exchanger and calculates the temperatures on the other side of
        the domain.
        """
        # Try to predict the new convection coefficients to keep number of iterations down. We're using the difference
        # between previous values to predict the next value and passing that in as the initial guess for the convection
        # coefficient solver. This provides a 15-30% speed up in solution time for the primary heat exchanger.
        # cc_start = time()
        h_H, h_C, converged = self._convection_coeffs(2 * self.h_H - self.h_H_old,
                                                      2 * self.h_C - self.h_C_old,
                                                      T_H,
                                                      T_C)  # 34.411 sec
        if not converged:
            raise ValueError('Convection coefficient solution has not converged')
        _, _, R_prime_T = self._thermal_resistance(h_H, h_C)
        # cc_stop = time()
        # self.cc_time += cc_stop - cc_start

        # other_start = time()

        self.h_H_old = self.h_H
        self.h_C_old = self.h_C
        self.h_H = h_H
        self.h_C = h_C

        dqdx_tube = (T_H - T_C) / R_prime_T

        self.p_fld.update(CP.PT_INPUTS, self.P_primary, T_H)  # 3.336 sec
        i_H_in = self.p_fld.hmass()
        self.s_fld.update(CP.PT_INPUTS, self.P_max, T_C)  # 4.480 sec
        i_C_out = self.s_fld.hmass()

        i_H_out_update = i_H_in - dqdx_tube * self.Delta_x / self.m_dot_H_tube
        i_C_in_update = i_C_out - dqdx_tube * self.Delta_x / self.m_dot_C_tube

        self.p_fld.update(CP.HmassP_INPUTS, i_H_out_update, self.P_primary)  # 3.674 sec
        T_H_out_update = self.p_fld.T()

        self.s_fld.update(CP.HmassP_INPUTS, i_C_in_update, self.P_max)  # 4.615 sec
        T_C_in_update = self.s_fld.T()

        # other_stop = time()
        # self.other_time += other_stop - other_start

        return T_H_out_update, T_C_in_update

    def diff_step(self, t, x):
        i_H, i_C = x

        self.h_H, self.h_C, T_H, T_C, converged = self._convection_coeffs_enthalpy(self.h_H, self.h_C, i_H, i_C)
        if not converged:
            raise ValueError('Convection coefficient solution has not converged')
        _, _, R_prime_T = self._thermal_resistance(self.h_H, self.h_C)

        dqdx_tube = (T_H - T_C) / R_prime_T

        di_H = -dqdx_tube / self.m_dot_H_tube
        di_C = -dqdx_tube / self.m_dot_C_tube

        return np.array([di_H, di_C])

    def _convection_coeffs(self, h_H, h_C, T_H, T_C):
        """
        Iteratively solves for the convection coefficients of a heat exchanger.
        """
        self.p_fld.update(CP.PT_INPUTS, self.P_primary, T_H)
        Pr_H_avg = self.p_fld.Prandtl()
        k_H = self.p_fld.conductivity()

        for i in range(self.maxiter):
            R_prime_H, R_prime_C, R_prime_T = self._thermal_resistance(h_H, h_C)

            dqdx_tube = (T_H - T_C) / R_prime_T
            T_w_o = T_H - dqdx_tube * R_prime_H

            self.p_fld.update(CP.PT_INPUTS, self.P_primary, T_w_o)
            Pr_H_s = self.p_fld.Prandtl()
            mu_H_local = self.p_fld.viscosity()

            Re_H = self.m_dot_H * self.d_o / (self.A_primary_flow_min * mu_H_local)

            C_z = 0.27
            m_z = 0.63
            n_z = 0.36
            C_nuscale = 1
            Nu_H = C_nuscale * C_z * abs(Re_H) ** m_z * abs(Pr_H_avg) ** n_z * abs(Pr_H_avg / Pr_H_s) ** 0.25
            h_H_update = Nu_H * k_H / self.d_o

            T_w_i = dqdx_tube * R_prime_C + T_C

            self.s_fld.update(CP.PT_INPUTS, self.P_max, T_C)
            i_C = self.s_fld.hmass()

            h_C_update, k_C = self._cold_convection(i_C, T_w_i)

            diff_h = np.array([(h_H_update - h_H) / h_H_update,
                               (h_C_update - h_C) / h_C_update])

            h_H = h_H_update
            h_C = h_C_update

            if np.all(abs(diff_h) < self.rtol):
                return h_H, h_C, True

        return h_H, h_C, False

    def _convection_coeffs_enthalpy(self, h_H, h_C, i_H, i_C):
        """
        Iteratively solves for the convection coefficients of a heat exchanger.
        """
        self.p_fld.update(CP.HmassP_INPUTS, i_H, self.P_primary)
        Pr_H_avg = self.p_fld.Prandtl()
        k_H = self.p_fld.conductivity()
        T_H = self.p_fld.T()

        self.s_fld.update(CP.HmassP_INPUTS, i_C, self.P_max)
        T_C = self.s_fld.T()

        for i in range(self.maxiter):
            R_prime_H, R_prime_C, R_prime_T = self._thermal_resistance(h_H, h_C)

            dqdx_tube = (T_H - T_C) / R_prime_T
            T_w_o = T_H - dqdx_tube * R_prime_H

            self.p_fld.update(CP.PT_INPUTS, self.P_primary, T_w_o)
            Pr_H_s = self.p_fld.Prandtl()
            mu_H_local = self.p_fld.viscosity()

            Re_H = self.m_dot_H * self.d_o / (self.A_primary_flow_min * mu_H_local)

            C_z = 0.27
            m_z = 0.63
            n_z = 0.36
            C_nuscale = 1
            Nu_H = C_nuscale * C_z * abs(Re_H) ** m_z * abs(Pr_H_avg) ** n_z * abs(Pr_H_avg / Pr_H_s) ** 0.25
            h_H_update = Nu_H * k_H / self.d_o

            T_w_i = dqdx_tube * R_prime_C + T_C

            h_C_update, k_C = self._cold_convection(i_C, T_w_i)

            diff_h = np.array([(h_H_update - h_H) / h_H_update,
                               (h_C_update - h_C) / h_C_update])

            h_H = h_H_update
            h_C = h_C_update

            if np.all(abs(diff_h) < self.rtol):
                return h_H, h_C, T_H, T_C, True

        return h_H, h_C, T_H, T_C, False

    def _thermal_resistance(self, h_H, h_C):
        """
        Calculates the thermal resistance between sides of a heat exchanger. This includes resistances from convection,
        conduction through the wall, and wall fouling.
        """
        R_prime_H = 1 / (h_H * np.pi * self.d_o)
        R_prime_C = 1 / (h_C * np.pi * self.d_i)
        R_prime_wall = np.log(self.d_o / self.d_i) / (2 * np.pi * self.k_t)
        R_prime_H_foul = self.R_foul_dprime / (np.pi * self.d_o)
        R_prime_C_foul = self.R_foul_dprime / (np.pi * self.d_i)
        R_prime_T = R_prime_H + R_prime_H_foul + R_prime_wall + R_prime_C + R_prime_C_foul
        return R_prime_H, R_prime_C, R_prime_T

    def _cold_convection(self, i, T_w):
        """
        Calculates the heat transfer coefficient and heat conductivity for the cold side of the heat exchanger.
        """
        self.s_fld.update(CP.HmassP_INPUTS, i, self.P_max)
        print('Phase:', self.s_fld.phase())
        if self.s_fld.phase() == 6:
            print('     P =', self.s_fld.p() * 1e-6, '  MPa')
            print('     P_crit =', self.s_fld.p_critical() * 1e-6, '  MPa')
            print('     T =', self.s_fld.T(), '  K')
            print('     h =', self.s_fld.hmass(), '  J/kgK')

        if self.s_fld.phase() == 6:
            h, k = self._boiling_convection(i, self.s_fld.Q(), T_w)
        else:
            h, k = self._single_phase_convection(i)

        return h, k

    def _single_phase_convection(self, i):
        """
        Handles cases where the cold side of the heat exchanger section is a single phase.
        """
        k = self.s_fld.conductivity()
        Re = 4 * self.m_dot_C_tube / (np.pi * self.d_i * self.s_fld.viscosity())
        Nu = 0.023 * abs(Re) ** 0.8 * abs(self.s_fld.Prandtl()) ** 0.4
        h = Nu * k / self.d_i
        return h, k

    def _boiling_convection(self, i, x, T_w):
        """
        Handles cases where the cold side of the heat exchanger boils.

        NOTE: I haven't ever seen this being run. Likely any
        scenario in which this is necessary is failing out for one reason or another before getting here.
        """
        # self.s_fld.update(CP.HmassQ_INPUTS, i, x)
        T_sat = self.s_fld.T()
        P_sat = self.s_fld.p()
        TC = self.s_fld.T_critical()

        if T_w > TC:
            T_inter1 = TC - 0.1
            self.s_fld.update(CP.QT_INPUTS, 1, T_inter1)
            P_w = self.s_fld.p()
        else:
            self.s_fld.update(CP.QT_INPUTS, 1, T_w)
            P_w = self.s_fld.p()
            # self.s_fld.update(CP.QT_INPUTS, 0, T_w)
        # k_w = self.s_fld.conductivity()
        # mu_w = self.s_fld.viscosity()
        # nu_w = mu_w / self.s_fld.rhomass()
        # Cp_w = self.s_fld.cpmass()

        self.s_fld.update(CP.QT_INPUTS, 0, T_sat)
        Cp_I = self.s_fld.cpmass()
        k_I = self.s_fld.conductivity()
        mu_I = self.s_fld.viscosity()
        nu_I = mu_I / self.s_fld.rhomass()

        T_inter2 = T_sat - 0.1
        self.s_fld.update(CP.PT_INPUTS, self.P_max, T_inter2)
        # x_sig = self.s_fld.Q()
        # self.s_fld.update(CP.QT_INPUTS, x_sig, T_inter2)
        sigma_I = self.s_fld.surface_tension()
        self.s_fld.update(CP.QT_INPUTS, 0, T_sat)
        Pr_I = self.s_fld.Prandtl()
        self.s_fld.update(CP.QT_INPUTS, 1, T_sat)
        mu_v = self.s_fld.viscosity()
        nu_v = mu_v / self.s_fld.rhomass()
        # self.s_fld.update(CP.PT_INPUTS, self.P_max, T_sat)
        # x_I = self.s_fld.Q()
        self.s_fld.update(CP.PQ_INPUTS, self.P_max, 1)
        H_V = self.s_fld.hmass()
        self.s_fld.update(CP.PQ_INPUTS, self.P_max, 0)
        H_L = self.s_fld.hmass()
        i_fg = H_V - H_L  # latent heat of vaporization

        DELTAP = abs(P_w - P_sat)
        DELTAT = abs(T_w - T_sat)

        # Re_I = 4 * self.m_dot_C_tube / (np.pi * self.d_i * mu_I)
        # Re_v = 4 * self.m_dot_C_tube / (np.pi * self.d_i * mu_v)
        Re_mod = 4 * self.m_dot_C_tube * (1 - x) / (np.pi * self.d_i * mu_I)

        X_tt = ((1 - x) / x) ** 0.9 * (nu_I / nu_v) ** 0.5 * (mu_I / mu_v) ** 0.1

        if 1 / X_tt > 0.1:
            F = 2.35 * (1 / X_tt + 0.213) ** 0.736
        else:
            F = 1

        S = 1 / (1 + 2.53e-6 * (Re_mod * F ** 1.25) ** 1.17)
        Nu_conv = 0.023 * abs(Re_mod) ** 0.8 * abs(Pr_I) ** 0.4
        h_conv = Nu_conv * k_I / self.d_i

        h_FZ_numerator = 0.00122 * k_I ** 0.79 * Cp_I ** 0.45 * nu_v ** 0.24 * DELTAT ** 0.24 * DELTAP ** 0.75
        h_FZ_denom = max(1e-6, sigma_I ** 0.5 * mu_I ** 0.29 * abs(i_fg) ** 0.24 * nu_I ** 0.49)
        h_FZ = h_FZ_numerator / h_FZ_denom
        C_nuscale = 1
        h = (h_FZ * S + h_conv * F) * C_nuscale

        return h, k_I


def worker(P_max, m_dot_H, m_dot_C):
    hx = PrimaryHX(P_max, m_dot_H, m_dot_C)
    try:
        hx.solve()
        print('good')
        # return [P_max, m_dot_H, m_dot_C, hx.T_H[-1], hx.T_C[-1], hx.i_H[-1], hx.i_C[-1], hx.Q_dot_in, hx.DeltaT_SGmin]
        return [P_max, m_dot_H, m_dot_C, hx.T_H, hx.T_C, hx.i_H, hx.i_C, hx.Q_dot_in, hx.DeltaT_SGmin]
    except Exception as e:
        print('bad', e)
        # return [P_max, m_dot_H, m_dot_C, np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.nan, np.nan]
        return [P_max, m_dot_H, m_dot_C, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]


def generate_inputs(N):
    inputs = np.array([np.random.uniform(8.2e6, 9.2e6, N),  # P_max
                       np.random.uniform(400, 700, N),  # m_dot_H
                       np.random.uniform(50, 800, N)]).T  # m_dot_C
    return inputs


def generate_data(N, procs=8):
    inputs = generate_inputs(N)

    with mp.Pool(procs) as p:
        out = p.starmap(worker, inputs)

    df = pd.DataFrame(out, columns=['P_max', 'm_dot_H', 'm_dot_C', 'T_H', 'T_C', 'h_H', 'h_C', 'Q_dot_in', 'DeltaT_SGmin'])

    return df


def MAPE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true, axis=0) * 100


def fit_plots(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred):
    fig1, axes1 = plt.subplots(1, 2)
    axes1[0].scatter(X_train[:, 0] * 1e-6, y_train[:, 0], c='r')
    axes1[0].scatter(X_train[:, 0] * 1e-6, y_train[:, 1], c='b')
    axes1[0].scatter(X_train[:, 0] * 1e-6, y_train_pred[:, 0], c='orange')
    axes1[0].scatter(X_train[:, 0] * 1e-6, y_train_pred[:, 1], c='cyan')
    axes1[0].set_xlabel('P_max [Pa]')
    axes1[0].set_ylabel('Temperature [K]')

    axes1[1].scatter(X_test[:, 0] * 1e-6, y_test[:, 0], c='r')
    axes1[1].scatter(X_test[:, 0] * 1e-6, y_test[:, 1], c='b')
    axes1[1].scatter(X_test[:, 0] * 1e-6, y_test_pred[:, 0], c='orange')
    axes1[1].scatter(X_test[:, 0] * 1e-6, y_test_pred[:, 1], c='cyan')
    axes1[1].set_xlabel('P_max [Pa]')
    axes1[1].set_ylabel('Temperature [K]')

    fig2, axes2 = plt.subplots(1, 2)
    axes2[0].scatter(X_train[:, 1], y_train[:, 0], c='r')
    axes2[0].scatter(X_train[:, 1], y_train[:, 1], c='b')
    axes2[0].scatter(X_train[:, 1], y_train_pred[:, 0], c='orange')
    axes2[0].scatter(X_train[:, 1], y_train_pred[:, 1], c='cyan')
    axes2[0].set_xlabel('m_dot_H')
    axes2[0].set_ylabel('Temperature [K]')

    axes2[1].scatter(X_test[:, 1], y_test[:, 0], c='r')
    axes2[1].scatter(X_test[:, 1], y_test[:, 1], c='b')
    axes2[1].scatter(X_test[:, 1], y_test_pred[:, 0], c='orange')
    axes2[1].scatter(X_test[:, 1], y_test_pred[:, 1], c='cyan')
    axes2[1].set_xlabel('m_dot_H')
    axes2[1].set_ylabel('Temperature [K]')

    fig3, axes3 = plt.subplots(1, 2)
    axes3[0].scatter(X_train[:, 2], y_train[:, 0], c='r')
    axes3[0].scatter(X_train[:, 2], y_train[:, 1], c='b')
    axes3[0].scatter(X_train[:, 2], y_train_pred[:, 0], c='orange')
    axes3[0].scatter(X_train[:, 2], y_train_pred[:, 1], c='cyan')
    axes3[0].set_xlabel('m_dot_C')
    axes3[0].set_ylabel('Temperature [K]')

    axes3[1].scatter(X_test[:, 2], y_test[:, 0], c='r')
    axes3[1].scatter(X_test[:, 2], y_test[:, 1], c='b')
    axes3[1].scatter(X_test[:, 2], y_test_pred[:, 0], c='orange')
    axes3[1].scatter(X_test[:, 2], y_test_pred[:, 1], c='cyan')
    axes3[1].set_xlabel('m_dot_C')
    axes3[1].set_ylabel('Temperature [K]')

    plt.show()


def clean_df(df):
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if len(df.iloc[i]['T_H']) == 1:
            keep[i] = False
    return df.iloc[keep]


if __name__ == '__main__':
    worker(P_max=8.22e6, m_dot_H=500, m_dot_C=150)

    # -------------------
    #    Generate Data
    # -------------------
    # fpath = './data/hx_profs3.json'
    # print(pd.read_json(fpath)['T_H'])
    # exit()
    # data = generate_data(N=8, procs=8)
    # for i in range(200):
    #     try:
    #         fpath = './data/hx_profs3.json'
    #         prev_df = pd.read_json(fpath)
    #
    #         start = time()
    #         data = generate_data(N=512, procs=8)
    #         stop = time()
    #         print('Exec Time:', stop - start, 'sec')
    #
    #         df_final = prev_df.append(data, ignore_index=True)
    #         df_final = df_final.dropna()
    #         print('Points:', len(df_final))
    #         df_final.to_json(fpath)
    #     except Exception as e:
    #         print(e, ':(')
    # # data = data.dropna()
    # # data.to_json(fpath)

    # --------------------------
    #    Exploratory Plotting
    # --------------------------
    # fpath = './data/hx_profs.json'
    # df = pd.read_json(fpath)
    #
    # plt.figure(1)
    # plt.scatter(df['P_max'], np.stack(df['T_H'].to_numpy(), axis=0)[:, -1], c='r')
    # plt.scatter(df['P_max'], np.stack(df['T_C'].to_numpy(), axis=0)[:, -1], c='b')
    # plt.xlabel('P_max [Pa]')
    # plt.ylabel('Temperature [K]')
    #
    # plt.figure(2)
    # plt.scatter(df['m_dot_H'], np.stack(df['T_H'].to_numpy(), axis=0)[:, -1], c='r')
    # plt.scatter(df['m_dot_H'], np.stack(df['T_C'].to_numpy(), axis=0)[:, -1], c='b')
    # plt.xlabel('m_dot_H')
    # plt.ylabel('Temperature [K]')
    #
    # plt.figure(3)
    # plt.scatter(df['m_dot_C'], np.stack(df['T_H'].to_numpy(), axis=0)[:, -1], c='r')
    # plt.scatter(df['m_dot_C'], np.stack(df['T_C'].to_numpy(), axis=0)[:, -1], c='b')
    # plt.xlabel('m_dot_C')
    # plt.ylabel('Temperature [K]')
    #
    # plt.show()

    # -----------------------
    #    Simple Regression
    # -----------------------
    # fpath = './data/hx_profs.json'
    # df = pd.read_json(fpath)
    #
    # inputs = df[['P_max', 'm_dot_H', 'm_dot_C']].to_numpy()
    #
    # T_H_out = np.stack(df['T_H'].to_numpy(), axis=0)[:, -1]
    # T_C_in = np.stack(df['T_C'].to_numpy(), axis=0)[:, -1]
    # targets = np.array([T_H_out, T_C_in]).T
    #
    # X_train, X_test, y_train, y_test = train_test_split(inputs, targets, train_size=0.8, random_state=0, shuffle=True)
    #
    # reg = LinearRegression()
    # reg = RandomForestRegressor(n_estimators=1000)
    # reg.fit(X_train, y_train)
    # y_train_pred = reg.predict(X_train)
    # y_test_pred = reg.predict(X_test)
    # print('Train:', MAPE(y_train, reg.predict(X_train)))
    # print('Test: ', MAPE(y_test, reg.predict(X_test)))
    # fit_plots(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred)
