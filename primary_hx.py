import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt


def euler(f, y0, z0, t_span, N):
    """
    Custom implementation of Euler's method for numerically solving an ODE. The z variable allows extra outputs to be
    stored and returned from the function besides the integrated trajectory. In this specific case, we're solving the
    ODE in terms of enthalpy, but we also wish to return and store temperatures from our derivative function.
    """
    t = np.linspace(t_span[0], t_span[1], N + 1)
    y_dims = (N + 1,) if (isinstance(y0, float) or len(y0) == 1) else (N + 1, len(y0))
    y = np.zeros(y_dims)
    z = np.zeros(y_dims)
    y[0] = y0
    z[0] = z0
    for i in range(N):
        h = t[i + 1] - t[i]
        fn = f(t[i], y[i])
        y[i + 1] = y[i] + fn[:2] * h
        z[i] = fn[2:]
    fn = f(t[-1], y[-1])
    z[-1] = fn[2:]
    return y.T, z.T


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
    d_c = 2 * r_c  # average tube column diameter
    k_t = 15.095  # [W/m-K] Thermal Resistance Analogy
    L = 24.4812312  # [m] "Average total tube length. Data from drawing #: NP12-01-A011-M-SA-2689-S02"
    P_primary = 12.8e6  # [Pa]

    # Some terms that remain constant; precalculated to save time in the long run
    R_foul_dprime = 5e-6  # [m^2-K/W] "Fouling factor for tube walls. Value selected to match temp profile to NuScale Analysis"
    R_prime_wall = np.log(d_o / d_i) / (2 * np.pi * k_t)
    R_prime_H_foul = R_foul_dprime / (np.pi * d_o)
    R_prime_C_foul = R_foul_dprime / (np.pi * d_i)
    R_prime_wall_tot = R_prime_wall + R_prime_H_foul + R_prime_C_foul

    C_z = 0.27
    m_z = 0.63
    n_z = 0.36
    C_nuscale = 1
    h_H_const1 = C_nuscale * C_z / d_o * (d_o / A_primary_flow_min) ** m_z

    def __init__(self, P_max, m_dot_H, m_dot_C, h_fld, c_fld, rtol=1e-6, maxiter=100):
        self.ERROR_FLAG = 0

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

        self.p_fld = h_fld
        self.s_fld = c_fld

        self.h_H = 9850
        self.h_H_old = 9850
        self.h_C = 7750
        self.h_C_old = 7750

        self.rtol = rtol
        self.maxiter = int(maxiter)

        self.h_H_const2 = m_dot_H ** self.m_z

        self.is_valid = False

    def solve(self, N=50):
        self.p_fld.update(CP.PT_INPUTS, self.P_primary, self.T_p)
        i_h_in = self.p_fld.hmass()
        self.s_fld.update(CP.PT_INPUTS, self.P_max, self.T_s)
        i_C_out = self.s_fld.hmass()

        try:
            (self.i_H, self.i_C), (self.T_H, self.T_C) = euler(self.enthalpy_diff, t_span=(0, self.L), y0=[i_h_in, i_C_out], z0=[self.T_p, self.T_s], N=N)
            self.Q_dot_in = self.m_dot_H * (i_h_in - self.i_H[-1])
            self.DeltaT_SGmin = np.min(self.T_H - self.T_C)
            self.x = np.linspace(0, self.L, len(self.T_H))
            self.is_valid = True
        except ValueError as ve:
            i_C_min = -380000
            self.Q_dot_in = self.m_dot_C * (i_C_out - i_C_min)
            self.DeltaT_SGmin = self.T_p - self.T_s
            self.is_valid = False

    def enthalpy_diff(self, t, x):
        i_H, i_C = x

        # Use linear approximation to precondition loop. Usually keeps the solver limited to just 2-3 iterations!
        h_H, self.h_C, T_H, T_C, dqdx_tube, converged = self._convection_coeffs(2 * self.h_H - self.h_H_old, i_H, i_C)
        self.h_H_old = self.h_H
        self.h_H = h_H

        if not converged:
            self.ERROR_FLAG = 1
            raise ValueError('Convection coefficient solution has not converged')

        di_H = -dqdx_tube / self.m_dot_H_tube
        di_C = -dqdx_tube / self.m_dot_C_tube

        return np.array([di_H, di_C, T_H, T_C])

    def _convection_coeffs(self, h_H, i_H, i_C):
        """
        Iteratively solves for the convection coefficients of a heat exchanger.
        """
        # Get hot fluid properties
        self.p_fld.update(CP.HmassP_INPUTS, i_H, self.P_primary)
        Pr_H_avg = self.p_fld.Prandtl()
        k_H = self.p_fld.conductivity()
        T_H = self.p_fld.T()
        h_H_const3 = Pr_H_avg ** (self.n_z + 0.25) * k_H

        # Get cold fluid properties
        self.s_fld.update(CP.HmassP_INPUTS, i_C, self.P_max)
        T_C = self.s_fld.T()
        Pr_C = self.s_fld.Prandtl()
        k_C = self.s_fld.conductivity()
        mu_C = self.s_fld.viscosity()
        Re_C = 4 * self.m_dot_C_tube / (np.pi * self.d_i * mu_C)
        Nu_C = 0.023 * abs(Re_C) ** 0.8 * abs(Pr_C) ** 0.4  # Dittus-Boelter correlation
        h_C = Nu_C * k_C / self.d_i
        R_prime_C = 1 / (h_C * np.pi * self.d_i)
        R_prime_other = R_prime_C + self.R_prime_wall_tot

        dqdx_tube = 0  # default value that shouldn't ever actually be used, but it keeps the linter from yelling at me...

        converged = False

        for i in range(self.maxiter):
            R_prime_H = 1 / (h_H * np.pi * self.d_o)
            R_prime_T = R_prime_H + R_prime_other

            dqdx_tube = (T_H - T_C) / R_prime_T
            T_w_o = T_H - dqdx_tube * R_prime_H

            self.p_fld.update(CP.PT_INPUTS, self.P_primary, T_w_o)  # 0.3 sec
            Pr_H_s = self.p_fld.Prandtl()
            mu_H_local = self.p_fld.viscosity()

            h_H_update = self.h_H_const1 * self.h_H_const2 * h_H_const3 * mu_H_local ** (-self.m_z) * Pr_H_s ** (-0.25)

            diff_h = abs(h_H_update - h_H) / h_H_update

            h_H = h_H_update

            if diff_h < self.rtol:
                converged = True
                break

        return h_H, h_C, T_H, T_C, dqdx_tube, converged

    def show_temp_profiles(self):
        self.x = np.linspace(0, self.L, len(self.T_H))
        plt.plot(self.x, self.T_H, label=r'$T_H$', color='red')
        plt.plot(self.x, self.T_C, label=r'$T_C$', color='blue')
        plt.legend()
        plt.xlabel('Axial Location (m)')
        plt.ylabel(r'Mean Fluid Temperature ($^\circ$C)')
        plt.show()

    def get_temp_profiles(self):
        self.x = np.linspace(0, self.L, len(self.T_H))
        return self.x, self.T_H, self.T_C
