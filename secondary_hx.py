import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt


class SecondaryHX:
    
    N_area = 500
    T_pinch = 5
    
    def __init__(self, h_fld, c_fld, h_hot_in, h_cold_in, m_dot_hot, m_dot_cold, Q_dot, P_hot, P_cold):
        self.h_fld = h_fld
        self.c_fld = c_fld
        self.P_hot = P_hot
        self.P_cold = P_cold

        # Hot-side states are indexed from outlet to inlet (reverse order). This is so that the indexed locations for the hot and cold sides correspond physically.
        self.h_h = np.zeros(self.N_area + 1)
        self.T_h = np.zeros(self.N_area + 1)
        self.phase_h = np.zeros(self.N_area + 1)

        # Cold-side states are indexed from inlet to outlet
        self.h_c = np.zeros(self.N_area + 1)
        self.T_c = np.zeros(self.N_area + 1)
        self.phase_c = np.zeros(self.N_area + 1)

        self.areas = np.zeros(self.N_area)

        self.Q_dot = Q_dot

        if Q_dot < 1e-6:
            self.area = 0
            self.penalty = 0
        else:
            self.area = self._hx_area(h_hot_in, h_cold_in, m_dot_hot, m_dot_cold, Q_dot, P_hot, P_cold)
            self.penalty = self._check_hx_valid(self.T_h, self.T_c)
            # plt.plot(self.T_c, label='T_C')
            # plt.plot(self.T_h, label='T_H')
            # plt.legend()
            # plt.show()

    def _hx_area(self, h_hot_in, h_cold_in, m_dot_hot, m_dot_cold, Q_dot, P_hot, P_cold):
        # Solve states for hot and cold sides at their inputs
        self.h_h[-1] = h_hot_in
        self.h_fld.update(CP.HmassP_INPUTS, h_hot_in, P_hot)
        self.T_h[-1] = self.h_fld.T()
        self.phase_h[-1] = self.h_fld.phase()

        self.h_c[0] = h_cold_in
        self.c_fld.update(CP.HmassP_INPUTS, h_cold_in, P_cold)
        self.T_c[0] = self.c_fld.T()
        self.phase_c[0] = self.c_fld.phase()

        # Find specific heat transferred on each side and the corresponding enthalpy change
        q_h = Q_dot / m_dot_hot
        q_c = Q_dot / m_dot_cold
        deltah_h = q_h / self.N_area
        deltah_c = q_c / self.N_area

        for j in range(1, self.N_area + 1):
            self.h_h[-j - 1] = self.h_h[-j] - deltah_h
            self.h_fld.update(CP.HmassP_INPUTS, self.h_h[-j - 1], P_hot)
            self.T_h[-j - 1] = self.h_fld.T()
            self.phase_h[-j - 1] = self.h_fld.phase()

            self.h_c[j] = self.h_c[j - 1] + deltah_c
            self.c_fld.update(CP.HmassP_INPUTS, self.h_c[j], P_cold)
            self.T_c[j] = self.c_fld.T()
            self.phase_c[j] = self.c_fld.phase()

        for k in range(self.N_area):
            U_o = self._get_u(self.phase_h[k], self.phase_c[k], P_hot, P_cold)
            deltaT_lm = self._log_mean_temp(self.T_h[k + 1], self.T_h[k], self.T_c[k], self.T_c[k + 1])
            deltaQ_dot = m_dot_hot * (self.h_h[k + 1] - self.h_h[k])
            # if np.isnan(deltaT_lm):
            #     print(deltaT_lm, self.T_h[k + 1], self.T_h[k], self.T_c[k], self.T_c[k + 1])
            if U_o * deltaT_lm != 0:
                self.areas[k] = deltaQ_dot / (U_o * deltaT_lm)

        area = np.sum(self.areas)

        return area

    def _get_u(self, phase_h, phase_c, P_h, P_c):
        d_o = 0.033401  # [m] tube outside diameter
        d_i = 0.0266446  # [m] tube inside diameter
        k_w = 16.3  # [W/m-K] approximate metal conductivity at average temperature

        h_h = self._conv_coeff(phase_h, 0, P_h)
        h_c = self._conv_coeff(phase_c, 1, P_c)

        h_o, h_i = (h_c, h_h) if h_h > h_c else (h_h, h_c)

        u = 1 / (1 / h_o + d_o * np.log(d_o / d_i) / (2 * k_w) + d_o / (d_i * h_i))

        return u

    @staticmethod
    def _check_hx_valid(T_h, T_c):
        deltaT = T_h - T_c
        deltaT_min = min(deltaT)
        if deltaT_min <= 0:
            return -deltaT_min + 10
        else:
            return 0
    
    @staticmethod
    def _conv_coeff(phase, side, P):
        p = P * 1e-6  # convert to [MPa] so I don't have to figure out what's going on with units with these interpolations...
        if phase in [1, 2, 3, 4,
                     5]:  # is supercritical/saturated vapor/superheated gas ("sensible gas"); only leaves twophase for other cases
            if p < 0.2:
                h = 0.1
            elif p < 1:
                h = 0.1 * (1 - (p - 0.2) / 0.8) + 0.325 * (p - 0.2) / 0.8
            elif p < 10:
                h = 0.325 * (1 - (p - 1) / 9) + 0.625 * (p - 1) / 9
            else:
                h = 0.625
        elif side == 0:  # hot side ("condensing")
            if p < 0.01:
                h = 1.75
            elif p < 0.1:
                h = 1.75 * (1 - (p - 0.01) / 0.09) + 3 * (p - 0.01) / 0.09
            elif p < 1:
                h = 3 * (1 - (p - 0.1) / 0.9) + 3.5 * (p - 0.1) / 9
            else:
                h = 3.5
        else:  # cold side ("boiling")
            h = 1.75

        return h * 1e3  # convert to [W/m^2-K]

    # @staticmethod
    def _log_mean_temp(self, Thi, Tho, Tci, Tco):
        dT1 = Thi - Tco
        dT2 = Tho - Tci
        # if dT1 / dT2 <= 0:
        #     plt.plot(self.T_c, label='T_C')
        #     plt.plot(self.T_h, label='T_H')
        #     plt.legend()
        #     plt.show()
        #     raise ZeroDivisionError
        return (dT1 - dT2) / np.log(dT1 / dT2)
