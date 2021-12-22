import numpy as np
import CoolProp.CoolProp as CP


class SecondaryCycle:
    T_max = 301 + 273.15  # maximum cycle temperature
    T_pinch = 5  # pinch temperature for heat exchangers
    T_min = 35 + 273.15  # minimum cycle temperature
    eta_p = 0.75  # pump efficiency
    eta_t = 0.85  # turbine efficiency

    REGENERATOR_INDICES = [[25, 26, 11, 12, 13, 14],  # in form [i_h_in, i_h_out, i_u, i_mixed, i_c_in, i_c_out]
                           [23, 24, 14, 15, 16, 17],
                           [21, 22, 17, 18, 19, 20]]

    def __init__(self, P_max, Pr_1, Pr_2, Pr_3, f1, f2, f3, fld):
        self.fld = fld

        self.ERROR_FLAG = 0

        self.P_max = P_max
        self.Pr_1 = Pr_1
        self.Pr_2 = Pr_2
        self.Pr_3 = Pr_3
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.fld.update(CP.QT_INPUTS, 0, self.T_min)
        self.P_min = self.fld.p()

        self.P = self.init_P()
        self.m_dot = self.init_m_dot()
        self.T = np.zeros(27)
        self.T[0] = self.T_max
        self.h = np.zeros(27)
        self.w_t = np.zeros(4)
        self.w_p = np.zeros(4)

        self.m_dot_tot = 0

        # ===========================================================
        # ==== These states are defined by the design parameters ====
        # ===========================================================
        # State 1 properties are given by the max temperature and pressure of the src
        self.fld.update(CP.PT_INPUTS, self.P_max, self.T_max)
        self.h[0] = self.fld.hmass()

        # Solve state 2 properties
        self.w_t[0], self.h[1], self.T[1] = self._turbine(self.h[0], self.P[0], self.P[1])

        # Solve state 3 properties
        self.T[[2, 21]] = self.T[1]
        self.h[[2, 21]] = self.h[1]

        # Solve state 4 properties
        self.w_t[1], self.h[3], self.T[3] = self._turbine(self.h[2], self.P[2], self.P[3])

        # Solve state 5 properties
        self.T[[4, 23]] = self.T[3]
        self.h[[4, 23]] = self.h[3]

        # Solve state 6 properties
        self.w_t[2], self.h[5], self.T[5] = self._turbine(self.h[4], self.P[4], self.P[5])

        # Solve state 7 properties
        self.T[[6, 25]] = self.T[5]
        self.h[[6, 25]] = self.h[5]

        # Solve state 8 properties
        self.fld.update(CP.QT_INPUTS, 0, self.T_min + 1)
        # self.fld.update(CP.QT_INPUTS, 0, self.T_min)
        P_min = self.fld.p()
        self.P[[7, 8, 9]] = P_min
        # self.fld.update(CP.PT_INPUTS, self.P[6], self.T[6])
        self.w_t[3], self.h[7], self.T[7] = self._turbine(self.h[6], self.P[6], self.P[7])
        self.fld.update(CP.HmassP_INPUTS, self.h[7], self.P[7])

        # Solve state 10 properties
        # self.fld.update(CP.PQ_INPUTS, self.P[9], 0)
        # self.T[9] = self.fld.T()
        self.fld.update(CP.PT_INPUTS, self.P[9], self.T_min)
        self.T[9] = self.T_min
        self.h[9] = self.fld.hmass()
        s10 = self.fld.smass()

        # Solve state 11 properties
        self.w_p[0], self.h[10], self.T[10] = self._pump(self.h[9], s10, self.P[10])

        # Solve state 12 enthalpy
        self.h[11] = self._hot_pinch_temp(self.P[7], self.h[7], self.m_dot[7], self.P[10], self.h[10], self.m_dot[10])

    def solve(self, use_fixed_pump_state_solver=False, rtol=1e-6, maxiter=100):
        if self.P_max * self.Pr_1 * self.Pr_2 * self.Pr_3 <= self.P_min:
            # print('P_min:        ', P_min)
            # print('P_min_actual: ', P_max * Pr_1 * Pr_2 * Pr_3)
            self.ERROR_FLAG = 10
            raise ValueError('P_max * Pr_1 * Pr_2 * Pr_3 <= P_min')

        """ Solve the Hi-P, Mid-Hi-P, and Mid-Low-P regenerator states"""
        for i, inds in enumerate(self.REGENERATOR_INDICES):
            i_h_in, i_h_out, i_u, i_mixed, i_c_in, i_c_out = inds

            if use_fixed_pump_state_solver:
                self.fixed_pump_states_solver(i_u, i_h_in)
            else:
                self.pinch_point_pump_states_solver(i_u, i_h_in, pinch_temp=5, rtol=rtol, maxiter=maxiter)

        # Solve state 9 enthalpy
        self.h[8] = self.h[7] - self.h[11] + self.h[10]

        # Other temperatures
        for ind in range(27):
            self.fld.update(CP.HmassP_INPUTS, self.h[ind], self.P[ind])
            self.T[ind] = self.fld.T()

        return True

    def fixed_pump_states_solver(self, i_u, i_h_in):
        i_pump_in = i_u + 1
        i_c_in = i_u + 2
        i_c_out = i_u + 3
        i_h_out = i_h_in + 1
        i_wp = int((i_pump_in - 9) / 3)

        # Handling for points with P_C > P_H > P_critical -- TODO: we'll just shut it off for now...
        if self.P[i_pump_in] >= self.fld.p_critical():
            self.pinch_point_pump_states_solver(i_u, i_h_in, pinch_temp=1e8)  # super high pinch_temp forces the regenerator off (was 5)
            return

        """
        STEP 1:
        Calculate desired state values based on a pump inlet temperature T_pump_in = T_sat_liq - 1.
        """
        # Calculate the desired pump inlet state
        self.fld.update(CP.PQ_INPUTS, self.P[i_pump_in], 0)
        T_sat = self.fld.T()
        h_sat = self.fld.hmass()
        T_pump_in_desired = T_sat - 1
        self.fld.update(CP.PT_INPUTS, self.P[i_pump_in], T_pump_in_desired)
        h_pump_in_desired = self.fld.hmass()
        s_pump_in_desired = self.fld.smass()

        # Calculate desired hot side output temperature and enthalpy using Eq. 6
        h_h_out_desired = (h_pump_in_desired * self.m_dot[i_pump_in] - self.h[i_u] * self.m_dot[i_u]) / self.m_dot[i_h_out]

        """
        STEP 2:
        Calculate upper and lower bounds on the possible states. We'll do this specifically by calculating upper and
        lower bounds on the regenerator hot side outlet state temperature and enthalpy. We've already assumed that this
        state's pressure is *subcritical*, so a phase change is possible. Therefore, we need to:
            1. Calculate the temperature and enthalpy at the saturated vapor line at the hot side pressure.
            2. If T_h_out_desired < T_h_sat, we're going to be bounded in our heat transfer by ...
        """
        # Calculate state values at saturated vapor and saturated liquid lines
        self.fld.update(CP.PQ_INPUTS, self.P[i_h_out], 1)
        T_h_sat = self.fld.T()
        h_h_sat_vap = self.fld.hmass()
        self.fld.update(CP.PQ_INPUTS, self.P[i_h_out], 0)
        h_h_sat_liq = self.fld.hmass()

        # Calculate h_pump_in_max using Eq. 6 and h_h_out = h_h_in; h_u is fixed.
        h_h_out_max = self.h[i_h_in]

        self.fld.update(CP.PSmass_INPUTS, self.P[i_c_in], s_pump_in_desired)
        h_s_out = self.fld.hmass()
        self.fld.update(CP.PT_INPUTS, self.P[i_c_in], T_h_sat)
        h_c_pinch = self.fld.hmass()
        h_pump_max = (h_h_out_max * self.m_dot[i_h_out] + self.h[i_u] * self.m_dot[i_u]) / self.m_dot[i_pump_in]
        h_pump_min = h_s_out - self.eta_p * (h_c_pinch - (1 - self.m_dot[i_h_in] / self.m_dot[i_c_in]) * self.h[i_u] - self.m_dot[i_h_in] / self.m_dot[i_c_in] * h_h_sat_vap)
        # print('Pump Index:', i_pump_in)
        # print('Desired:', h_pump_in_desired)
        # print('Minimum:', h_pump_min)
        # print('Maximum:', h_pump_max)
        # print('Condensation:', h_h_sat_liq)
        # print('valid' if h_pump_in_desired >= min(h_pump_max, h_pump_min) else 'invalid')

        T_pinch = 0.1  # We'll use a T_pinch of 0.1 to make sure we have some bound on the size of the regenerator
        case = -1
        if h_h_out_desired >= h_h_out_max or self.m_dot[i_h_in] == 0:
            case = 0
            # print('**0010')
            # Hot side enthalpy cannot increase, so we disable this regenerator and outputs = inputs. This means that
            # that the fluid will be extra subcooled going into the pump, which is fine for operations.
            h_h_out = h_h_out_max
        elif h_h_sat_vap <= h_h_out_desired < h_h_out_max:
            # print('**0020')
            # Hot side outlet enthalpy is less than the inlet enthalpy but greater than the saturated vapor enthalpy
            # (i.e. we don't desire the fluid to enter the two-phase region), and the pinch point will be at the hot side
            # output/cold side inlet. Should only need to check small pinch temperature condition.
            self.fld.update(CP.HmassP_INPUTS, h_h_out_desired, self.P[i_h_out])
            T_h_out_desired = self.fld.T()

            # Traverse pump to calculate cold side inlet state
            w_p, h_c_in_desired, T_c_in_desired = self._pump(h_pump_in_desired, s_pump_in_desired, self.P[i_c_in])

            if T_h_out_desired > T_c_in_desired + T_pinch:
                case = 2
                T_h_out = T_h_out_desired
                h_h_out = h_h_out_desired
            else:
                case = 3
                # Need to iterate in this case. Raising T_h_out will increase T_pump_in and T_c_in for a fixed T_u
                converged = False
                T_h_out = T_c_in_desired + T_pinch
                h_h_out = h_h_out_desired
                for i in range(100):
                    self.fld.update(CP.PT_INPUTS, self.P[i_h_out], T_h_out)
                    h_h_out = self.fld.hmass()
                    h_pump_in = (self.m_dot[i_h_out] * h_h_out + self.m_dot[i_u] * self.h[i_u]) / self.m_dot[i_pump_in]
                    self.fld.update(CP.HmassP_INPUTS, h_pump_in, self.P[i_pump_in])
                    s_pump_in = self.fld.smass()
                    w_p, h_c_in, T_c_in = self._pump(h_pump_in, s_pump_in, self.P[i_c_in])

                    if T_h_out >= T_c_in + T_pinch:
                        converged = True
                        break

                    T_h_out = T_c_in + T_pinch

                if not converged:
                    raise ValueError('Gaseous regenerator iterative solver did not converge.')
        # elif h_h_sat_liq <= h_h_out_desired <= h_h_sat_vap:
        elif h_h_out_desired < h_h_sat_vap:
            case = 4
            # print('**0030')
            # Here, we want the hot side output to be in the two-phase region. We need to be careful to maintain a
            # positive pinch temperature here. We'll use a...

            # Traverse pump to calculate desired cold side inlet state
            w_p, h_c_in_desired, T_c_in_desired = self._pump(h_pump_in_desired, s_pump_in_desired, self.P[i_c_in])
            # h_c_out_desired = h_c_in_desired + self.m_dot[i_h_in] * (self.h[i_h_in] - self.h[i_h_out]) / self.m_dot[i_c_in]
            h_c_out_desired = h_c_in_desired + self.m_dot[i_h_in] * (self.h[i_h_in] - h_h_out_desired) / self.m_dot[i_c_in]

            # Calculate h_c_out_max based off of the hot side input state and the hot side saturation values.
            self.fld.update(CP.PT_INPUTS, self.P[i_c_in], T_h_sat)
            h_c_pinch = self.fld.hmass()
            h_c_out_max = h_c_pinch + self.m_dot[i_h_in] * (self.h[i_h_in] - h_h_sat_vap) / self.m_dot[i_c_out]

            # Check the desired h_c_out vs its maximum value. Note that this is equivalent to enforcing a minimum
            # condition on h_h_out, so if our h_h_out wasn't caught by the first conditional statement (h_h_out_desired
            # >= h_h_out_max), and it passes this condition, we'll be able to use our desired value.
            if h_c_out_desired >= h_c_out_max:
                # print('div\n')
                if i_pump_in == 12:
                    self.ERROR_FLAG = 1
                elif i_pump_in == 15:
                    self.ERROR_FLAG = 2
                elif i_pump_in == 18:
                    self.ERROR_FLAG = 3
                else:
                    self.ERROR_FLAG = -1
                raise ValueError("Two-phase regenerator solver: h_c_out_desired >= h_c_out_max (pump at state {}).".format(i_pump_in))

            h_h_out = h_h_out_desired
        else:  # h_h_out_desired < h_h_sat_liq:
            # print('**0040')
            # This would mean we'd want the hot side to condense completely. I'm not sure if this ever happens, but this
            # could complicate things because we'd have *two* potential pinch points: (1) where the hot side enters the
            # two-phase region, or (2) at the hot side outlet. For now... Let's just raise an error and see if this ever
            # happens before worrying about implementing this case.
            # raise ValueError('Seems like we want the hot side to condense completely... This case has not yet been implemented.')
            # TODO: We should implement a real solution here
            # h_h_out = h_h_sat_liq
            raise ValueError('Liquid iterative path not yet implemented.')

        self.h[i_h_out] = h_h_out
        self.fld.update(CP.HmassP_INPUTS, self.h[i_h_out], self.P[i_h_out])  # Having some problems with this line
        self.h[i_pump_in] = (self.h[i_h_out] * self.m_dot[i_h_out] + self.h[i_u] * self.m_dot[i_u]) / self.m_dot[i_pump_in]
        self.fld.update(CP.HmassP_INPUTS, self.h[i_pump_in], self.P[i_pump_in])
        if self.fld.Q() > 0:
            # print('div\n')
            if i_pump_in == 12:
                self.ERROR_FLAG = 1
            elif i_pump_in == 15:
                self.ERROR_FLAG = 2
            elif i_pump_in == 18:
                self.ERROR_FLAG = 3
            else:
                self.ERROR_FLAG = -1
            raise ValueError("Two-phase regenerator solver: liquid in pump (pump at state {}).".format(i_pump_in))

        self.w_p[i_wp], self.h[i_c_in], self.T[i_c_in] = self._pump(self.fld.hmass(), self.fld.smass(), self.P[i_c_in])
        self.h[i_c_out] = self.h[i_c_in] + self.m_dot[i_h_in] * (self.h[i_h_in] - self.h[i_h_out]) / self.m_dot[i_c_in]
        # if h_pump_in_desired >= h_pump_max:
        #     print(h_pump_max)
        # elif h_pump_min <= h_pump_in_desired < h_pump_max:
        #     print(h_pump_in_desired)
        # else:
        #     print('here', h_pump_min)
        # print(self.h[i_pump_in])
        # print('conv\n')
        print('Case:', case)

    def pinch_point_pump_states_solver(self, i_u, i_h_in, pinch_temp, rtol=1e-6, maxiter=100):
        i_pump_in = i_u + 1
        i_c_in = i_u + 2
        i_c_out = i_u + 3
        i_h_out = i_h_in + 1
        i_wp = int((i_pump_in - 9) / 3)

        h = (self.m_dot[i_u] * self.h[i_u] + self.m_dot[i_h_in] * self.h[i_h_in]) / self.m_dot[i_pump_in]  # this is the upper bound for h

        # if h < self.h[i_u]:
        #     self.h[i_h_out] = self.h[i_h_in]
        #     self.h[i_pump_in] = h
        #     self.fld.update(CP.HmassP_INPUTS, h, self.P[i_pump_in])
        #     w_p, h_c_in, T_c_in = self._pump(h, self.fld.smass(), self.P[i_c_in])
        #     return True

        regen_converged = False

        stored_h = []

        for j in range(maxiter):
            # pump
            stored_h.append(h)
            self.fld.update(CP.HmassP_INPUTS, h, self.P[i_pump_in])
            # self.w_p[i_wp], self.h[i_c_in], self.T[i_c_in] = self._pump(h, self.fld.smass(), self.P[i_c_in])
            w_p, h_c_in, T_c_in = self._pump(h, self.fld.smass(), self.P[i_c_in])
            # regenerator
            h_c_out = self._hot_pinch_temp(self.P[i_h_in], self.h[i_h_in], self.m_dot[i_h_in],
                                           self.P[i_c_in], h_c_in, self.m_dot[i_c_in], pinch_temp=pinch_temp)
            # Eq. 3 to get enthalpy of hot side output
            h_h_out = self.h[i_h_in] - self.m_dot[i_c_in] * (h_c_out - h_c_in) / self.m_dot[i_h_out]
            # Eq. 6 to get enthalpy after mixing/before pump
            h_pump_in = (self.m_dot[i_u] * self.h[i_u] + self.m_dot[i_h_out] * h_h_out) / self.m_dot[i_pump_in]

            # Calculate error. Update h guess with newly-calculated h[i_pump_in]
            if h_pump_in != 0 and abs((h_pump_in - h) / h_pump_in) < rtol:
                regen_converged = True
                self.h[i_pump_in] = h_pump_in
                self.h[i_h_out] = h_h_out
                self.h[i_c_in] = h_c_in
                self.h[i_c_out] = h_c_out
                self.w_p[i_wp] = w_p
                break

            h = h_pump_in

        return regen_converged

    def init_P(self):
        P = np.zeros(27)
        P[[0, 19, 20]] = 1
        P[[1, 2, 16, 17, 18, 21, 22]] = self.Pr_1
        P[[3, 4, 13, 14, 15, 23, 24]] = self.Pr_1 * self.Pr_2
        P[[5, 6, 10, 11, 12, 25, 26]] = self.Pr_1 * self.Pr_2 * self.Pr_3
        P *= self.P_max
        return P

    def init_m_dot(self):
        m_dot = np.ones(27)
        m_dot[[2, 3, 15, 16, 17]] = 1 - self.f1
        m_dot[[4, 5, 12, 13, 14]] = (1 - self.f1) * (1 - self.f2)
        m_dot[[6, 7, 8, 9, 10, 11]] = (1 - self.f1) * (1 - self.f2) * (1 - self.f3)
        m_dot[[21, 22]] = self.f1
        m_dot[[23, 24]] = (1 - self.f1) * self.f2
        m_dot[[25, 26]] = (1 - self.f1) * (1 - self.f2) * self.f3
        return m_dot

    def _turbine(self, h_in, P_in, P_out):
        """
        - Gets inlet quality using (h, P) calculation
        - Calculates outlet enthalpy for isentropic expansion
        - Adjusts turbine efficiency based on quality (average of inlet and isentropic outlet)
        - Calculates actual
        """
        self.fld.update(CP.HmassP_INPUTS, h_in, P_in)
        x1 = self.fld.Q()
        phase1 = self.fld.phase()
        x_in = self._adjust_quality(x1, phase1)
        s_in = self.fld.smass()
        s_s_out = s_in
        self.fld.update(CP.PSmass_INPUTS, P_out, s_s_out)
        h_s_out = self.fld.hmass()
        x2 = self.fld.Q()
        phase2 = self.fld.phase()
        x_out_s = self._adjust_quality(x2, phase2)
        x_a = 0.5 * (x_in + x_out_s)
        eta_a = self.eta_t * (1 - 0.72 * (1 - x_a))
        W_dot_t_s_m_dot = h_in - h_s_out
        W_dot_t_m_dot = W_dot_t_s_m_dot * eta_a
        h_out = h_in - W_dot_t_m_dot
        self.fld.update(CP.HmassP_INPUTS, h_out, P_out)
        T_out = self.fld.T()
        return W_dot_t_m_dot, h_out, T_out

    def _pump(self, h_in, s_in, P_out):
        self.fld.update(CP.PSmass_INPUTS, P_out, s_in)  # output with no entropy change
        h_s_out = self.fld.hmass()
        W_dot_s_p_m_dot = h_s_out - h_in  # pump work needed w/ no entropy change
        W_dot_p_m_dot = W_dot_s_p_m_dot / self.eta_p  # actual pump work needed d/t efficiency losses
        h_out = h_in + W_dot_p_m_dot  # output enthalpy
        self.fld.update(CP.HmassP_INPUTS, h_out, P_out)  # actual output state
        T_out = self.fld.T()
        return W_dot_p_m_dot, h_out, T_out

    def _adjust_quality(self, x, phase):
        """
        Returns an appropriate quality value given the input phase. The quality value returned by CoolProp for phases
        other than two-phase mixtures don't have an interpretable quality value. This returns a value appropriate for
        the phase.
        """
        if phase == 6:  # two-phase
            x_ret = x
        elif phase == 0:  # liquid
            x_ret = 0
        elif phase in [1, 2, 3, 5]:  # supercritical, supercritical gas, supercritical liquid, gas
            x_ret = 1
        else:
            raise ValueError('Unsupported phase: {}'.format(phase))
        return x_ret

    def _hot_pinch_temp(self, P_h, h_h_in, m_dot_h, P_c, h_c_in, m_dot_c, pinch_temp=None):
        """
        This is a bunch of conditions for what's essentially just

        h_c_out = h_c_pinch + delta_h * m_dot_H / m_dot_C  (Eq. 3)

        This is a function only of the input hot and cold states. Note that we know the ratio of mass flow rates from
        the design parameters; we don't need to know the actual mass flow rates!
        """
        T_pinch = self.T_pinch if pinch_temp is None else pinch_temp

        P_crit = self.fld.p_critical()
        self.fld.update(CP.HmassP_INPUTS, h_h_in, P_h)
        T_h_in = self.fld.T()
        self.fld.update(CP.HmassP_INPUTS, h_c_in, P_c)
        T_c_in = self.fld.T()

        if T_h_in - T_c_in > T_pinch:  # difference in input temperatures is greater than pinch temp, so we'll have some heat transfer
            if P_c < P_crit and P_h < P_crit:  # below supercritical
                self.fld.update(CP.PQ_INPUTS, P_h, 1)
                T_h_sat, h_h_sat = self.fld.T(), self.fld.hmass()
                self.fld.update(CP.PQ_INPUTS, P_c, 1)
                T_c_sat = self.fld.T()
                if T_h_sat - T_pinch == T_c_sat:  # difference between hot and cold sat temps equals the pinch temp; MATLAB comment: don't throw an error if saturated
                    # This should never be true!
                    # Requires hot side to be a gas or no heat transfer occurs
                    # Puts pinch point at Q_c=0
                    # Puts hot side at saturation line
                    self.fld.update(CP.PQ_INPUTS, P_c, 0)
                    h_c_pinch = self.fld.hmass()
                    delta_h = max(0, h_h_in - h_h_sat)  # brings hot side to saturation line; if hot side inlet is at h_sat or lower, no heat transfer occurs
                elif T_h_sat - T_pinch > T_c_in:  # MATLAB comment: condensation can occur
                    # Requires hot side to be a gas or no heat transfer occurs
                    # Puts pinch point at (P_c, T_h_sat-T_pinch)
                    #     - this would allow hot side to potentially go all the way to being a liquid
                    #     - also note that this cannot give a result in the two-phase region
                    # Remember that cold side will never be able to get to two-phase region if hot side is condensing
                    # Hot side gets put at saturated vapor
                    self.fld.update(CP.PT_INPUTS, P_c, T_h_sat - T_pinch)
                    h_c_pinch = self.fld.hmass()
                    delta_h = max(0, h_h_in - h_h_sat)  # brings hot side to saturation line
                else:  # MATLAB comment: condensation cannot occur
                    # Puts pinch point at cold side inlet
                    # Hot side can't condense, must stay a gas/supercritical gas
                    # Hot side temperature brought down to T_c_in+T_pinch, staying a gas/supercritical gas
                    h_c_pinch = h_c_in
                    self.fld.update(CP.PT_INPUTS, P_h, T_c_in + T_pinch)
                    h_superheat = self.fld.hmass()
                    delta_h = max(0, h_h_in - h_superheat)
            elif P_h < P_crit:  # hot side not supercritical
                self.fld.update(CP.PQ_INPUTS, P_h, 1)
                T_h_sat, h_h_sat = self.fld.T(), self.fld.hmass()
                if T_h_sat - T_c_in > T_pinch:  # temp difference greater than pinch temp
                    self.fld.update(CP.PT_INPUTS, P_c, T_h_sat - T_pinch)
                    h_c_pinch = self.fld.hmass()
                    delta_h = max(0, h_h_in - h_h_sat)
                else:
                    h_c_pinch = h_c_in
                    self.fld.update(CP.PT_INPUTS, P_h, T_c_in + T_pinch)
                    h_superheat = self.fld.hmass()
                    delta_h = max(0, h_h_in - h_superheat)
            else:  # both are supercritical
                # Approximates pinch temperature stuff by going from just above the critical pressure to just below it?
                # This approach was likely originally implemented when P_max was some fixed value just above the critical point.
                P_f1 = P_crit - 1  # TODO: MATLAB script also subtracts 1, but theirs is in [MPa], not [Pa] as it is here!
                self.fld.update(CP.PQ_INPUTS, P_f1, 1)
                h_h_pinch = self.fld.hmass()
                self.fld.update(CP.HmassP_INPUTS, h_h_pinch, P_c)
                T_h_pinch = self.fld.T()
                if T_h_pinch - T_c_in > T_pinch:
                    self.fld.update(CP.PT_INPUTS, P_c, T_h_pinch - T_pinch)
                    h_c_pinch = self.fld.hmass()
                    delta_h = max(0, h_h_in - h_h_pinch)
                else:
                    h_c_pinch = h_c_in
                    self.fld.update(CP.PT_INPUTS, P_h, T_c_in + T_pinch)
                    h_superheat = self.fld.hmass()
                    delta_h = max(0, h_h_in - h_superheat)
            Q_dot = m_dot_h * delta_h
        else:  # inputs are within the pinch temperature, so no heat transfer here
            h_c_pinch = h_c_in
            Q_dot = 0

        return h_c_pinch + Q_dot / m_dot_c  # h_c_pinch + delta_h * (m_dot_h / m_dot_c)

    def is_valid_solution(self):
        # Run back through the regenerators and make sure everything converges in 1 iteration
        mass_fracs = [self.f3, self.f2, self.f1]
        for i, inds in enumerate(self.REGENERATOR_INDICES):
            i_h_in, i_h_out, i_u, i_mixed, i_c_in, i_c_out = inds

            # Go across the pump
            self.fld.update(CP.HmassP_INPUTS, self.h[i_mixed], self.P[i_mixed])
            w_p, h_c_in, T_c_in = self._pump(self.h[i_mixed], self.fld.smass(), self.P[i_c_in])

            # Solve the regenerator according to pinch temp analysis
            h_c_out = self._hot_pinch_temp(self.P[i_h_in], self.h[i_h_in], self.m_dot[i_h_in],
                                           self.P[i_c_in], h_c_in, self.m_dot[i_c_in])

            # Solve for h_h_out with Eq. 3
            h_h_out = self.h[i_h_in] - (h_c_out - h_c_in) / mass_fracs[i]

            # Start with i_u (known) and i_h_out (solved for previously) to get i_mixed via Eq. 6
            h_mixed = (self.m_dot[i_u] * self.h[i_u] + self.m_dot[i_h_out] * h_h_out) / self.m_dot[i_mixed]

            if abs((self.h[i_mixed] - h_mixed) / h_mixed) > self.rtol:
                return False

        return True

    def get_hx_states(self):
        HX_INDICES = [[7, 10], [25, 13], [23, 16], [21, 19]]

        for i, (i_h_in, i_c_in) in enumerate(HX_INDICES):
            print('HX', i + 1)
            print('Hot:  ', CP.PhaseSI('H', self.h[i_h_in], 'P', self.P[i_h_in], 'Methanol'))
            print('Cold: ', CP.PhaseSI('H', self.h[i_c_in], 'P', self.P[i_c_in], 'Methanol'))
            print(self.P[i_c_in] > self.P[i_h_in], self.T[i_h_in] - self.T[i_c_in])

    def check_pump_states(self):
        PUMP_INLET_INDICES = [12, 15, 18]
        # H_IN_INDICES = [25, 23, 21]
        PHASES = {0: 'liquid',
                  1: 'supercritical',
                  2: 'supercritical gas',
                  3: 'supercritical liquid',
                  4: 'critical point',
                  5: 'gas',
                  6: 'two-phase'}

        for i, i_in in enumerate(PUMP_INLET_INDICES):
            self.fld.update(CP.HmassP_INPUTS, self.h[i_in], self.P[i_in])
            if self.fld.phase() not in [0, 1, 3]:  # fluid must be liquid or supercritical or an error gets thrown
                if self.fld.phase() == 6 and self.fld.Q() == 0:
                    continue
                # raise ValueError('Pump inlet (State {}) has phase "{}".'.format(i_in + 1, PHASES.get(self.fld.phase())))
                # print('Pump inlet (State {}) has phase "{}".'.format(i_in + 1, PHASES.get(self.fld.phase())), self.P[i_in] > self.fld.p_critical())
                return False

        return True
