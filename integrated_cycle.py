import numpy as np
import scipy.optimize as opt
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

from src.lcoe import hx_cost, pump_cost, condenser_cost
from src.secondary_hx import SecondaryHX
from src.primary_cycle import PrimaryCycle
from src.primary_hx import PrimaryHX
from src.secondary_cycle import SecondaryCycle


class IntegratedCycle:
    """
    IntegratedCycle models the NuScale SMR with a secondary transcritical Rankine cycle with Methanol as its working
    fluid. The primary and secondary cycles are solved simultaneously.

    @author: Jacob Bryan
    @date: 13 August 2020
    """
    # ======================================
    # =   LCOE & Penalty Model Constants   =
    # ======================================
    CEPCI_2019 = 607.9248  # "Chemical Engineering Plant Cost Index in 2018"
    CEPCI_2016 = 541.7  # "Chemical Engineering Plant Cost Index in 2016"
    LCOE_2016 = 86  # [$/MWh] "Levelized cost of electricity of NuScale plant from: http://www.nuscalepower.com/smr-benefits/economical/operating-costs"
    LCOE_2019 = LCOE_2016 * CEPCI_2019 / CEPCI_2016
    OvernightEquipCost_frac = 0.08067  # "fraction of equipment costs of baseline cycle that contributes to the plant overnight costs (12 modules total): from baseline cycle model"
    BaselineCycleCost = 20841690  # [$]
    LCOECapital_frac = 0.5  # "fraction of LCOE that covers capital costs of the plant from: http://www.nuscalepower.com/smr-benefits/economical/construction-cost"
    Q_dot_in_baseline = 160e6
    T_pinch = 5  # [K] pinch point delta temperature in feedwater heaters
    T_normal = 10  # [K] value used to normalize hxer temperature constraints
    T_out_normal = 1  # [K] value used to normalize hxer temperature constraints
    x_normal = 0.1  # value used to normalize quality at mixing exit constraints
    x_expansion_limit = 0.87
    eta_Baseline = 0.3122

    def __init__(self, P_max, Pr_1, Pr_2, Pr_3, f1, f2, f3, p_fld='Water', s_fld='Methanol', p_backend='HEOS', s_backend='REFPROP'):
        """
        Parameters
        ----------
        P_max : float
            Maximum secondary cycle pressure [Pa]

        Pr_1 : float
            Pressure ratio of the first turbine. Ranges from 0 to 1.

        Pr_2 : float
            Pressure ratio of the second turbine. Ranges from 0 to 1.

        Pr_3 : float
            Pressure ratio of the third turbine. Ranges from 0 to 1.

        f1 : float
            Mass fraction of the first splitter. Ranges from 0 to 1.

        f2 : float
            Mass fraction of the second splitter. Ranges from 0 to 1.

        f3 : float
            Mass fraction of the third splitter. Ranges from 0 to 1.

        p_fld : string (Optional)
            Working fluid of the primary cycle. Default is 'Water'.

        s_fld : string (Optional)
            Working fluid of the secondary cycle. Default is 'Methanol'.

        p_backend : string (Optional)
            Specifies which backend CoolProp should use when calculating properties for the primary cycle fluid. Default
            is 'HEOS'. NOTE: Using the REFPROP backend for both primary and secondary fluids will cause the model to run
            significantly slower!

        s_backend : string (Optional)
            Specifies which backend CoolProp should use when calculating properties for the secondary cycle fluid. Default
            is 'REFPROP'. NOTE: Using the REFPROP backend for both primary and secondary fluids will cause the model to run
            significantly slower!
        """
        self.P = np.zeros(27)
        self.T = np.zeros(27)
        self.h = np.zeros(27)
        self.m_dot = np.zeros(27)
        self.w_t = np.zeros(4)
        self.w_p = np.zeros(4)

        self.P_max = P_max
        self.Pr_1 = Pr_1
        self.Pr_2 = Pr_2
        self.Pr_3 = Pr_3
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        if isinstance(p_fld, CP.AbstractState):
            self.p_fld = p_fld
        else:
            self.p_fld = CP.AbstractState(p_backend, p_fld)
        if isinstance(s_fld, CP.AbstractState):
            self.s_fld = s_fld
        else:
            self.s_fld = CP.AbstractState(s_backend, s_fld)

        self.eta_FL = np.nan
        self.DeltaT_SGmin = np.nan
        self.W_dot_t = np.nan
        self.W_dot_p = np.nan
        self.W_dot_net = np.nan
        self.Q_dot_in = np.nan

        self.hxers = None

        self.primary_cycle = None
        self.m_dot_H = 0
        self.secondary_cycle = None
        self.m_dot_C = 0
        self.primary_hx = None

        self.I_ERROR_FLAG = 0
        self.S_ERROR_FLAG = 0
        self.P_ERROR_FLAG = 0
        self.H_ERROR_FLAG = 0

    def solve(self, rtol=1e-6, maxiter=100):
        """
        Solves the integrated primary and secondary cycles.
        """
        # Solve secondary cycle
        self.secondary_cycle = SecondaryCycle(self.P_max, self.Pr_1, self.Pr_2, self.Pr_3, self.f1, self.f2, self.f3, self.s_fld)
        try:
            sec_cycle_converged = self.secondary_cycle.solve(use_fixed_pump_state_solver=True, rtol=rtol, maxiter=maxiter)
        except ValueError as ve:
            self.S_ERROR_FLAG = self.secondary_cycle.ERROR_FLAG
            raise ve

        if not sec_cycle_converged:
            self.S_ERROR_FLAG = -2
            raise ValueError('Secondary cycle solver did not converge!')
        else:
            self.S_ERROR_FLAG = self.secondary_cycle.ERROR_FLAG

        # Copy over secondary cycle states to parent class object for ease of access
        self.P = self.secondary_cycle.P
        self.T = self.secondary_cycle.T
        self.h = self.secondary_cycle.h
        self.m_dot = self.secondary_cycle.m_dot
        self.w_t = self.secondary_cycle.w_t
        self.w_p = self.secondary_cycle.w_p

        # Specific turbine, pump, and net power; first law efficiency of secondary cycle calculation
        w_dot_t = np.sum(self.w_t * self.m_dot[[1, 3, 5, 7]])
        w_dot_p = np.sum(self.w_p * self.m_dot[[10, 13, 16, 19]])
        w_dot_net = w_dot_t - w_dot_p
        if w_dot_net < 0:
            self.I_ERROR_FLAG = 1
            raise ValueError('w_dot_net was negative')
        if self.h[0] - self.h[20] < 0:
            self.I_ERROR_FLAG = 2
            raise ValueError('Primary heat exchanger enthalpy difference was negative')
        self.eta_FL = w_dot_net / (self.h[0] - self.h[20])

        self.primary_cycle = PrimaryCycle(self.p_fld)

        # Main loop initial guesses (default secant method)
        x0 = self.Q_dot_in_baseline  # baseline
        x1 = 400 * (self.h[0] - self.h[20])  # physics-informed guess based on secondary cycle solution and 'reasonable' m_dot_C value

        # Main loop (primary cycle and primary heat exchanger) solution
        res = opt.root_scalar(f=self._main_loop_objective, x0=x0, x1=x1, method='secant', rtol=rtol, maxiter=maxiter)

        if not res.converged:
            self.I_ERROR_FLAG = 3
            raise ValueError('Integrated cycle solve has not converged!')

        if not self.primary_hx.is_valid:
            self.I_ERROR_FLAG = 4
            raise ValueError('Primary HX has a ValueError somewhere!')

        # Final calculation of total power terms
        self.Q_dot_in = res.root
        self.W_dot_t = np.sum(self.w_t * self.m_dot_C * self.m_dot[[1, 3, 5, 7]])
        self.W_dot_p = np.sum(self.w_p * self.m_dot_C * self.m_dot[[10, 13, 16, 19]])
        self.W_dot_net = self.W_dot_t - self.W_dot_p

    def _main_loop_objective(self, Q_dot_in):
        """
        Objective function for main loop solver. Contains calls for the primary cycle and primary heat exchanger solvers.
        """
        # Solve primary and secondary cycle mass flow rates as function of Q_dot_in estimate
        self.m_dot_C = Q_dot_in / (self.h[0] - self.h[20])
        try:
            self.m_dot_H = self.primary_cycle.solve(Q_dot_in)
        except ValueError as ve:
            self.P_ERROR_FLAG = self.primary_cycle.ERROR_FLAG

        # Primary HX Solver
        self.primary_hx = PrimaryHX(self.P_max, self.m_dot_H, self.m_dot_C, self.p_fld, self.s_fld)
        try:
            self.primary_hx.solve()
        except ValueError as ve:
            self.H_ERROR_FLAG = self.primary_hx.ERROR_FLAG
        Q_dot_update = self.primary_hx.Q_dot_in
        self.DeltaT_SGmin = self.primary_hx.DeltaT_SGmin

        return Q_dot_update - Q_dot_in

    def LCOE(self):
        """
        Calculates levelized cost of energy. Returns LCOE and percent change in LCOE.
        """
        # Scale m_dot array (stored as mass fractions) to actual mass flow rates
        self.m_dot *= self.m_dot_C

        self.hxers = self.make_secondary_hxers()

        # Calculate cost terms
        C = np.zeros(9)
        C[0] = hx_cost(self.hxers[0].area, self.P[10] * 1e-6)
        C[1] = hx_cost(self.hxers[1].area, self.P[13] * 1e-6)
        C[2] = hx_cost(self.hxers[2].area, self.P[16] * 1e-6)
        C[3] = hx_cost(self.hxers[3].area, self.P[19] * 1e-6)
        C[4] = pump_cost(self.w_p[0] * self.m_dot[10] * 1e-3, self.P[10] * 1e-6)
        C[5] = pump_cost(self.w_p[1] * self.m_dot[13] * 1e-3, self.P[13] * 1e-6)
        C[6] = pump_cost(self.w_p[2] * self.m_dot[16] * 1e-3, self.P[16] * 1e-6)
        C[7] = pump_cost(self.w_p[3] * self.m_dot[19] * 1e-3, self.P[19] * 1e-6)
        C[8] = condenser_cost(self.W_dot_t, CEPCI=self.CEPCI_2019)  # turbines and condenser

        C_total = np.sum(C)
        # grouped_costs = np.array([C[:4].sum(), C[4:8].sum(), C[8]])
        # print(grouped_costs / C_total * 100)

        # Calculate LCOE of the cycle
        eff_frac = self.Q_dot_in * self.eta_FL / (self.Q_dot_in_baseline * self.eta_Baseline)
        cost_percent_change = (C_total - self.BaselineCycleCost) / self.BaselineCycleCost
        # LCOE_new = (self.LCOE_2019 + self.LCOE_2019 * self.LCOECapital_frac * self.OvernightEquipCost_frac * cost_percent_change) / eff_frac
        LCOE_new_2016 = (self.LCOE_2016 + self.LCOE_2016 * self.LCOECapital_frac * self.OvernightEquipCost_frac * cost_percent_change) / eff_frac
        # LCOE_percent_change = (LCOE_new - self.LCOE_2019) / self.LCOE_2019 * 100
        LCOE_percent_change_2016 = (LCOE_new_2016 - self.LCOE_2016) / self.LCOE_2016 * 100

        # return LCOE_new, LCOE_percent_change, C
        return LCOE_new_2016, LCOE_percent_change_2016

    def penalty(self):
        """
        Calculates and returns penalty term.
        """
        g = np.zeros(16)

        if self.hxers is None:
            # self.LCOE()  # HX objects are created in LCOE(); call this function if they haven't been made yet to avoid throwing an error. TODO: This isn't pretty...
            self.hxers = self.make_secondary_hxers()

        for i in range(4):  # g1 to g4 -- condenser and heat exchanger regen
            g[i] = self.hxers[i].penalty / self.T_normal

        for i, j in zip([4, 5, 6, 7], [10, 13, 16, 19]):  # g5 to g8 -- temperature change in cold side of heat exchangers
            g[i] = max(0, (self.T[j] - self.T[j + 1]) / self.T_out_normal)

        for i, j_pump_in in zip([8, 9, 10], [12, 15, 18]):  # g9 to g11 -- pump quality constraints
            g[i] = self.pump_penalty(j_pump_in)
            # g[i] = self.quality_penalty(self.h[j], self.P[j], P_crit) / self.x_normal

        for i, j in zip([11, 12, 13, 14], [1, 3, 5, 7]):  # g12 to g15 -- turbine expansion limits
            self.s_fld.update(CP.HmassP_INPUTS, self.h[j], self.P[j])
            g[i] = max(0, (self.x_expansion_limit - self._adjust_quality(self.s_fld.Q(), self.s_fld.phase()))) / self.x_normal

        g[15] = max(0, self.T_pinch - self.DeltaT_SGmin) / self.T_normal  # g16 -- primary heat exchanger pinch temp

        penalty = np.sum(g ** 2)  # Using element-wise square of individual terms to discourage higher individual penalty values

        return penalty
        # return penalty, g
        # return g, penalty

    def quality_penalty(self, h, P, P_crit):
        if P > P_crit:
            return 0
        else:
            self.s_fld.update(CP.HmassP_INPUTS, h, P)
            Q = self.s_fld.Q()
            phase = self.s_fld.phase()
            return self._adjust_quality(Q, phase)

    def pump_penalty(self, i_pump_in):
        if self.P[i_pump_in] > self.s_fld.p_critical():
            # No need to subcool if the fluid is supercritical
            pen = 0
        else:
            # Otherwise, we would like our fluid subcooled by 1C; a penalty will be incurred if subcooled by less than that
            self.s_fld.update(CP.PQ_INPUTS, self.P[i_pump_in], 0)
            T_sat = self.s_fld.T()
            T_lim = T_sat - 1
            T_actual = self.T[i_pump_in]
            pen = max(0, T_actual - T_lim)  # linear penalty term that has a maximum of 1
        return pen

    def make_secondary_hxers(self):
        # Required heat transfer rates for secondary cycle heat exchangers
        Q_dot_hx1 = self.m_dot[7] * (self.h[7] - self.h[8])
        Q_dot_hx2 = self.m_dot[25] * (self.h[25] - self.h[26])
        Q_dot_hx3 = self.m_dot[23] * (self.h[23] - self.h[24])
        Q_dot_hx4 = self.m_dot[21] * (self.h[21] - self.h[22])

        # Solve secondary HX geometries & profiles
        hxers = [SecondaryHX(self.s_fld, self.s_fld, self.h[7], self.h[10], self.m_dot[7], self.m_dot[10], Q_dot_hx1,
                             self.P[7], self.P[10]),
                 SecondaryHX(self.s_fld, self.s_fld, self.h[25], self.h[13], self.m_dot[25], self.m_dot[13], Q_dot_hx2,
                             self.P[25], self.P[13]),
                 SecondaryHX(self.s_fld, self.s_fld, self.h[23], self.h[16], self.m_dot[23], self.m_dot[16], Q_dot_hx3,
                             self.P[23], self.P[16]),
                 SecondaryHX(self.s_fld, self.s_fld, self.h[21], self.h[19], self.m_dot[21], self.m_dot[19], Q_dot_hx4,
                             self.P[21], self.P[19])]
        # try:
        #     hxers = [SecondaryHX(self.s_fld, self.s_fld, self.h[7], self.h[10], self.m_dot[7], self.m_dot[10], Q_dot_hx1, self.P[7], self.P[10]),
        #              SecondaryHX(self.s_fld, self.s_fld, self.h[25], self.h[13], self.m_dot[25], self.m_dot[13], Q_dot_hx2, self.P[25], self.P[13]),
        #              SecondaryHX(self.s_fld, self.s_fld, self.h[23], self.h[16], self.m_dot[23], self.m_dot[16], Q_dot_hx3, self.P[23], self.P[16]),
        #              SecondaryHX(self.s_fld, self.s_fld, self.h[21], self.h[19], self.m_dot[21], self.m_dot[19], Q_dot_hx4, self.P[21], self.P[19])]
        # except ZeroDivisionError:
        #     print(self.h[21], self.h[19])
        #     print('[{:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}, {:.8f}]'.format(self.P_max, self.Pr_1, self.Pr_2, self.Pr_3, self.f1, self.f2, self.f3))
        #     exit()

        return hxers

    @staticmethod
    def _adjust_quality(x, phase):
        """
        Returns an appropriate quality value given the input phase. The quality value returned by CoolProp for phases
        other than two-phase mixtures don't have an interpretable quality value. This returns a value appropriate for
        the phase.
        """
        if phase == 6:  # two-phase
            x_ret = x
        elif phase in [0, 3]:  # liquid or supercritical liquid
            x_ret = 0
        elif phase in [1, 2, 5]:  # supercritical, supercritical gas, gas
            x_ret = 1
        else:  # critical point, unknown, not imposed
            raise ValueError('Unsupported phase: {}'.format(phase))
        return x_ret

    def get_component_costs(self):
        """
        Calculates levelized cost of energy. Returns LCOE and percent change in LCOE.
        """
        # Scale m_dot array (stored as mass fractions) to actual mass flow rates
        self.m_dot *= self.m_dot_C

        self.hxers = self.make_secondary_hxers()

        # Calculate cost terms
        C = np.zeros(9)
        C[0] = hx_cost(self.hxers[0].area, self.P[10] * 1e-6)
        C[1] = hx_cost(self.hxers[1].area, self.P[13] * 1e-6)
        C[2] = hx_cost(self.hxers[2].area, self.P[16] * 1e-6)
        C[3] = hx_cost(self.hxers[3].area, self.P[19] * 1e-6)
        C[4] = pump_cost(self.w_p[0] * self.m_dot[10] * 1e-3, self.P[10] * 1e-6)
        C[5] = pump_cost(self.w_p[1] * self.m_dot[13] * 1e-3, self.P[13] * 1e-6)
        C[6] = pump_cost(self.w_p[2] * self.m_dot[16] * 1e-3, self.P[16] * 1e-6)
        C[7] = pump_cost(self.w_p[3] * self.m_dot[19] * 1e-3, self.P[19] * 1e-6)
        C[8] = condenser_cost(self.W_dot_t, CEPCI=self.CEPCI_2019)  # turbines and condenser

        return C
