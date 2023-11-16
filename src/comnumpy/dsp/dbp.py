from channels.optical import Chromatic_Dispersion, Kerr_NonLinearity, EDFA
from channels.ssfm import Base_Fiber_Link
from .utils import Amplificator

class Base_DBP(Base_Fiber_Link):

    direction = -1

    def __init__(self, N_span, StPS, L_span, gamma=1.3*1e-3, lamb=1.55 * 10**-6, c=3*(10**8), alpha_dB= 0.2*1e-3, F_s=1, step_structure="symmetric", step_type="linear", step_log_factor=0.4, step_scheme=1, include_edfa=False, name="DBP"):
        self.N_span = N_span
        self.StPS = StPS
        self.L_span = L_span
        self.gamma = gamma 
        self.lamb = lamb
        self.c = c
        self.alpha_dB = alpha_dB
        self.F_s = F_s 
        self.step_structure = step_structure
        self.step_type = step_type
        self.step_log_factor = step_log_factor
        self.step_scheme = step_scheme # 0: no gain, 1: gain in CD, 2: gain in nl
        self.name = name 
        self.include_edfa = include_edfa

        self.prepare()

    def prepare(self):
        """
        store the list of module
        """
        edfa = EDFA(self.alpha_dB, L_span = self.L_span)
        self.gain = 1/edfa.gain  # compensate for EDFA gain
        self.module_list = self.get_module_list()

    def get_linear_module(self, dz):
        """
        return a linear layer
        """
        return Chromatic_Dispersion(dz, alpha_dB=self.alpha_dB, lamb=self.lamb, c=self.c, F_s=self.F_s, direction=self.direction)

    def get_non_linear_module(self, dz):
        """
        return a non-linear layer
        """
        return self.Kerr_NonLinearity(dz, self.gamma, alpha_dB=0, direction = self.direction)


    def get_pre_span_module_list(self):
        """
        return the module list before a span
        """
        amplificator = Amplificator(self.gain)
        return [amplificator]

    def get_step_module_list(self, dz):
        """
        return the module list for a single step  of length dz
        """
        nl = self.get_non_linear_model(dz)
        cd = self.get_Chromatic_Dispersion(dz)
        return [cd, nl]

    def get_post_span_module_list(self):
        """
        return the module list after a span
        """
        return []


