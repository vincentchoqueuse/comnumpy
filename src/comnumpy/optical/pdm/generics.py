from dataclasses import dataclass, field
from typing import Any, List, Generator, Optional, Dict, Union, Callable
import copy
import numpy as np
from comnumpy.core.generics import Sequential, Processor
import matplotlib.pyplot as plt


@dataclass
class PDMWrapper(Sequential):
    """A PDM Wrapper encapsulates a SISO processor object to apply for each the input stream."""

    module_list: Any
    debug: bool = False
    name: str = "PDM_Wrapper"
    callbacks: Optional[Dict[Union[str, int], Callable]] = field(default_factory=dict)
    N_t: int = field(default=2, init=False)
    
    def __post_init__(self):
        """Initialize PDM-specific attributes."""
        # N_t is already set to 2 by default via field()

    def get_module_by_name(self, module_name: str) -> Any:
        """Retrieve a module by its name."""
        if hasattr(self.module_list, 'name'):
            if self.module_list.name == module_name:
                return self.module_list
        raise AttributeError(f"Module '{module_name}' not found in class {self.__class__.__name__}.")

    def get_module(self, index: int) -> Any:
        module = copy.deepcopy(self.module_list)
        if isinstance(module, Sequential):
            for submodule in module.module_list:
                submodule.name = f"{submodule.name}_pol{index}"
        return module

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y_list = []
        if X.ndim == 1:
            X = X.reshape((2, -1))
        
        if isinstance(self.module_list, Processor):
            sequential = self.module_list
            for index in range(self.N_t):
                x_temp = X[index, :]
                y = sequential(x_temp)
                Y_list.append(y)
        elif isinstance(self.module_list, Sequential):
            for index in range(self.N_t):
                sequential = self.get_module(index)
                x_temp = X[index, :]
                y = sequential(x_temp)
                Y_list.append(y)
        Y = np.array(Y_list)
        return Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)




@dataclass
class ChannelWrapper(Sequential):
    """A Channel Wrapper simulates multiple segments of polarization-multiplexed communication channels."""

    module_list: List = field(default_factory=list)
    seq_obj: Any = None
    L: int = 1
    params: Any = None
    debug: bool = False
    name: str = 'channel_wrapper'
    callbacks: Optional[Dict[Union[str, int], Callable]] = field(default_factory=dict)
    dgd_gen: Optional[Generator] = field(default=None, init=False, repr=False)
    theta_gen: Optional[Generator] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize ChannelWrapper attributes."""
        if self.params is not None:
            self.dgd_gen = (dgd for dgd in self.params[1][0])
            self.theta_gen = (theta for theta in self.params[1][1])

    def set_params(self) -> Sequential:
        module = copy.deepcopy(self.seq_obj)
        # Accept raw list of processors by wrapping into Sequential
        if isinstance(module, list):
            module = Sequential(module_list=module)
        # Duck typing: any object with a module_list attribute is treated as sequential
        if not hasattr(module, 'module_list'):
            print("[ChannelWrapper debug] seq_obj type:", type(self.seq_obj))
            print("[ChannelWrapper debug] dir(seq_obj):", dir(self.seq_obj))
            raise AttributeError('Object does not provide a module_list attribute; cannot set params')
        for submodule in module.module_list:
            if getattr(submodule, 'name', None) == 'SOP_Drift':
                submodule.linewidth = self.params[0]
            if getattr(submodule, 'name', None) == 'PMD':
                submodule.t_dgd = next(self.dgd_gen)
                submodule.theta = next(self.theta_gen)
            if getattr(submodule, 'name', None) == 'PDL':
                submodule.gamma_db = np.random.choice(self.params[2])
        return module


    def forward(self, X: np.ndarray) -> np.ndarray:
        # Build the chain once if it's static, or just optimize the params
        # instead of deepcopying in a loop.
        Y = X
        # Create the segments once
        segments = [self.set_params() for _ in range(self.L)]
        
        # Process the signal
        for seq in segments:
            Y = seq(Y)
        return Y

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

@dataclass
class ChannelWrapper_(Sequential):
    """
    DP-PDL channel wrapper following Farsi et al., JLT 2022.

    H_k = Π_{n=1}^L Γ_n J_{k,n}
    """

    module_list: List = field(default_factory=list)
    seq_obj: Any = None
    L: int = 1                     # number of segments
    params: Any = None
    debug: bool = False
    name: str = 'channel_wrapper'
    callbacks: Optional[Dict] = field(default_factory=dict)

    dgd_gen: Optional[Generator] = field(default=None, init=False, repr=False)
    theta_gen: Optional[Generator] = field(default=None, init=False, repr=False)

    segments: List = field(default_factory=list, init=False)

    def __post_init__(self):
        """
        Build the channel ONCE.
        """
        if self.params is None:
            raise ValueError("ChannelWrapper requires params")

        # --- generators for PMD ---
        if self.params[1] is not None:
            self.dgd_gen = iter(self.params[1][0])

        delta_p_tot = self.params[0]
        delta_p_seg = delta_p_tot / self.L   # Eq. (7)

        # --- build segments ---
        self.segments = []
        for seg_idx in range(self.L):

            module = copy.deepcopy(self.seq_obj)
            if isinstance(module, list):
                module = Sequential(module_list=module)

            for submodule in module.module_list:

                # SOP: per-segment linewidth
                if getattr(submodule, 'name', None) == 'SOP_Drift':
                    submodule.linewidth = delta_p_seg
                    submodule.seg = 1

                # PMD: static per segment
                if getattr(submodule, 'name', None) == 'PMD':
                    submodule.t_dgd = next(self.dgd_gen)

                # PDL: static per segment
                if getattr(submodule, 'name', None) == 'PDL':
                    submodule.gamma_db = self.params[2]
                    submodule.gamma = (10**(submodule.gamma_db/10)-1)/(10**(submodule.gamma_db/10)+1)

            self.segments.append(module)


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagate through all segments.
        SOP memory is preserved across calls.
        """
        Y = X
        for seg in self.segments:
            Y = seg(Y)
        return Y

    def __call__(self, X):
        return self.forward(X)


@dataclass
class ChannelWrapper__(Sequential):

    module_list: List = field(default_factory=list)
    seq_obj: Any = None
    L: int = 1
    params: Any = None
    debug: bool = False
    name: str = 'channel_wrapper'
    callbacks: Optional[Dict] = field(default_factory=dict)

    dgd_gen: Optional[Generator] = field(default=None, init=False, repr=False)
    segments: List = field(default_factory=list, init=False)

    # store instantaneous PDL values
    rho_k: List = field(default_factory=list, init=False)

    # -------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------
    def __post_init__(self):

        if self.params is None:
            raise ValueError("ChannelWrapper requires params")

        delta_p_tot = self.params[0]
        pmd_params  = self.params[1]
        phi_seg_db  = self.params[2]

        delta_p_seg = delta_p_tot / self.L

        if pmd_params is not None:
            self.dgd_gen = iter(pmd_params[0])

        self.segments = []

        for _ in range(self.L):

            module = copy.deepcopy(self.seq_obj)
            if isinstance(module, list):
                module = Sequential(module_list=module)

            for submodule in module.module_list:

                nm = getattr(submodule, 'name', None)

                if nm == 'SOP_Drift':
                    submodule.linewidth = delta_p_seg
                    submodule.segments = 1
                    submodule.sigma2 = 2 * np.pi * submodule.linewidth * submodule.T_symb

                elif nm == 'PMD' and self.dgd_gen is not None:
                    submodule.t_dgd = next(self.dgd_gen)

                elif nm == 'PDL':
                    submodule.gamma_db = phi_seg_db
                    submodule.gamma = (10**(phi_seg_db/10)-1) / \
                                      (10**(phi_seg_db/10)+1)

            self.segments.append(module)

    # -------------------------------------------------------------
    # Average aggregated PDL (Eq. 10)
    # -------------------------------------------------------------
    def average_aggregated_pdl_db(self, H_list):

        rho_vals = []

        for Hk in H_list:
            s = np.linalg.svd(Hk, compute_uv=False)
            rho_lin = (s[0]**2) / (s[1]**2)
            rho_vals.append(rho_lin)

        rho_mean = np.mean(rho_vals)

        return 10 * np.log10(rho_mean)

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:

        Y = np.zeros_like(X)
        self.rho_k = []

        Nsym = X.shape[1]
        H_list = []

        for k in range(Nsym):

            H_k = np.eye(2, dtype=complex)
            x_k = X[:, k:k+1]

            # propagate through segments
            for seg in self.segments:

                x_k = seg(x_k)

                # ----- extract linear action of segment -----
                e1 = np.array([[1],[0]], dtype=complex)
                e2 = np.array([[0],[1]], dtype=complex)

                h1 = seg(e1)
                h2 = seg(e2)

                H_seg = np.hstack([h1, h2])

                # accumulate full channel
                H_k = H_seg @ H_k

            # store final full channel only
            H_list.append(H_k)

            Y[:, k:k+1] = x_k

            # instantaneous aggregated PDL (Eq. 9)
            s = np.linalg.svd(H_k, compute_uv=False)
            rho_db = 20 * np.log10(s[0] / s[1])

            self.rho_k.append(rho_db)

        if self.debug:
            print("Mean instantaneous PDL (dB):", np.mean(self.rho_k))
            print("Aggregated PDL Eq(10) (dB):",
                  self.average_aggregated_pdl_db(H_list))

        return Y

    # -------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------
    def mean_pdl(self):
        return np.mean(self.rho_k)

    def __call__(self, X):
        return self.forward(X)
