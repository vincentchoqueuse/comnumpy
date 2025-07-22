import numpy as np
import dataclasses
import pprint
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, List, Callable, Union, Dict, Any


@dataclass
class Processor():
    r"""
    Base class for processing modules.

    This class provides a basic structure for processing modules, including a forward method
    that defines how input data should be processed. Derived classes should implement the
    forward method to define their specific processing logic.

    Signal Model
    ------------

    The generic model for a processor is :

    .. math::
        \mathbf{Y} = \mathbf{f}(\mathbf{X};\boldsymbol \theta) 
    
    * :math:`\mathbf{X}` corresponds to the input data,
    * :math:`\mathbf{Y}` corresponds to the output data,
    * :math:`\boldsymbol \theta` corresponds to the processor parameters.
    * :math:`\mathbf{f}(.)` is a multidimensional nonlinear function.

    When a processor is called with input data, it automatically computes the output data by calling its :code:`forward` method.

    .. NOTE::
        A processor is not necessarly fully deterministic. Some processor can also contain a stochastic part.

    """
    debug: bool = field(default=False, init=False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Process the input data
        """
        return x

    def set_debug(self, debug=None):
        """
        Change the debugging mode
        """
        if debug is None:
            debug = not self.debug

        self.debug = debug

    def prepare(self, X: np.ndarray ) -> np.ndarray:
        """
        Prepare the object before calling the forward method
        """
        pass

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Process the input data by calling the forward method.
        This method allows the processor to be called as a function. 
        
        If debug is True, this method also store the output_data
        """
        self.prepare(X)
        if self.debug:
            Y = self.forward(X)
            print(f"processor={self.name}: output_shape={Y.shape}, type={Y.dtype}")
            self.Y = Y  # save data for debugging
            return Y
        else:
            return self.forward(X)


@dataclass
class Sequential():
    r"""
    A sequential container for processing modules. 
    
    
    This class allows to create complex chain by stacking :math:`L` different processor modules. These processors are executed in the order they are added.

    Signal Model
    ------------

    - **Initialisation**:

    .. math::
        \mathbf{Y}_0 = \mathbf{X}

    - **Iterations**. For :math:`l=0, \cdots, L-1`, perform the following assignement

      .. math::
            \mathbf{Y}_{l+1} = \mathbf{f}_l(\mathbf{Y}_{l};\boldsymbol \theta_l)

      where :math:`\mathbf{f}_l()` corresponds to the multidimensional function of the :math:`l^{th}` processor.

    The :code:`forward` method returns the last output :math:`\mathbf{Y}_{L}`


    Callbacks
    ---------
    You can optionally provide a dictionary of callbacks via the
    :code:`callbacks` attribute. These functions will be called after each
    processor executes, receiving that processor's output as input. Keys of
    the dictionary correspond to processor names (:code:`processor.name`) in the chain.


    Attributes
    ----------
    module_list : list
        Ordered list of processing modules to be executed sequentially.
    debug : bool, optional
        Enables debug mode if True (default is False).
    name : str, optional
        Name of the sequential processor (default is 'sequential').
    callbacks : dict, optional
        Dictionary of callback functions called after each processor. Keys
        are processor names (str) or indices (int), values are callables
        accepting the processor output.
    """
    module_list: list
    debug: bool = False
    name: str = 'sequential'
    callbacks: Optional[Dict[Union[str, int], Callable]] = field(default_factory=dict)


    def asdict(self):
        dict = {}
        for index, module in enumerate(self.module_list):
            dict[f"id{index}"] = dataclasses.asdict(module)

        return dict
    
    def __repr__(self):
        """
        Show content of a sequential object
        """
        object_dict = self.asdict()
        return pprint.pformat(object_dict, indent=4)

    def set_debug(self, debug=None):
        """
        Change the debugging mode
        """
        for module in self.module_list:
            module.set_debug(debug)

    def profile_execution_time(self, X: np.ndarray):
        """
        Start profiling
        """
        Y = X
        time_elapsed = {}

        for index_processor, processor in enumerate(self.module_list):
            start_time = time.time()
            Y = processor(Y)
            stop_time = time.time()
            time_elapsed[processor.name] = stop_time - start_time
           
        return time_elapsed

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Process the input data through all modules in the sequence.
        """
        Y = X
        for processor in self.module_list:
            Y = processor(Y)

            # run callback if needed
            key = getattr(processor, 'name', None)
            if key in self.callbacks:
                self.callbacks[key](Y)
        return Y

    def get_module_by_index(self, index: int):
        """
        Retrieve a module from the module list by its index.
        """
        N_modules = len(self.module_list)
        if index >= N_modules:
            raise ValueError(f"Index {index} is out of bounds for sequential with {N_modules} modules")

        return self.module_list[index]

    def set_module_by_index(self, module: Processor, index: int):
        """
        Set module by index
        """
        self.module_list[index] = module

    def get_module_by_name(self, module_name: str):
        """
        Retrieve a module from the module list by its name.
        """
        for module in self.module_list:
            if hasattr(module, 'name'):
                if module.name == module_name:
                    return module
        raise AttributeError(f"Module '{module_name}' not found in class {self.__class__.__name__}.")

    def __getitem__(self, key):
        """
        Retrieve a module using the [] operator by its name or index.

        Parameters
        ----------
        key : str or int
            If a string, retrieves the module by name.
            If an integer, retrieves the module by index.

        Returns
        -------
        The module corresponding to the given name or index.
        """
        if isinstance(key, str):
            return self.get_module_by_name(key)
        elif isinstance(key, int):
            return self.get_module_by_index(key)
        else:
            raise TypeError("Key must be a string (for name) or an integer (for index).")

    def __call__(self, x, debug: bool = False):
        """
        Process the input data by calling the forward method.
        """
        return self.forward(x)


