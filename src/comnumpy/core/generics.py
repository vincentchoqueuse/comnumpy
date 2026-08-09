import numpy as np
import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Union, Dict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
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
    # init=False fields use default_factory: with slots=True a plain default
    # would live on the (removed) class attribute and never reach the instance
    debug: bool = field(default_factory=bool, init=False)
    Y: Optional[np.ndarray] = field(default_factory=lambda: None, init=False, repr=False)

    def forward(self, x: np.ndarray, /) -> np.ndarray:
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

    def prepare(self, X: np.ndarray, /) -> None:
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
            name = getattr(self, "name", type(self).__name__)
            logger.debug("processor=%s: output_shape=%s, type=%s",
                         name, Y.shape, Y.dtype)
            self.Y = Y  # save data for debugging
            return Y
        else:
            return self.forward(X)


@dataclass(slots=True)
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
    taps : list of str, optional, keyword-only
        Block ids (see :meth:`block_ids`) whose output should be recorded
        during ``forward``. Taps are the *only* observation mechanism:
        they are chain metadata, so the module list describes the
        communication system and nothing else, and the cost is one
        dictionary store per tapped block (a reference is kept, no copy
        -- blocks allocate fresh outputs, so the reference stays valid).
        Retrieve with :meth:`tap`.
    wiring : dict, optional, keyword-only
        Extra data edges, as ``{"block_id.param": "source_block_id"}``.
        Before a target block runs, the chain assigns it the signal
        produced by the source block *in the same pass* -- this is how a
        data-aided estimator receives a reference generated upstream
        (``{"phase_comp.target_data": "tx"}``). The source is tapped
        automatically and must come earlier in the chain, so the value is
        never stale. Like taps, the edge is chain metadata: blocks stay
        declarative and never hold a reference to another block.

    Examples
    --------
    >>> from comnumpy.core.generators import SymbolGenerator
    >>> from comnumpy.core.channels import AWGN
    >>> chain = Sequential([SymbolGenerator(4), AWGN(snr_dB=10)],
    ...                    taps=["generator"])
    >>> y = chain.seed(1)(5)
    >>> print(chain.tap("generator"))
    [0 0 3 1 3]
    """
    module_list: list
    debug: bool = False
    name: str = 'sequential'
    callbacks: Optional[Dict[Union[str, int], Callable]] = field(default_factory=dict)
    taps: Optional[list] = field(default=None, kw_only=True)
    wiring: Optional[Dict[str, str]] = field(default=None, kw_only=True)
    # signals recorded at the declared taps (references, not copies)
    tapped_: dict = field(init=False, repr=False, default_factory=dict)


    def block_ids(self):
        """
        Return the addressable identifier of each block (decision D31/D34).

        The identifier is the block's ``name`` field when it has one
        (lower snake case), otherwise the class name in snake case; a
        numeric suffix disambiguates duplicates (``awgn``, ``awgn_2``).
        """
        import re

        def base_id(module):
            name = getattr(module, "name", None)
            if not name:
                name = type(module).__name__
            slug = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
            slug = re.sub(r"[^0-9a-zA-Z]+", "_", slug).strip("_").lower()
            return slug or "block"

        ids, seen = [], {}
        for module in self.module_list:
            base = base_id(module)
            seen[base] = seen.get(base, 0) + 1
            ids.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
        return ids

    def get_module_by_id(self, block_id: str):
        """Retrieve a module by its :meth:`block_ids` identifier."""
        ids = self.block_ids()
        if block_id not in ids:
            raise KeyError(f"unknown block id {block_id!r}; known: {ids}")
        return self.module_list[ids.index(block_id)]

    def seed(self, seed):
        """
        Seed every stochastic block deterministically (decision D6).

        A ``numpy.random.SeedSequence`` spawned from ``seed`` gives each
        block that declares a ``seed`` field its own independent child
        seed, then re-runs its parametric initialization so the RNG is
        rebuilt. Same chain + same seed = same signal, whatever the
        number or order of stochastic blocks.

        Examples
        --------
        >>> from comnumpy.core.generators import SymbolGenerator
        >>> chain = Sequential([SymbolGenerator(4)])
        >>> _ = chain.seed(42)
        >>> x1 = chain(5)
        >>> _ = chain.seed(42)
        >>> print(all(chain(5) == x1))
        True
        """
        seed_sequence = np.random.SeedSequence(seed)
        stochastic = [
            module for module in self.module_list
            if dataclasses.is_dataclass(module)
            and "seed" in {f.name for f in dataclasses.fields(module)}
        ]
        for module, child in zip(stochastic, seed_sequence.spawn(len(stochastic)),
                                 strict=True):
            module.seed = int(child.generate_state(1)[0])
            post_init = getattr(module, "__post_init__", None)
            if post_init is not None:
                post_init()
        return self

    def set_params(self, **params):
        """
        Reconfigure blocks after construction (decision D34).

        Parameters are addressed with the dotted notation
        ``"<block_id>.<field>"`` where ``<block_id>`` comes from
        :meth:`block_ids`. After all assignments, the parametric
        precomputation of each touched block (``__post_init__``) is
        re-run, so the block state stays consistent.

        Examples
        --------
        >>> from comnumpy.core.channels import AWGN
        >>> chain = Sequential([AWGN(sigma2=0.1)])
        >>> _ = chain.set_params(**{"awgn.sigma2": 0.01})
        >>> chain[0].sigma2
        0.01
        """
        touched = []
        for dotted, value in params.items():
            block_id, _, field_name = dotted.partition(".")
            if not field_name:
                raise ValueError(
                    f"parameter {dotted!r} is not in the dotted form "
                    f"'<block_id>.<field>' (block ids: {self.block_ids()})")
            module = self.get_module_by_id(block_id)
            field_names = {f.name for f in dataclasses.fields(module) if f.init}
            if field_name not in field_names:
                raise AttributeError(
                    f"{type(module).__name__} has no parameter {field_name!r}; "
                    f"available: {sorted(field_names)}")
            setattr(module, field_name, value)
            if module not in touched:
                touched.append(module)

        # re-run the parametric precomputation once per touched block
        for module in touched:
            post_init = getattr(module, "__post_init__", None)
            if post_init is not None:
                post_init()
        return self

    @staticmethod
    def _format_value(value):
        if isinstance(value, np.ndarray):
            return f"ndarray{value.shape}"
        if isinstance(value, Sequential):
            return "Sequential(...)"
        return repr(value)

    def _block_repr(self, module):
        if dataclasses.is_dataclass(module):
            args = ", ".join(
                f"{f.name}={self._format_value(getattr(module, f.name))}"
                for f in dataclasses.fields(module) if f.repr and f.init)
            return f"{type(module).__name__}({args})"
        return repr(module)

    def __repr__(self):
        """Structural view of the chain (decision D33a)."""
        lines = [f"{type(self).__name__}("]
        for index, module in enumerate(self.module_list):
            lines.append(f"  ({index}): {self._block_repr(module)}")
        lines.append(")")
        return "\n".join(lines)

    def summary(self, X, print_out=True):
        """
        Run the chain on ``X`` and tabulate each block's output shape,
        dtype and execution time (decision D33b). Returns the rows.
        """
        rows = []
        Y = X
        for index, (block_id, module) in enumerate(
                zip(self.block_ids(), self.module_list, strict=True)):
            start_time = time.time()
            Y = module(Y)
            elapsed_ms = (time.time() - start_time) * 1e3
            shape = getattr(Y, "shape", None)
            dtype = getattr(Y, "dtype", type(Y).__name__)
            rows.append((index, type(module).__name__, block_id,
                         tuple(shape) if shape is not None else None,
                         str(dtype), elapsed_ms))

        if print_out:
            header = (f"{'#':<4} {'block':<28} {'id':<20} "
                      f"{'output shape':<18} {'dtype':<12} {'time ms':>8}")
            print(header)
            print("-" * len(header))
            for index, cls, block_id, shape, dtype, ms in rows:
                print(f"{index:<4} {cls:<28} {block_id:<20} "
                      f"{str(shape):<18} {dtype:<12} {ms:>8.2f}")
        return rows

    def to_mermaid(self):
        """
        Mermaid flowchart of the chain (decision D33c), renderable by
        sphinxcontrib-mermaid or any mermaid viewer.
        """
        lines = ["flowchart LR"]
        ids = self.block_ids()
        for block_id, module in zip(ids, self.module_list, strict=True):
            lines.append(f'    {block_id}["{type(module).__name__}"]')
        for a, b in zip(ids[:-1], ids[1:], strict=True):
            lines.append(f"    {a} --> {b}")
        return "\n".join(lines)

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

        for processor in self.module_list:
            start_time = time.time()
            Y = processor(Y)
            stop_time = time.time()
            time_elapsed[processor.name] = stop_time - start_time

        return time_elapsed

    def _resolve_edges(self):
        """Validate taps/wiring against the block ids; return the plan.

        Returns ``(ids, recorded, feeds)`` where ``recorded`` is the set of
        ids to store during the pass and ``feeds`` maps a block index to
        the ``(param, source_id)`` pairs to assign before it runs.
        """
        if not self.taps and not self.wiring:
            return None, set(), {}

        ids = self.block_ids()
        index_of = {block_id: i for i, block_id in enumerate(ids)}
        recorded = set(self.taps or ())
        unknown = sorted(recorded - index_of.keys())
        if unknown:
            raise KeyError(f"unknown tap ids {unknown}; known block ids: {ids}")

        feeds: Dict[int, list] = {}
        for target, source in (self.wiring or {}).items():
            block_id, _, param = target.partition(".")
            if not param:
                raise KeyError(
                    f"wiring key {target!r} must be 'block_id.param', "
                    f"e.g. 'phase_comp.target_data'")
            for candidate, role in ((block_id, "target"), (source, "source")):
                if candidate not in index_of:
                    raise KeyError(
                        f"wiring {target!r}: unknown {role} block id "
                        f"{candidate!r}; known block ids: {ids}")
            if index_of[source] >= index_of[block_id]:
                raise ValueError(
                    f"wiring {target!r} reads {source!r}, which runs at or "
                    f"after the target block -- a data edge must point "
                    f"forward, otherwise the value would come from the "
                    f"previous run.")
            recorded.add(source)
            feeds.setdefault(index_of[block_id], []).append((param, source))
        return ids, recorded, feeds

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Process the input data through all modules in the sequence.
        """
        ids, recorded, feeds = self._resolve_edges()

        Y = X
        for index, processor in enumerate(self.module_list):
            # feed declared data edges before the block runs (wiring)
            for param, source in feeds.get(index, ()):
                setattr(processor, param, self.tapped_[source])

            Y = processor(Y)

            # record the output at declared taps (reference, no copy)
            if ids is not None and ids[index] in recorded:
                self.tapped_[ids[index]] = Y

            # run callback if needed
            key = getattr(processor, 'name', None)
            if key in self.callbacks:
                self.callbacks[key](Y)
        return Y

    def tap(self, block_id: str):
        """
        Return the signal recorded at a declared tap during the last run.
        """
        if block_id not in self.tapped_:
            raise KeyError(
                f"no signal recorded for tap {block_id!r}; recorded: "
                f"{sorted(self.tapped_)}, declared taps: {self.taps or []} "
                f"(declare the tap, then run the chain)")
        return self.tapped_[block_id]

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

    def __call__(self, x):
        """
        Process the input data by calling the forward method.
        """
        return self.forward(x)


