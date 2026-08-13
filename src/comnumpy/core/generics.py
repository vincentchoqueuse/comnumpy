import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Set, Tuple, TypedDict,
                    Union)

import numpy as np

__all__ = ["Processor", "ChainGraph", "Sequential"]

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

    def set_debug(self, debug: Optional[bool] = None) -> None:
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


class ChainGraph(TypedDict):
    """Renderer-independent model returned by :meth:`Sequential.graph`."""

    nodes: List[Dict[str, str]]
    signal_edges: List[Tuple[str, str]]
    data_edges: List[Tuple[str, str, str]]
    taps: List[str]


# (index, class name, block id, output shape, dtype, elapsed ms)
_SummaryRow = Tuple[int, str, str, Optional[Tuple[int, ...]], str, float]
# (block ids, ids to record, {block index: [(param, source id)]})
_EdgePlan = Tuple[Optional[List[str]], Set[str], Dict[int, List[Tuple[str, str]]]]


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
    elapsed_ : float
        Wall time of the last pass, in seconds (data-dependent, hence the
        trailing underscore, decision D23). Every call records it, so
        "how long does this chain take" needs no stopwatch around the
        call site; :meth:`profile_execution_time` breaks the same number
        down block by block.
    wiring : dict, optional, keyword-only
        Extra data edges, as ``{"block_id.param": "source_block_id"}``.
        Before a target block runs, the chain assigns it the signal
        produced by the source block *in the same pass* -- this is how a
        data-aided estimator receives a reference generated upstream
        (``{"phase_comp.reference": "tx"}``). The source is tapped
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
    >>> chain.elapsed_ > 0
    True
    """
    module_list: List[Processor]
    debug: bool = False
    name: str = 'sequential'
    callbacks: Optional[Dict[Union[str, int], Callable[[Any], None]]] = \
        field(default_factory=dict)
    taps: Optional[List[str]] = field(default=None, kw_only=True)
    wiring: Optional[Dict[str, str]] = field(default=None, kw_only=True)
    # signals recorded at the declared taps (references, not copies)
    tapped_: Dict[str, Any] = field(init=False, repr=False, default_factory=dict)
    # wall time of the last pass (D23: data-dependent, hence the underscore)
    elapsed_: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self._check_resampling_filters()

    def _check_resampling_filters(self) -> None:
        r"""Warn when a rate change carries its anti-alias filter by hand.

        A decimation by :math:`L` needs the band limited to :math:`1/L`
        of the Nyquist frequency beforehand, and an interpolation by
        :math:`L` needs the images removed at the same cutoff
        afterwards. :class:`~comnumpy.core.processors.Downsampler` and
        :class:`~comnumpy.core.processors.Upsampler` build that filter
        themselves from ``L`` when ``use_filter=True``.

        Writing the pair out instead states the cutoff twice -- once as
        ``L``, once as the filter's ``wn`` -- and nothing then keeps the
        two in step. That is the defect D41 exists to prevent, and it is
        silent: a cutoff below the signal removes part of it, which
        looks exactly like a channel impairment and puts a floor under
        every curve measured through the chain.

        This warns rather than raises. A brick-wall filter next to a
        rate change is not always the anti-alias filter -- selecting one
        channel out of a multiplex before decimating is a legitimate
        pair of filters doing two jobs -- so the chain says what it sees
        and computes anyway.
        """
        from comnumpy.core.filters import BWFilter
        from comnumpy.core.processors import Downsampler, Upsampler

        pairs = zip(self.module_list, self.module_list[1:], strict=False)
        for first, second in pairs:
            if isinstance(first, BWFilter) and isinstance(second, Downsampler):
                mask, resampler, side, kind = first, second, "before", "aliasing"
            elif isinstance(first, Upsampler) and isinstance(second, BWFilter):
                mask, resampler, side, kind = second, first, "after", "imaging"
            else:
                continue
            if resampler.use_filter or resampler.L <= 0:
                continue        # two filters, deliberately, or nothing to check
            expected = 1.0 / resampler.L
            name = type(resampler).__name__
            if abs(mask.wn - expected) > 1e-9 * max(expected, 1.0):
                too_narrow = mask.wn < expected
                logger.warning(
                    "a BWFilter of cutoff %g sits %s %s(L=%d), whose anti-%s "
                    "filter has cutoff 1/L = %g. If it is meant to be that "
                    "filter, its cutoff is wrong and the chain is %s. Write "
                    "%s(%d, use_filter=True), which derives the cutoff from L "
                    "and cannot disagree with it (D41).",
                    mask.wn, side, name, resampler.L, kind, expected,
                    "throwing signal away, which reads as a channel "
                    "impairment and floors every curve measured through it"
                    if too_narrow else
                    "passing a band the rate change cannot carry, which folds "
                    "back onto the signal",
                    name, resampler.L)
            else:
                logger.warning(
                    "BWFilter(%g) %s %s(L=%d) is exactly the filter "
                    "%s(%d, use_filter=True) builds. Prefer the argument: "
                    "the cutoff is then written once, and cannot drift away "
                    "from the rate change it belongs to (D41).",
                    mask.wn, side, name, resampler.L, name, resampler.L)

    def block_ids(self) -> List[str]:
        """
        Return the addressable identifier of each block (decision D31/D34).

        The identifier is the block's ``name`` field when it has one
        (lower snake case), otherwise the class name in snake case; a
        numeric suffix disambiguates duplicates (``awgn``, ``awgn_2``).
        """
        import re

        def base_id(module: object) -> str:
            name = getattr(module, "name", None)
            if not name:
                name = type(module).__name__
            slug = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
            slug = re.sub(r"[^0-9a-zA-Z]+", "_", slug).strip("_").lower()
            return slug or "block"

        ids: List[str] = []
        seen: Dict[str, int] = {}
        for module in self.module_list:
            base = base_id(module)
            seen[base] = seen.get(base, 0) + 1
            ids.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
        return ids

    def get_module_by_id(self, block_id: str) -> Processor:
        """Retrieve a module by its :meth:`block_ids` identifier."""
        ids = self.block_ids()
        if block_id not in ids:
            raise KeyError(f"unknown block id {block_id!r}; known: {ids}")
        return self.module_list[ids.index(block_id)]

    def seed(self, seed: int) -> "Sequential":
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
            # setattr, not module.seed: the field was discovered by name
            # above and no base class declares it (noqa: B010 is the price)
            setattr(module, "seed",  # noqa: B010
                    int(child.generate_state(1)[0]))
            post_init = getattr(module, "__post_init__", None)
            if post_init is not None:
                post_init()
        return self

    def _split_parameter(self, key: str) -> tuple[str, str]:
        """Split ``"block.field"`` or ``"block__field"`` (decision D34).

        The dotted form is the addressing used everywhere a parameter is
        held in a variable -- :func:`~comnumpy.monte_carlo.sweep` builds those
        strings. A dot is not a valid character in a keyword argument,
        though, so writing one by hand costs a ``**{...}`` wrapper. The
        double underscore is the same address spelled as an identifier,
        which is scikit-learn's convention for the same method and is
        accepted here for that reason.
        """
        if "." in key:
            return tuple(key.split(".", 1))                    # type: ignore[return-value]
        # The first double underscore is the separator, unambiguously:
        # `block_ids` collapses every run of non-alphanumerics into a
        # single underscore, so no block id can contain one (checked in
        # tests/core/test_set_params.py).
        block_id, _, field_name = key.partition("__")
        return block_id, field_name

    def set_params(self, **params: Any) -> "Sequential":
        """
        Reconfigure blocks after construction (decision D34).

        Parameters are addressed by block and field, written either
        ``"<block_id>.<field>"`` or ``<block_id>__<field>`` -- the second
        is the same address as a plain keyword argument, so it needs no
        ``**{...}`` around it. ``<block_id>`` comes from
        :meth:`block_ids`. After all assignments, the parametric
        precomputation of each touched block (``__post_init__``) is
        re-run, so the block state stays consistent.

        Examples
        --------
        >>> from comnumpy.core.channels import AWGN
        >>> chain = Sequential([AWGN(sigma2=0.1)])
        >>> _ = chain.set_params(awgn__sigma2=0.01)
        >>> chain[0].sigma2
        0.01

        The dotted form is the one to use when the address is computed:

        >>> _ = chain.set_params(**{"awgn.sigma2": 0.02})
        >>> chain[0].sigma2
        0.02
        """
        touched: List[Processor] = []
        for dotted, value in params.items():
            block_id, field_name = self._split_parameter(dotted)
            if not field_name:
                raise ValueError(
                    f"parameter {dotted!r} does not name a block and a "
                    f"field, as '<block_id>.<field>' or "
                    f"'<block_id>__<field>' (block ids: {self.block_ids()})")
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
    def _format_value(value: object) -> str:
        if isinstance(value, np.ndarray):
            return f"ndarray{value.shape}"
        if isinstance(value, Sequential):
            return "Sequential(...)"
        return repr(value)

    def _block_repr(self, module: object) -> str:
        if dataclasses.is_dataclass(module):
            args = ", ".join(
                f"{f.name}={self._format_value(getattr(module, f.name))}"
                for f in dataclasses.fields(module) if f.repr and f.init)
            return f"{type(module).__name__}({args})"
        return repr(module)

    def __repr__(self) -> str:
        """Structural view of the chain (decision D33a)."""
        lines = [f"{type(self).__name__}("]
        for index, module in enumerate(self.module_list):
            lines.append(f"  ({index}): {self._block_repr(module)}")
        lines.append(")")
        return "\n".join(lines)

    def summary(self, X: Any, print_out: bool = True) -> List[_SummaryRow]:
        """
        Run the chain on ``X`` and tabulate each block's output shape,
        dtype and execution time (decision D33b). Returns the rows.
        """
        rows: List[_SummaryRow] = []
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

    def graph(self) -> "ChainGraph":
        """
        Structural view of the chain: nodes, signal edges, data edges.

        The renderer-independent model behind :meth:`to_mermaid`. Signal
        edges are the implicit linear path; data edges are the ones
        declared in ``wiring`` (decision D42c), which a picture of the
        chain must show or it is not telling the truth.

        Returns
        -------
        dict
            ``nodes`` (list of ``{"id", "type"}``), ``signal_edges``
            (list of ``(source, target)``), ``data_edges`` (list of
            ``(source, target, param)``) and ``taps`` (observed ids).

        Examples
        --------
        >>> from comnumpy.core.generators import SymbolGenerator
        >>> from comnumpy.core.channels import AWGN
        >>> g = Sequential([SymbolGenerator(4), AWGN(snr_dB=10)]).graph()
        >>> print(g["signal_edges"])
        [('generator', 'awgn')]
        """
        ids = self.block_ids()
        data_edges: List[Tuple[str, str, str]] = []
        for target, source in (self.wiring or {}).items():
            block_id, _, param = target.partition(".")
            data_edges.append((source, block_id, param))
        return {
            "nodes": [{"id": block_id, "type": type(module).__name__}
                      for block_id, module in zip(ids, self.module_list, strict=True)],
            "signal_edges": list(zip(ids[:-1], ids[1:], strict=True)),
            "data_edges": data_edges,
            "taps": list(self.taps or ()),
        }

    def to_mermaid(self) -> str:
        """
        Mermaid flowchart of the chain (decision D33c), renderable by
        sphinxcontrib-mermaid, GitHub, or any mermaid viewer.

        Signal flow is drawn with solid arrows, declared data edges
        (``wiring``, D42c) with dashed arrows labelled by the parameter
        they feed, and tapped blocks are outlined -- so the picture shows
        every edge the chain actually has.

        Examples
        --------
        >>> from comnumpy.core.generators import SymbolGenerator
        >>> from comnumpy.core.channels import AWGN
        >>> chain = Sequential([SymbolGenerator(4), AWGN(snr_dB=10)],
        ...                    taps=["generator"])
        >>> print(chain.to_mermaid())
        flowchart LR
            generator["SymbolGenerator"]
            awgn["AWGN"]
            generator --> awgn
            classDef tapped stroke-dasharray: 4 2
            class generator tapped
        """
        model = self.graph()
        lines = ["flowchart LR"]
        for node in model["nodes"]:
            lines.append(f'    {node["id"]}["{node["type"]}"]')
        for a, b in model["signal_edges"]:
            lines.append(f"    {a} --> {b}")
        for source, target, param in model["data_edges"]:
            lines.append(f"    {source} -.->|{param}| {target}")
        if model["taps"]:
            lines.append("    classDef tapped stroke-dasharray: 4 2")
            lines.append(f"    class {','.join(model['taps'])} tapped")
        return "\n".join(lines)

    def set_debug(self, debug: Optional[bool] = None) -> None:
        """
        Change the debugging mode
        """
        for module in self.module_list:
            module.set_debug(debug)

    def profile_execution_time(self, X: np.ndarray) -> Dict[str, float]:
        """Time each block over one pass, keyed by block id.

        The pass is the ordinary one: taps are recorded and the data
        edges declared in ``wiring`` are fed, so a chain that profiles is
        the chain that runs. The keys are the ids of :meth:`block_ids`,
        so two blocks sharing a name still get one entry each.

        Parameters
        ----------
        X : np.ndarray
            Input signal, or the requested size when the chain starts
            with a source block.

        Returns
        -------
        dict
            Seconds spent in each block, in chain order.

        Examples
        --------
        >>> from comnumpy.core.generators import SymbolGenerator
        >>> from comnumpy.core.channels import AWGN
        >>> chain = Sequential([SymbolGenerator(4), AWGN(snr_dB=10)])
        >>> print(list(chain.profile_execution_time(8)))
        ['generator', 'awgn']
        """
        ids, recorded, feeds = self._resolve_edges()
        if ids is None:
            ids = self.block_ids()

        Y = X
        time_elapsed: Dict[str, float] = {}

        for index, processor in enumerate(self.module_list):
            for param, source in feeds.get(index, ()):
                setattr(processor, param, self.tapped_[source])
            start_time = time.time()
            Y = processor(Y)
            stop_time = time.time()
            if ids[index] in recorded:
                self.tapped_[ids[index]] = Y
            time_elapsed[ids[index]] = stop_time - start_time

        self.elapsed_ = sum(time_elapsed.values())
        return time_elapsed

    def _resolve_edges(self) -> _EdgePlan:
        """Validate taps/wiring against the block ids; return the plan.

        Returns ``(ids, recorded, feeds)`` where ``recorded`` is the set of
        ids to store during the pass and ``feeds`` maps a block index to
        the ``(param, source_id)`` pairs to assign before it runs.
        """
        if not self.taps and not self.wiring:
            return None, set(), {}

        ids = self.block_ids()
        index_of = {block_id: i for i, block_id in enumerate(ids)}
        recorded: Set[str] = set(self.taps or ())
        unknown = sorted(recorded - index_of.keys())
        if unknown:
            raise KeyError(f"unknown tap ids {unknown}; known block ids: {ids}")

        feeds: Dict[int, List[Tuple[str, str]]] = {}
        for target, source in (self.wiring or {}).items():
            block_id, _, param = target.partition(".")
            if not param:
                raise KeyError(
                    f"wiring key {target!r} must be 'block_id.param', "
                    f"e.g. 'phase_comp.reference'")
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
        start = time.perf_counter()
        for index, processor in enumerate(self.module_list):
            # feed declared data edges before the block runs (wiring)
            for param, source in feeds.get(index, ()):
                setattr(processor, param, self.tapped_[source])

            Y = processor(Y)

            # record the output at declared taps (reference, no copy)
            if ids is not None and ids[index] in recorded:
                self.tapped_[ids[index]] = Y

            # run callback if needed
            # callbacks= may be explicitly None, and `key in None` used to
            # raise a TypeError from inside the pass
            callbacks = self.callbacks or {}
            key = getattr(processor, 'name', None)
            if key is not None and key in callbacks:
                callbacks[key](Y)
        self.elapsed_ = time.perf_counter() - start
        return Y

    def tap(self, block_id: str) -> Any:
        """
        Return the signal recorded at a declared tap during the last run.
        """
        if block_id not in self.tapped_:
            raise KeyError(
                f"no signal recorded for tap {block_id!r}; recorded: "
                f"{sorted(self.tapped_)}, declared taps: {self.taps or []} "
                f"(declare the tap, then run the chain)")
        return self.tapped_[block_id]

    def get_module_by_index(self, index: int) -> Processor:
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

    def get_module_by_name(self, module_name: str) -> Processor:
        """
        Retrieve a module from the module list by its name.
        """
        for module in self.module_list:
            if getattr(module, "name", None) == module_name:
                return module
        raise AttributeError(f"Module '{module_name}' not found in class {self.__class__.__name__}.")

    def __getitem__(self, key: Union[str, int]) -> Processor:
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
        # the annotation says int, but nothing stops a notebook from
        # slicing a chain, and the message is better than numpy's
        if not isinstance(key, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"key must be a block name (str) or an index (int), got "
                f"{type(key).__name__}")
        return self.get_module_by_index(key)

    def __call__(self, x: Any) -> np.ndarray:
        """
        Process the input data by calling the forward method.
        """
        return self.forward(x)


