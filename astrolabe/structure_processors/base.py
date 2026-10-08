import hashlib
import importlib
import logging
import os
import pickle
import shutil
import tempfile
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from astrolabe.core import global_config
from astrolabe.utils.log_events import event_span, log_event


logger = logging.getLogger(__name__)


class StructureProcessorRegistry:
    """
    Registry for coordinate-processing implementations.
    Stable configuration identifiers select docking, alignment or rescue classes.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, predocker_class: type):
        """Register a structure processor under a stable configuration key."""
        if not issubclass(predocker_class, StructureProcessor):
            raise ValueError(
                f"Class {predocker_class.__name__} must inherit from StructureProcessor"
            )
        cls._registry[name] = predocker_class
        logging.info(f"Registered predocker: {name} -> {predocker_class.__name__}")

    @classmethod
    def get(cls, name: str) -> type:
        """Resolve a structure processor by its configuration key."""
        if name not in cls._registry:
            # Lazy-import predocker modules on demand to keep CLI startup fast.
            module = _PREDOCKER_TYPE_TO_MODULE.get(name)
            if module is not None:
                importlib.import_module(module)

        if name not in cls._registry:
            raise KeyError(f"Predocker '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list:
        """List registered structure-processor configuration keys."""
        return list(cls._registry.keys())


def initialize_predockers(predocker_args):
    predocker_terms = predocker_args.get("predocker_terms")
    global_config.init_const("predockers", {})
    for alias, config in predocker_terms.items():
        predocker_class = StructureProcessorRegistry.get(config._type)
        kwargs = {}
        for default_key in _predocker_default_keys(str(config._type)):
            kwargs.update(predocker_args.get(default_key, {}))
        kwargs.update(config)
        predocker = predocker_class(**kwargs)
        setattr(predocker, "alias", str(alias))
        global_config.predockers[alias] = predocker


def _predocker_default_keys(predocker_type: str) -> list[str]:
    canonical = _PREDOCKER_TYPE_ALIASES.get(predocker_type, predocker_type)
    keys = []
    for key in _PREDOCKER_DEFAULT_BASES.get(canonical, ()):
        if key not in keys:
            keys.append(key)
    for key in (canonical, predocker_type):
        if key not in keys:
            keys.append(key)
    return keys


def assign_predocker_to_mutators(mutators, predocker_args):
    aliases = predocker_args["predocker_terms"]
    for mutator in mutators:
        if mutator.name in predocker_args:
            mutator.predocker = predocker_args[mutator.name]
            if not all(name in aliases for name in predocker_args[mutator.name]):
                raise ValueError(
                    f"Predocker names for mutator {mutator.name} must be in {aliases}"
                )


def run_nested_predocker(predocker_names, input_file, *, runtime_oligo: str | None = None):
    for alias in predocker_names:
        predocker = global_config.predockers[alias]
        setattr(predocker, "alias", str(alias))
        if runtime_oligo is not None and hasattr(predocker, "runtime_clone"):
            predocker = predocker.runtime_clone(oligo=runtime_oligo)
            setattr(predocker, "alias", str(alias))
        input_file = predocker.run(input_file)
    return input_file


def _current_stage_name() -> str | None:
    try:
        stage_manager = global_config.stage_manager
    except RuntimeError:
        return None
    if stage_manager is None:
        return None
    return getattr(stage_manager, "current_stage", None)


def _current_step() -> int | None:
    try:
        return global_config.step
    except RuntimeError:
        return None


def run_nested_predocker_on_scores(predocker_names, evaluation_scores):
    """Run a mutator predocker chain against full evaluator scores when supported."""
    scores = evaluation_scores
    for alias in predocker_names:
        predocker = global_config.predockers[alias]
        run_on_scores = getattr(predocker, "run_on_scores", None)
        if callable(run_on_scores):
            scores = run_on_scores(scores)
            continue

        updated_paths = {
            oligo: run_nested_predocker([alias], path, runtime_oligo=oligo)
            for oligo, path in scores.complex_paths.items()
        }
        scores = scores.clone()
        scores.complex_paths = updated_paths
    return scores


class BaseProcessor(ABC):
    """
    Base class for processors that support method chaining.
    Allows multiple processors to be chained together for sequential processing.
    """

    def __init__(self, debug: bool = False, out_dir: Optional[str] = None):
        """
        Initialize the processor.

        Args:
            debug: If True, save outputs to debug directory
            out_dir: Output directory for debug mode
        """
        self.debug = debug
        self.out_dir = out_dir or os.getcwd()
        self._next_processor: Optional["BaseProcessor"] = None

    def chain(self, next_processor: "BaseProcessor") -> "BaseProcessor":
        """
        Chain this processor with another processor.

        Args:
            next_processor: The next processor in the chain

        Returns:
            The next processor for further chaining
        """
        self._next_processor = next_processor
        return next_processor

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process the input data.

        Args:
            input_data: Input data to process

        Returns:
            Processed data
        """
        pass

    def run_chain(self, input_data: Any) -> Any:
        """
        Run the complete processing chain.

        Args:
            input_data: Initial input data

        Returns:
            Final processed data
        """
        result = self.process(input_data)

        if self._next_processor:
            return self._next_processor.run_chain(result)

        return result


class StructureProcessor(ABC):
    """
    Base class for all predockers. Predockers are used to prepare structure input for sequence design models.
    Supports caching mechanism to avoid repeated expensive computations.
    """

    def __init__(self, cache_enabled: bool = True, **kwargs):
        """
        Initialize the predocker with given arguments.

        Args:
            debug: If True, save outputs to debug directory instead of temp files
            out_dir: Output directory for debug mode
            cache_enabled: Whether to enable caching mechanism
        """
        self.debug = global_config.is_debug
        self.out_dir = global_config.out_dir
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        self._cache_candidate_bank: Dict[str, list[Any]] = {}
        self._cache_rng = random.Random()
        self._condition = kwargs.get("_condition", None)
        self._condition_missing_defaults = dict(
            kwargs.get("_condition_missing_defaults", {}) or {}
        )

        # Create debug output directory if needed
        self.debug_dir = Path(self.out_dir) / "predocker"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Debug mode enabled. Outputs will be saved to: {self.debug_dir}")

    def _generate_cache_key(self, input_file) -> str:
        """
        Generate a unique cache key based on input parameters.

        Returns:
            MD5 hash as cache key
        """
        # Convert args and kwargs to a hashable representation
        cache_data = {
            "input_file": input_file,
            "class_name": self.__class__.__name__,
        }

        # Serialize and hash
        serialized = pickle.dumps(cache_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(serialized).hexdigest()

    def eval_condition(self) -> bool:
        decisions = global_config.decisions
        eval_result = decisions.evaluation_results[decisions.last_accepted]
        loss_computer = global_config.loss_computer
        score_context = getattr(loss_computer, "score_context", None)
        if callable(score_context):
            total_env = score_context(eval_result)
        else:
            per_oligo_scores, per_terms_score = loss_computer._extract_scores_with_registry(
                eval_result
            )
            total_env = {**per_oligo_scores, **per_terms_score}
        total_env = {**self._condition_missing_defaults, **total_env}
        return loss_computer.eval_condition(total_env, self._condition)

    def _get_output_path(self, filename: str) -> str:
        """
        Get output file path based on debug mode.

        Args:
            filename: Base filename

        Returns:
            Full output path
        """
        return str(self.debug_dir / filename)

    def _event_alias(self) -> str:
        return str(getattr(self, "alias", None) or self.__class__.__name__)

    def run(self, input_file) -> Any:
        """
        Run the predocker with caching support.
        This method should be cached since it is expensive to run and maybe called multiple times with the same input.

        Returns:
            Processing results
        """
        span = event_span(
            logger=logger,
            component="predocker",
            alias=self._event_alias(),
            stage=_current_stage_name(),
            step=_current_step(),
            input=input_file,
        )
        with span:
            if self._condition is not None:
                condition_met = self.eval_condition()
                log_event(
                    logger=logger,
                    level=logging.DEBUG,
                    event="predocker_condition",
                    component="predocker",
                    alias=self._event_alias(),
                    stage=_current_stage_name(),
                    step=_current_step(),
                    status="pass" if condition_met else "skip",
                    detail_level="debug",
                    detail=self._condition,
                    input=input_file,
                )
                if not condition_met:
                    span.skip(artifact=input_file, detail="condition_false")
                    return input_file
            input_file = self.preprocess(input_file)
            log_event(
                logger=logger,
                level=logging.DEBUG,
                event="predocker_preprocess",
                component="predocker",
                alias=self._event_alias(),
                stage=_current_stage_name(),
                step=_current_step(),
                status="success",
                detail_level="debug",
                artifact=input_file,
            )
            if not self.cache_enabled:
                result = self._run_impl(input_file)
                span.finish(artifact=result)
                return result

            # Generate cache key
            cache_key = self._generate_cache_key(input_file)

            # Check cache
            if cache_key in self._cache:
                result = self.take_cache(cache_key)
                log_event(
                    logger=logger,
                    level=logging.DEBUG,
                    event="predocker_cache",
                    component="predocker",
                    alias=self._event_alias(),
                    stage=_current_stage_name(),
                    step=_current_step(),
                    status="cache_hit",
                    detail_level="debug",
                    detail=cache_key[:8],
                    artifact=result,
                )
                span.finish(status="cache_hit", artifact=result, detail=cache_key[:8])
                return result

            log_event(
                logger=logger,
                level=logging.DEBUG,
                event="predocker_cache",
                component="predocker",
                alias=self._event_alias(),
                stage=_current_stage_name(),
                step=_current_step(),
                status="cache_miss",
                detail_level="debug",
                detail=cache_key[:8],
            )

            # Run implementation and cache result
            result = self._run_impl(input_file)
            cached_result = self._prepare_cached_result(cache_key, input_file, result)

            self._cache[cache_key] = cached_result
            self._after_cache_store(cache_key, input_file, cached_result)
            log_event(
                logger=logger,
                level=logging.DEBUG,
                event="predocker_cache",
                component="predocker",
                alias=self._event_alias(),
                stage=_current_stage_name(),
                step=_current_step(),
                status="cache_store",
                detail_level="debug",
                detail=cache_key[:8],
                artifact=cached_result,
            )
            result = self.take_cache(cache_key)
            span.finish(artifact=result)
            return result

    def _set_cache_candidates(self, cache_key, candidates):
        bank = [candidate for candidate in candidates if candidate is not None]
        if bank:
            self._cache_candidate_bank[cache_key] = bank

    def take_cache(self, cache_key):
        bank = self._cache_candidate_bank.get(cache_key)
        if bank:
            if len(bank) == 1:
                return bank[0]
            choice = self._cache_rng.choice(bank)
            logging.info(
                "Cache candidate sampling for %s returned %s",
                cache_key[:8],
                os.path.basename(str(choice)),
            )
            return choice
        return self._cache[cache_key]

    def _after_cache_store(self, cache_key, input_file, result):
        """Hook for subclasses to capture metadata after caching a result."""
        # Default implementation is a no-op.
        return None

    def _prepare_cached_result(self, cache_key, input_file, result):
        """Normalize a fresh result into the path stored in `_cache`."""
        cached_result = str(self.debug_dir / os.path.basename(result))
        if os.path.abspath(str(result)) != os.path.abspath(cached_result):
            shutil.move(result, cached_result)
        else:
            cached_result = str(result)
        return cached_result

    def preprocess(self, input_file) -> Any:
        """
        Preprocess input data before running the predocker.
        This method can be overridden by subclasses to perform any necessary preprocessing.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Preprocessed data
        """
        return input_file

    @abstractmethod
    def _run_impl(self, input_file) -> Any:
        """
        Actual implementation of the predocker logic.
        Subclasses should implement this method instead of run().

        Returns:
            Processing results
        """
        pass

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
        self._cache_candidate_bank.clear()
        logging.info(f"Cache cleared for {self.__class__.__name__}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "candidate_bank_sizes": {
                key: len(value) for key, value in self._cache_candidate_bank.items()
            },
        }


# Minimal mapping for lazy imports (avoid importing heavy predockers unless requested).
_PREDOCKER_TYPE_TO_MODULE: dict[str, str] = {
    "all_alanine": "astrolabe.structure_processors.all_alanine",
    "autodock_gpu": "astrolabe.structure_processors.autodock",
    "autodock_vina": "astrolabe.structure_processors.autodock",
    "ligand_fpocket_alignment": "astrolabe.structure_processors.autodock",
    "ligand_centroid_alignment": "astrolabe.structure_processors.autodock",
    "RAPiDock": "astrolabe.structure_processors.peptide_docker",
    "dimer_interface_placer": "astrolabe.structure_processors.ligand_interface_placer",
    "chemnet_placer": "astrolabe.structure_processors.ligand_interface_placer",
    "symmetry_canonicalizer": "astrolabe.structure_processors.symmetry",
    "hotspot_reference_align": "astrolabe.structure_processors.hotspot_reference",
    "haddock3_nucleic": "astrolabe.structure_processors.haddock3",
    "HADDOCK3_nucleic": "astrolabe.structure_processors.haddock3",
}

_PREDOCKER_TYPE_ALIASES: dict[str, str] = {
    "HADDOCK3_nucleic": "haddock3_nucleic",
}

_PREDOCKER_DEFAULT_BASES: dict[str, tuple[str, ...]] = {
    "haddock3_nucleic": ("haddock3",),
}


# Decorator for registering predockers
def register_structure_processor(name: str):
    """
    Decorator to register a predocker class.

    Args:
        name: Name to register the predocker under
    """

    def decorator(cls):
        StructureProcessorRegistry.register(name, cls)
        return cls

    return decorator
