from __future__ import annotations

import os
import unittest
from unittest import mock


class _RecordingExecutor:
    constructed_max_workers: list[int | None] = []

    def __init__(self, max_workers: int | None = None, **_kwargs: object) -> None:
        type(self).constructed_max_workers.append(max_workers)

    def __enter__(self) -> "_RecordingExecutor":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        return None

    def map(self, fn, iterable):
        return list(map(fn, iterable))


class TestAf3RefStructureParallelism(unittest.TestCase):
    def _make_fold_input(self, seq: str):
        from alphafold3.common.folding_input import Input, ProteinChain
        from alphafold3.constants import mmcif_names
        from alphafold3.data import msa as msa_module

        unpaired = msa_module.Msa.from_empty(
            query_sequence=seq,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        ).to_a3m()
        paired = msa_module.Msa.from_empty(
            query_sequence=seq,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        ).to_a3m()
        return Input(
            name="test_af3_refstructure_parallelism",
            chains=[
                ProteinChain(
                    id="A",
                    sequence=seq,
                    ptms=[],
                    unpaired_msa=unpaired,
                    paired_msa=paired,
                    templates=[],
                )
            ],
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )

    def _make_two_chain_fold_input(self, *, seq_a: str, seq_b: str):
        from alphafold3.common.folding_input import Input, ProteinChain
        # Use explicit A3M strings so unpaired != paired, ensuring we exercise the
        # cache for both code paths (two unique A3Ms per chain).
        def _a3m(query: str, *, extra: str | None) -> str:
            lines = [">query", query]
            if extra is not None:
                lines.extend([">hit", extra])
            return "\n".join(lines) + "\n"

        unpaired_a = _a3m(seq_a, extra=None)
        paired_a = _a3m(seq_a, extra=seq_a)
        unpaired_b = _a3m(seq_b, extra=None)
        paired_b = _a3m(seq_b, extra=seq_b)

        return Input(
            name="test_af3_msa_cache",
            chains=[
                ProteinChain(
                    id="A",
                    sequence=seq_a,
                    ptms=[],
                    unpaired_msa=unpaired_a,
                    paired_msa=paired_a,
                    templates=[],
                ),
                ProteinChain(
                    id="B",
                    sequence=seq_b,
                    ptms=[],
                    unpaired_msa=unpaired_b,
                    paired_msa=paired_b,
                    templates=[],
                ),
            ],
            rng_seeds=[0],
            bonded_atom_pairs=None,
            user_ccd=None,
        )

    def test_ref_structure_uses_threadpool_when_workers_set(self) -> None:
        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation
        from alphafold3.model import features as af3_features

        _RecordingExecutor.constructed_max_workers = []
        cache = getattr(af3_features, "_REFSTRUCTURE_REFERENCE_CACHE", None)
        if cache is not None:
            cache.clear()
        fold_input = self._make_fold_input(seq="A" * 10)
        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)

        with mock.patch.dict(
            os.environ,
            {
                "ALPHAFOLD3_REFSTRUCTURE_WORKERS": "2",
                # Ensure this assertion remains stable even if other tests enable caching.
                "ALPHAFOLD3_REFSTRUCTURE_CACHE": "0",
            },
            clear=False,
        ):
            with mock.patch("concurrent.futures.ThreadPoolExecutor", _RecordingExecutor):
                featurisation.featurise_input(
                    fold_input=fold_input,
                    ccd=ccd,
                    buckets=None,
                    verbose=False,
                )

        self.assertIn(2, _RecordingExecutor.constructed_max_workers)

    def test_msa_cache_avoids_reparsing_a3m(self) -> None:
        import numpy as np

        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation
        from alphafold3.data import msa as msa_module
        from alphafold3.model import features as af3_features

        fold_input = self._make_two_chain_fold_input(seq_a="A" * 10, seq_b="C" * 10)
        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)

        cache = getattr(af3_features, "_MSA_FEATURES_CACHE", None)
        if cache is not None:
            cache.clear()

        with mock.patch.dict(
            os.environ,
            {
                "ALPHAFOLD3_REFSTRUCTURE_WORKERS": "1",
                "ALPHAFOLD3_REFSTRUCTURE_CACHE": "0",
                "ALPHAFOLD3_MSA_CACHE": "1",
                "ALPHAFOLD3_MSA_CACHE_MAX_ENTRIES": "16",
            },
            clear=False,
        ):
            with mock.patch(
                "alphafold3.data.msa.Msa.from_a3m", wraps=msa_module.Msa.from_a3m
            ) as from_a3m:
                first = featurisation.featurise_input(
                    fold_input=fold_input,
                    ccd=ccd,
                    buckets=None,
                    verbose=False,
                )[0]
                second = featurisation.featurise_input(
                    fold_input=fold_input,
                    ccd=ccd,
                    buckets=None,
                    verbose=False,
                )[0]

        # Two chains, (unpaired + paired) per chain.
        self.assertEqual(from_a3m.call_count, 4)

        for key in ("msa", "msa_mask", "deletion_matrix", "profile", "deletion_mean"):
            np.testing.assert_array_equal(second[key], first[key], err_msg=key)

    def test_parallel_ref_structure_matches_sequential(self) -> None:
        import numpy as np

        from alphafold3.constants import chemical_components
        from alphafold3.data import featurisation
        from alphafold3.model import features as af3_features

        fold_input = self._make_fold_input(seq="A" * 10)
        ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)

        cache = getattr(af3_features, "_REFSTRUCTURE_REFERENCE_CACHE", None)
        if cache is not None:
            cache.clear()

        with mock.patch.dict(
            os.environ,
            {
                "ALPHAFOLD3_REFSTRUCTURE_WORKERS": "1",
                "ALPHAFOLD3_REFSTRUCTURE_CACHE": "0",
            },
            clear=False,
        ):
            baseline = featurisation.featurise_input(
                fold_input=fold_input,
                ccd=ccd,
                buckets=None,
                verbose=False,
            )[0]

        with mock.patch.dict(
            os.environ,
            {
                "ALPHAFOLD3_REFSTRUCTURE_WORKERS": "2",
                "ALPHAFOLD3_REFSTRUCTURE_CACHE": "0",
            },
            clear=False,
        ):
            parallel = featurisation.featurise_input(
                fold_input=fold_input,
                ccd=ccd,
                buckets=None,
                verbose=False,
            )[0]

        for key in (
            "ref_mask",
            "ref_element",
            "ref_charge",
            "ref_atom_name_chars",
            "ref_space_uid",
        ):
            np.testing.assert_array_equal(parallel[key], baseline[key], err_msg=key)
        np.testing.assert_array_equal(parallel["ref_pos"], baseline["ref_pos"], err_msg="ref_pos")
