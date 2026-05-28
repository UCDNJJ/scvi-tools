from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch import LightningDataModule
from scipy.sparse import csr_matrix, issparse, vstack
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, get_worker_info

import scvi
from scvi import REGISTRY_KEYS
from scvi.dataloaders._samplers import BatchDistributedSampler
from scvi.model._utils import parse_device_args
from scvi.utils import dependencies

if TYPE_CHECKING:
    from typing import Any

    import lamindb as ln
    import pandas as pd
    import tiledbsoma as soma
    from anndata import AnnData


def _normalize_h5_index_key(index_key: object) -> str:
    if isinstance(index_key, bytes):
        return index_key.decode()
    return str(index_key)


def _maybe_python_scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value


def _scan_h5ad_metadata(
    path: str, obs_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    try:
        from anndata.io import read_elem
    except ImportError:  # pragma: no cover
        from anndata.experimental import read_elem

    with h5py.File(path, "r") as handle:
        var_group = handle["var"]
        var_index_key = _normalize_h5_index_key(var_group.attrs.get("_index", "_index"))
        var_names = np.asarray(read_elem(var_group[var_index_key]), dtype=object)

        obs_group = handle["obs"]
        obs_index_key = _normalize_h5_index_key(obs_group.attrs.get("_index", "_index"))
        obs_names = np.asarray(read_elem(obs_group[obs_index_key]), dtype=object).astype(str)

        obs_data = {}
        for column in obs_columns:
            if column in obs_group:
                obs_data[column] = np.asarray(read_elem(obs_group[column]))
            else:
                obs_data[column] = np.full(len(obs_names), None, dtype=object)

    return var_names, obs_names, obs_data


class MappedCollectionDataModule(LightningDataModule):
    @dependencies("lamindb")
    def __init__(
        self,
        collection: ln.Collection,
        batch_key: str | None = None,
        label_key: str | None = None,
        unlabeled_category: str | None = "Unknown",
        sample_key: str | None = None,
        batch_size: int = 128,
        collection_val: ln.Collection | None = None,
        accelerator: str = "auto",
        device: int | str = "auto",
        shuffle: bool = True,
        model_name: str = "SCVI",
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._batch_key = batch_key
        self._label_key = label_key
        self._sample_key = sample_key
        self.model_name = model_name
        self.shuffle = shuffle
        self.unlabeled_category = unlabeled_category
        self._parallel = kwargs.pop("parallel", True)
        self.labels_ = None
        self.samples_ = None
        self._categorical_covariate_keys = categorical_covariate_keys
        self._continuous_covariate_keys = continuous_covariate_keys

        # here we initialize MappedCollection to use in a pytorch DataLoader
        obs_keys = self._batch_key  # we must have batch keys
        if self._label_key is not None:
            obs_keys = [obs_keys] + [self._label_key]
        if self._sample_key is not None:
            obs_keys = [obs_keys] + [self._sample_key]
        if self._categorical_covariate_keys is not None:
            obs_keys = (
                obs_keys + self._categorical_covariate_keys
                if type(obs_keys).__name__ == "list"
                else [obs_keys] + self._categorical_covariate_keys
            )
        if self._continuous_covariate_keys is not None:
            obs_keys = (
                obs_keys + self._continuous_covariate_keys
                if type(obs_keys).__name__ == "list"
                else [obs_keys] + self._continuous_covariate_keys
            )

        self._dataset = collection.mapped(obs_keys=obs_keys, parallel=self._parallel, **kwargs)
        if collection_val is not None:
            self._validset = collection_val.mapped(
                obs_keys=obs_keys, parallel=self._parallel, **kwargs
            )
        else:
            self._validset = None

        # generate encodings
        if self._label_key is not None:
            self.labels_ = self._dataset.get_merged_labels(self._label_key).astype(str)
        if self._sample_key is not None:
            # CURRENTLY IMPLEMENTED FOR MRVI
            sample_key = self._sample_key
            encoder = self._dataset.encoders[sample_key]

            # Initialize a counter to count per encoded sample index
            from collections import Counter

            sample_counter = Counter()

            # Loop through the raw AnnData artifacts in the collection
            for artifact in collection.artifacts.all():
                adata = artifact.load()
                sample_column = (
                    adata.obs[sample_key].astype(str).values
                )  # Ensure str for encoder mapping
                sample_indices = [encoder[val] for val in sample_column]
                sample_counter.update(sample_indices)

            # Build tensor of counts aligned to encoder indices
            counts = np.zeros(len(encoder), dtype=np.float32)
            for idx, count in sample_counter.items():
                counts[idx] = count

            self.n_obs_per_sample = torch.tensor(counts, dtype=torch.float32)
        else:
            self.n_obs_per_sample = torch.tensor([])
        if self._categorical_covariate_keys is not None:
            self.categorical_covariate_keys_ = [
                self._dataset.encoders[cat_cov_key]
                for cat_cov_key in self._categorical_covariate_keys
            ]

        # need by scvi and lightning.pytorch
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False
        _, _, self.device = parse_device_args(
            accelerator=accelerator, devices=device, return_device="torch"
        )

    def close(self):
        self._dataset.close()
        self._validset.close()

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader_val(shuffle=self.shuffle)

    def inference_dataloader(
        self, shuffle=False, batch_size=4096, indices=None, parallel_cpu_count=None
    ):
        """Dataloader for inference with `on_before_batch_transfer` applied."""
        if shuffle is None:
            shuffle = self.shuffle
        dataloader = self._create_dataloader(shuffle, batch_size, indices, parallel_cpu_count)
        return self._InferenceDataloader(dataloader, self.on_before_batch_transfer)

    def _create_dataloader(self, shuffle, batch_size=None, indices=None, parallel_cpu_count=None):
        if self._parallel:
            if parallel_cpu_count is None:
                num_workers = os.cpu_count() - 1
            else:
                num_workers = parallel_cpu_count
            worker_init_fn = self._dataset.torch_worker_init_fn
        else:
            num_workers = 0
            worker_init_fn = None
        if batch_size is None:
            batch_size = self._batch_size
        if indices is not None:
            dataset = self._dataset[indices]
        else:
            dataset = self._dataset
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
        )

    def _create_dataloader_val(
        self, shuffle, batch_size=None, indices=None, parallel_cpu_count=None
    ):
        if self._validset is not None:
            if self._parallel:
                if parallel_cpu_count is None:
                    num_workers = os.cpu_count() - 1
                else:
                    num_workers = parallel_cpu_count
                worker_init_fn = self._validset.torch_worker_init_fn
            else:
                num_workers = 0
                worker_init_fn = None
            if batch_size is None:
                batch_size = self._batch_size
            if indices is not None:
                validset = self._validset[indices]
            else:
                validset = self._validset
            return DataLoader(
                validset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn,
            )
        else:
            pass

    @property
    def n_obs(self) -> int:
        return self._dataset.n_obs

    @property
    def var_names(self) -> int:
        return self._dataset.var_joint

    @property
    def n_vars(self) -> int:
        return self._dataset.n_vars

    @property
    def n_batch(self) -> int:
        if self._batch_key is None:
            return 1
        return len(self._dataset.encoders[self._batch_key])

    @property
    def n_labels(self) -> int:
        if self._label_key is None:
            return 0
        return len(self.labels)

    @property
    def n_samples(self) -> int:
        if self._sample_key is None:
            return 0
        return len(self.samples)

    @property
    def labels(self) -> np.ndarray:
        if self._label_key is None:
            return None
        combined = np.concatenate(
            (list(self._dataset.encoders[self._label_key].keys()), [self.unlabeled_category])
        )
        unique_values, idx = np.unique(combined, return_index=True)
        unique_values = unique_values[np.argsort(idx)]
        return unique_values.astype(object)

    @property
    def samples(self) -> np.ndarray:
        if self._sample_key is None:
            return None
        combined = list(self._dataset.encoders[self._sample_key].keys())
        unique_values, idx = np.unique(combined, return_index=True)
        unique_values = unique_values[np.argsort(idx)]
        return unique_values.astype(object)

    @property
    def unlabeled_category(self) -> str:
        """String assigned to unlabeled cells."""
        if not hasattr(self, "_unlabeled_category"):
            raise AttributeError("`unlabeled_category` not set.")
        return self._unlabeled_category

    @unlabeled_category.setter
    def unlabeled_category(self, value: str | None):
        if not (value is None or isinstance(value, str)):
            raise ValueError("`unlabeled_category` must be a string or None.")
        self._unlabeled_category = value

    @property
    def extra_categorical_covs(self) -> dict:
        if self._categorical_covariate_keys is None:
            out = {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_categorical_covs": 0},
            }
        else:
            mapping = dict(
                zip(
                    self._categorical_covariate_keys,
                    [np.array(list(x.values())) for x in self.categorical_covariate_keys_],
                    strict=False,
                )
            )
            out = {
                "data_registry": {"attr_key": "_scvi_extra_categorical_covs", "attr_name": "obsm"},
                "state_registry": {
                    "field_keys": self._categorical_covariate_keys,
                    "mapping": mapping,
                    "n_cats_per_key": [len(mapping[map]) for map in mapping.keys()],
                },
                "summary_stats": {
                    "n_extra_categorical_covs": len(self._categorical_covariate_keys)
                },
            }
        return out

    @property
    def extra_continuous_covs(self) -> dict:
        if self._continuous_covariate_keys is None:
            out = {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_continuous_covs": 0},
            }
        else:
            out = {
                "data_registry": {"attr_key": "_scvi_extra_continuous_covs", "attr_name": "obsm"},
                "state_registry": {
                    "columns": np.array(self._continuous_covariate_keys, dtype=object)
                },
                "summary_stats": {"n_extra_continuous_covs": len(self._continuous_covariate_keys)},
            }
        return out

    @property
    def registry(self) -> dict:
        return {
            "scvi_version": scvi.__version__,
            "model_name": self.model_name,
            "setup_args": {
                "layer": None,
                "batch_key": self._batch_key,
                "labels_key": self._label_key,
                "samples_key": self._sample_key,
                "size_factor_key": None,
                "categorical_covariate_keys": self._categorical_covariate_keys,
                "continuous_covariate_keys": self._continuous_covariate_keys,
            },
            "field_registries": {
                "X": {
                    "data_registry": {"attr_name": "X", "attr_key": None},
                    "state_registry": {
                        "n_obs": self.n_obs,
                        "n_vars": self.n_vars,
                        "column_names": self.var_names,
                    },
                    "summary_stats": {"n_vars": self.n_vars, "n_cells": self.n_obs},
                },
                "batch": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_batch"},
                    "state_registry": {
                        "categorical_mapping": self.batch_labels,
                        "original_key": self._batch_key,
                    },
                    "summary_stats": {"n_batch": self.n_batch},
                },
                "labels": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_labels"},
                    "state_registry": {
                        "categorical_mapping": self.labels,
                        "original_key": self._label_key,
                        "unlabeled_category": self.unlabeled_category,
                    },
                    "summary_stats": {"n_labels": self.n_labels},
                },
                "ind_x": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_indices"},
                    "state_registry": {},
                    "summary_stats": {},
                },
                "sample": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_sample"},
                    "state_registry": {
                        "categorical_mapping": self.samples,
                        "original_key": self._sample_key,
                    },
                    "n_obs_per_sample": {"n_obs_per_sample": self.n_obs_per_sample},
                    "summary_stats": {"n_sample": self.n_samples},
                },
                "size_factor": {
                    "data_registry": {},
                    "state_registry": {},
                    "summary_stats": {},
                },
                "extra_categorical_covs": self.extra_categorical_covs,
                "extra_continuous_covs": self.extra_continuous_covs,
            },
            "setup_method_name": "setup_datamodule",
        }

    @property
    def batch_labels(self) -> int | None:
        if self._batch_key is None:
            return None
        return self._dataset.encoders[self._batch_key]

    @property
    def label_keys(self) -> int | None:
        if self._label_key is None:
            return None
        return self._dataset.encoders[self._label_key]

    @property
    def sample_keys(self) -> int | None:
        if self._sample_key is None:
            return None
        return self._dataset.encoders[self._sample_key]

    def on_before_batch_transfer(
        self,
        batch,
        dataloader_idx,
    ):
        X_KEY: str = "X"
        BATCH_KEY: str = "batch"
        LABEL_KEY: str = "labels"
        SAMPLE_KEY: str = "sample"
        CAT_COVS_KEY: str = "extra_categorical_covs"
        CONT_COVS_KEY: str = "extra_continuous_covs"

        return {
            X_KEY: batch["X"].float(),
            BATCH_KEY: batch[self._batch_key][:, None] if self._batch_key is not None else None,
            LABEL_KEY: batch[self._label_key][:, None] if self._label_key is not None else 0,
            CAT_COVS_KEY: torch.cat(
                [batch[k][:, None] for k in self._categorical_covariate_keys], dim=1
            )
            if self._categorical_covariate_keys is not None
            else None,
            CONT_COVS_KEY: torch.cat(
                [batch[k][:, None] for k in self._continuous_covariate_keys], dim=1
            )
            if self._continuous_covariate_keys is not None
            else None,
            SAMPLE_KEY: batch[self._sample_key][:, None] if self._sample_key is not None else None,
        }

    class _InferenceDataloader:
        """Wrapper to apply `on_before_batch_transfer` during iteration."""

        def __init__(self, dataloader, transform_fn):
            self.dataloader = dataloader
            self.transform_fn = transform_fn

        def __iter__(self):
            for batch in self.dataloader:
                yield self.transform_fn(batch, dataloader_idx=None)

        def __len__(self):
            return len(self.dataloader)


class _IndexDataset(Dataset):
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> int:
        return int(self.indices[index])


class _MappedCollectionDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        mapped_indices: np.ndarray,
        public_indices: np.ndarray | None = None,
    ):
        self.dataset = dataset
        self.mapped_indices = np.asarray(mapped_indices, dtype=np.int64)
        self.public_indices = (
            np.asarray(public_indices, dtype=np.int64)
            if public_indices is not None
            else np.arange(len(self.mapped_indices), dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.mapped_indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = dict(self.dataset[int(self.mapped_indices[index])])
        sample["_indices"] = int(self.public_indices[index])
        return sample

    @staticmethod
    def torch_worker_init_fn(worker_id: int) -> None:
        del worker_id
        dataset = get_worker_info().dataset
        mapped = dataset.dataset
        if hasattr(mapped, "parallel"):
            mapped.parallel = False
        if hasattr(mapped, "storages"):
            mapped.storages = []
        if hasattr(mapped, "conns"):
            mapped.conns = []
        make_connections = getattr(mapped, "_make_connections", None)
        path_list = getattr(mapped, "path_list", None)
        if callable(make_connections) and path_list is not None:
            make_connections(path_list, parallel=False)


class _CollectionBackedAnnData:
    def __init__(
        self,
        collection: ln.Collection,
        obs_columns: list[str],
    ):
        self._artifacts = list(collection.artifacts.all())
        self._obs_columns = list(dict.fromkeys(obs_columns))
        self._adatas: list[AnnData | None] = [None] * len(self._artifacts)
        self._adatas_pid: int | None = None
        self._artifact_n_obs: list[int] = [0] * len(self._artifacts)
        self.obs_to_location: dict[str, tuple[int, int]] = {}
        self.obs_metadata: dict[str, dict[str, object]] = {}
        self.var_names: np.ndarray | None = None

        for artifact_idx, artifact in enumerate(self._artifacts):
            path = self._get_artifact_path(artifact)
            if path is not None and os.fspath(path).lower().endswith(".h5ad"):
                var_names, obs_names, obs_data = _scan_h5ad_metadata(path, self._obs_columns)
                self._set_var_names(var_names)
                self._register_obs_from_arrays(artifact_idx, obs_names, obs_data)
                continue

            adata = self._open_artifact(artifact)
            try:
                self._register_var_names(adata)
                self._register_obs(artifact_idx, adata)
            finally:
                self._close_adata(adata)

        if self.var_names is None:
            self.var_names = np.asarray([], dtype=object)

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    def close(self) -> None:
        for adata in self._adatas:
            self._close_adata(adata)
        self._adatas = [None] * len(self._artifacts)
        self._adatas_pid = None

    def fetch_rows(
        self,
        obs_names: np.ndarray,
        densify: bool = True,
    ) -> np.ndarray | csr_matrix:
        out = np.zeros((len(obs_names), self.n_vars), dtype=np.float32)
        sparse_rows: list[csr_matrix] | None = (
            [csr_matrix((1, self.n_vars), dtype=np.float32) for _ in range(len(obs_names))]
            if not densify
            else None
        )
        row_positions: dict[int, list[int]] = defaultdict(list)
        row_indices: dict[int, list[int]] = defaultdict(list)
        for out_idx, obs_name in enumerate(obs_names):
            location = self.obs_to_location.get(str(obs_name))
            if location is None:
                continue
            artifact_idx, row_idx = location
            row_positions[artifact_idx].append(out_idx)
            row_indices[artifact_idx].append(row_idx)

        close_after_fetch = self._is_worker_process()
        for artifact_idx, positions in row_positions.items():
            adata = self._get_adata(artifact_idx)
            try:
                artifact_row_indices = np.asarray(row_indices[artifact_idx], dtype=np.int64)
                sort_order = np.argsort(artifact_row_indices)
                sorted_row_indices = artifact_row_indices[sort_order]
                matrix = adata[sorted_row_indices].X
                matrix_is_sparse = issparse(matrix)
                if matrix_is_sparse:
                    matrix = matrix.tocsr()
                else:
                    matrix = np.asarray(matrix)
                inverse_order = np.empty_like(sort_order)
                inverse_order[sort_order] = np.arange(len(sort_order))
                matrix = matrix[inverse_order]
                output_positions = np.asarray(positions, dtype=np.int64)
                if sparse_rows is not None and matrix_is_sparse:
                    for row_idx, output_position in enumerate(output_positions):
                        sparse_rows[output_position] = matrix[row_idx].astype(
                            np.float32, copy=False
                        )
                else:
                    if sparse_rows is not None:
                        out = (
                            vstack(sparse_rows, format="csr")
                            .toarray()
                            .astype(np.float32, copy=False)
                        )
                        sparse_rows = None
                    if issparse(matrix):
                        matrix = matrix.toarray()
                    out[output_positions] = np.asarray(matrix, dtype=np.float32)
            finally:
                if close_after_fetch:
                    self._close_adata(adata)
        if sparse_rows is not None:
            return vstack(sparse_rows, format="csr")
        return out

    @staticmethod
    def _open_artifact(artifact) -> AnnData:
        path = _CollectionBackedAnnData._get_artifact_path(artifact)
        if path is not None:
            return ad.read_h5ad(path, backed="r")
        return artifact.load()

    @staticmethod
    def _get_artifact_path(artifact) -> str | None:
        for attr in ("path", "local_filepath"):
            value = getattr(artifact, attr, None)
            if value is not None:
                return os.fspath(value)
        cache = getattr(artifact, "cache", None)
        if callable(cache):
            try:
                cached = cache()
            except TypeError:
                cached = None
            if cached is not None:
                return os.fspath(cached)
        return None

    def _register_var_names(self, adata: AnnData) -> None:
        self._set_var_names(np.asarray(adata.var_names, dtype=object))

    def _set_var_names(self, var_names: np.ndarray) -> None:
        if self.var_names is None:
            self.var_names = var_names
        elif not np.array_equal(self.var_names, var_names):
            raise ValueError(
                "All artifacts in a modality collection must share the same var_names."
            )

    def _register_obs(self, artifact_idx: int, adata: AnnData) -> None:
        self._artifact_n_obs[artifact_idx] = adata.n_obs
        obs_frame = adata.obs.reindex(columns=self._obs_columns, fill_value=None)
        for row_idx, obs_name in enumerate(adata.obs_names.astype(str)):
            if obs_name in self.obs_to_location:
                raise ValueError(f"Duplicate obs_name {obs_name!r} found in collection.")
            self.obs_to_location[obs_name] = (artifact_idx, row_idx)
            self.obs_metadata[obs_name] = {
                key: value.item() if hasattr(value, "item") else value
                for key, value in obs_frame.iloc[row_idx].to_dict().items()
            }

    def _register_obs_from_arrays(
        self,
        artifact_idx: int,
        obs_names: np.ndarray,
        obs_data: dict[str, np.ndarray],
    ) -> None:
        self._artifact_n_obs[artifact_idx] = len(obs_names)
        seen_obs_names = set(self.obs_to_location)
        for obs_name in obs_names:
            if obs_name in seen_obs_names:
                raise ValueError(f"Duplicate obs_name {obs_name!r} found in collection.")
            seen_obs_names.add(obs_name)

        self.obs_to_location.update(
            {
                obs_name: (artifact_idx, row_idx)
                for row_idx, obs_name in enumerate(obs_names)
            }
        )
        if not self._obs_columns:
            self.obs_metadata.update({obs_name: {} for obs_name in obs_names})
            return
        self.obs_metadata.update(
            {
                obs_name: {
                    key: _maybe_python_scalar(value)
                    for key, value in zip(self._obs_columns, row_values, strict=True)
                }
                for obs_name, row_values in zip(
                    obs_names,
                    zip(*(obs_data[key] for key in self._obs_columns), strict=True),
                    strict=True,
                )
            }
        )

    @staticmethod
    def _is_worker_process() -> bool:
        return get_worker_info() is not None

    def _get_adata(self, artifact_idx: int) -> AnnData:
        current_pid = os.getpid()
        if self._adatas_pid != current_pid:
            self.close()
            self._adatas_pid = current_pid
        if self._is_worker_process():
            return self._open_artifact(self._artifacts[artifact_idx])
        adata = self._adatas[artifact_idx]
        if adata is None:
            adata = self._open_artifact(self._artifacts[artifact_idx])
            self._adatas[artifact_idx] = adata
        return adata

    @staticmethod
    def _close_adata(adata: AnnData | None) -> None:
        if adata is None:
            return
        file_manager = getattr(adata, "file", None)
        if file_manager is not None:
            try:
                file_manager.close()
            except (AttributeError, OSError, ValueError):
                pass


class MultiVIMappedCollectionDataModule(LightningDataModule):
    """Data module for training :class:`~scvi.model.MULTIVI` from Lamin collections.

    The module reads RNA and/or ATAC AnnData artifacts from Lamin collections and exposes
    a datamodule interface compatible with :meth:`~scvi.model.MULTIVI.train`.

    Parameters
    ----------
    rna_collection
        Lamin collection containing RNA AnnData artifacts. Can be ``None`` when training
        from ATAC-only data.
    atac_collection
        Lamin collection containing ATAC AnnData artifacts. Can be ``None`` when training
        from RNA-only data.
    batch_key
        Optional obs key used for batches.
    batch_size
        Minibatch size.
    shuffle
        Whether to shuffle the training dataloader.
    categorical_covariate_keys
        Optional list of obs keys for categorical covariates.
    continuous_covariate_keys
        Optional list of obs keys for continuous covariates.
    sparse_atac
        If ``True``, preserve sparse ATAC batches from the collection and emit
        :func:`torch.sparse_csr_tensor` in the collate function when possible. For ATAC peak
        matrices with hundreds of thousands of regions, dense per-batch tensors can exceed GPU
        memory. Keeping ATAC sparse end-to-end avoids materializing a
        ``(batch_size, n_regions)`` dense tensor on the CPU side. Set ``sparse_atac=False`` if
        your model's ATAC encoder requires dense input. Some model paths may densify on-device.
    sparse_rna
        If ``True``, preserve sparse RNA batches and emit :func:`torch.sparse_csr_tensor` when
        possible.
    drop_dataset_tail
        When running under DDP, whether to drop the tail of the dataset to make it
        evenly divisible by the number of replicas. Passed to
        :class:`~scvi.dataloaders.BatchDistributedSampler`.
    drop_last
        When running under DDP, whether to drop the last incomplete batch per replica.
        Passed to :class:`~scvi.dataloaders.BatchDistributedSampler`.
    num_workers
        Number of worker processes for the :class:`~torch.utils.data.DataLoader`. Setting
        ``num_workers > 0`` can significantly speed up training by overlapping data loading
        with model computation. Default is ``0`` (single-process loading).
    pin_memory
        If ``True``, the data loader will copy tensors into CUDA pinned memory before
        returning them. Only effective when using a GPU. Default is ``False``.
    persistent_workers
        If ``True``, worker processes will not be shut down between epochs. Forced to
        ``False`` when ``num_workers == 0``. Default is ``False``.
    prefetch_factor
        Number of batches loaded in advance by each worker. Only passed to the
        :class:`~torch.utils.data.DataLoader` when ``num_workers > 0``. Default is ``None``.

    Notes
    -----
    At least one of ``rna_collection`` or ``atac_collection`` must be provided.

    When PyTorch Distributed Data Parallel (DDP) is active (i.e.
    :func:`torch.distributed.is_initialized` returns ``True``),
    :class:`~scvi.dataloaders.BatchDistributedSampler` is used automatically so that
    each rank sees a non-overlapping subset of the data.

    Examples
    --------
    >>> import lamindb as ln
    >>> from scvi.dataloaders import MultiVIMappedCollectionDataModule
    >>> from scvi.model import MULTIVI
    >>> atac_collection = ln.Collection.get(name="my_atac_collection")
    >>> datamodule = MultiVIMappedCollectionDataModule(
    ...     rna_collection=None,
    ...     atac_collection=atac_collection,
    ...     batch_key="batch",
    ...     batch_size=128,
    ... )
    >>> model = MULTIVI(adata=None, registry=datamodule.registry, modality_weights="equal")
    >>> model.train(datamodule=datamodule, max_epochs=10)
    >>> latent = model.get_latent_representation(
    ...     dataloader=datamodule.inference_dataloader(batch_size=128)
    ... )
    """

    @dependencies("lamindb")
    def __init__(
        self,
        rna_collection: ln.Collection | None = None,
        atac_collection: ln.Collection | None = None,
        batch_key: str | None = None,
        batch_size: int = 128,
        shuffle: bool = True,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        sparse_atac: bool = False,
        sparse_rna: bool = False,
        drop_dataset_tail: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
    ):
        super().__init__()
        if rna_collection is None and atac_collection is None:
            raise ValueError(
                "At least one of `rna_collection` or `atac_collection` must be provided."
            )
        self._batch_key = batch_key
        self._batch_size = batch_size
        self.shuffle = shuffle
        self._drop_dataset_tail = drop_dataset_tail
        self._drop_last = drop_last
        self.model_name = "MULTIVI"
        self._categorical_covariate_keys = categorical_covariate_keys
        self._continuous_covariate_keys = continuous_covariate_keys
        self._sparse_atac = sparse_atac
        self._sparse_rna = sparse_rna
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._prefetch_factor = prefetch_factor
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

        obs_columns = [batch_key] if batch_key is not None else []
        if categorical_covariate_keys is not None:
            obs_columns.extend(categorical_covariate_keys)
        if continuous_covariate_keys is not None:
            obs_columns.extend(continuous_covariate_keys)

        self._mapped_atac_dataset = None
        self._mapped_atac_indices = None

        self._rna_source = (
            _CollectionBackedAnnData(rna_collection, obs_columns)
            if rna_collection is not None
            else None
        )
        self._atac_source = (
            _CollectionBackedAnnData(atac_collection, obs_columns)
            if atac_collection is not None
            else None
        )
        self._sources = tuple(
            source for source in (self._rna_source, self._atac_source) if source is not None
        )
        obs_name_sets = [set(source.obs_to_location) for source in self._sources]

        self._global_obs_names = np.array(
            sorted(set.union(*obs_name_sets)),
            dtype=object,
        )
        self._metadata = self._build_metadata()
        self._init_encoders()
        if rna_collection is None and atac_collection is not None:
            self._mapped_atac_dataset = atac_collection.mapped(parallel=True)
            self._mapped_atac_indices = self._build_mapped_indices(self._atac_source)
            self._dataset = _MappedCollectionDataset(
                self._mapped_atac_dataset,
                self._mapped_atac_indices,
            )
        else:
            self._dataset = _IndexDataset(np.arange(self.n_obs, dtype=np.int64))

    def close(self):
        if self._mapped_atac_dataset is not None:
            self._mapped_atac_dataset.close()
        for source in self._sources:
            source.close()

    @property
    def n_obs(self) -> int:
        return len(self._global_obs_names)

    @property
    def var_names(self) -> np.ndarray:
        if self._rna_source is None:
            return np.asarray([], dtype=object)
        return self._rna_source.var_names

    @property
    def n_vars(self) -> int:
        if self._rna_source is None:
            return 0
        return self._rna_source.n_vars

    @property
    def region_names(self) -> np.ndarray:
        if self._atac_source is None:
            return np.asarray([], dtype=object)
        return self._atac_source.var_names

    @property
    def n_regions(self) -> int:
        if self._atac_source is None:
            return 0
        return self._atac_source.n_vars

    @property
    def n_batch(self) -> int:
        return 1 if self._batch_key is None else len(self.batch_labels)

    @property
    def n_labels(self) -> int:
        return 1

    @property
    def n_cats_per_cov(self) -> list[int]:
        if self._categorical_covariate_keys is None:
            return []
        return [len(encoder.classes_) for encoder in self._categorical_covariate_encoders]

    @property
    def n_continuous_cov(self) -> int:
        if self._continuous_covariate_keys is None:
            return 0
        return len(self._continuous_covariate_keys)

    @property
    def batch_labels(self) -> np.ndarray:
        if self._batch_key is None:
            return np.array(["batch_0"], dtype=object)
        return self._batch_encoder.classes_.astype(object)

    @property
    def labels(self) -> np.ndarray:
        return np.array(["label_0"], dtype=object)

    @property
    def extra_categorical_covs(self) -> dict:
        if self._categorical_covariate_keys is None:
            return {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_categorical_covs": 0},
            }
        mapping = dict(
            zip(
                self._categorical_covariate_keys,
                [
                    encoder.classes_.astype(object)
                    for encoder in self._categorical_covariate_encoders
                ],
                strict=True,
            )
        )
        return {
            "data_registry": {"attr_key": "_scvi_extra_categorical_covs", "attr_name": "obsm"},
            "state_registry": {
                "field_keys": self._categorical_covariate_keys,
                "mapping": mapping,
                "n_cats_per_key": self.n_cats_per_cov,
            },
            "summary_stats": {"n_extra_categorical_covs": len(self._categorical_covariate_keys)},
        }

    @property
    def extra_continuous_covs(self) -> dict:
        if self._continuous_covariate_keys is None:
            return {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_continuous_covs": 0},
            }
        return {
            "data_registry": {"attr_key": "_scvi_extra_continuous_covs", "attr_name": "obsm"},
            "state_registry": {"columns": np.array(self._continuous_covariate_keys, dtype=object)},
            "summary_stats": {"n_extra_continuous_covs": len(self._continuous_covariate_keys)},
        }

    @property
    def registry(self) -> dict:
        return {
            "scvi_version": scvi.__version__,
            "model_name": self.model_name,
            "setup_args": {
                "layer": None,
                "batch_key": self._batch_key,
                "labels_key": None,
                "size_factor_key": None,
                "categorical_covariate_keys": self._categorical_covariate_keys,
                "continuous_covariate_keys": self._continuous_covariate_keys,
            },
            "field_registries": {
                REGISTRY_KEYS.X_KEY: {
                    "data_registry": {"attr_name": "X", "attr_key": None},
                    "state_registry": {
                        "n_obs": self.n_obs,
                        "n_vars": self.n_vars,
                        "column_names": self.var_names,
                    },
                    "summary_stats": {"n_vars": self.n_vars, "n_cells": self.n_obs},
                },
                REGISTRY_KEYS.ATAC_X_KEY: {
                    "data_registry": {"attr_name": "obsm", "attr_key": REGISTRY_KEYS.ATAC_X_KEY},
                    "state_registry": {
                        "n_obs": self.n_obs,
                        "n_vars": self.n_regions,
                        "column_names": self.region_names,
                    },
                    "summary_stats": {"n_atac": self.n_regions},
                },
                REGISTRY_KEYS.BATCH_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_batch"},
                    "state_registry": {
                        "categorical_mapping": self.batch_labels,
                        "original_key": self._batch_key,
                    },
                    "summary_stats": {"n_batch": self.n_batch},
                },
                REGISTRY_KEYS.LABELS_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_labels"},
                    "state_registry": {
                        "categorical_mapping": self.labels,
                        "original_key": None,
                        "unlabeled_category": None,
                    },
                    "summary_stats": {"n_labels": self.n_labels},
                },
                REGISTRY_KEYS.INDICES_KEY: {
                    "data_registry": {"attr_name": "obs", "attr_key": "_indices"},
                    "state_registry": {},
                    "summary_stats": {},
                },
                REGISTRY_KEYS.SIZE_FACTOR_KEY: {
                    "data_registry": {},
                    "state_registry": {},
                    "summary_stats": {},
                },
                REGISTRY_KEYS.CAT_COVS_KEY: self.extra_categorical_covs,
                REGISTRY_KEYS.CONT_COVS_KEY: self.extra_continuous_covs,
            },
            "setup_method_name": "setup_datamodule",
        }

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self._dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader | None:
        return None

    def inference_dataloader(
        self,
        shuffle: bool = False,
        batch_size: int | None = None,
        indices=None,
    ):
        if indices is None:
            dataset = self._dataset
        elif self._mapped_atac_dataset is not None:
            indices = np.asarray(indices, dtype=np.int64)
            dataset = _MappedCollectionDataset(
                self._mapped_atac_dataset,
                self._mapped_atac_indices[indices],
                public_indices=indices,
            )
        else:
            dataset = _IndexDataset(np.asarray(indices, dtype=np.int64))
        return self._create_dataloader(dataset, shuffle=shuffle, batch_size=batch_size)

    def _build_metadata(self) -> dict[str, dict[str, object]]:
        metadata: dict[str, dict[str, object]] = {}
        for obs_name in self._global_obs_names:
            values = {}
            for source in self._sources:
                source_values = source.obs_metadata.get(obs_name)
                if source_values is None:
                    continue
                for key, value in source_values.items():
                    if key not in values or values[key] is None:
                        values[key] = value
                    elif value is not None and values[key] != value:
                        raise ValueError(
                            f"Conflicting obs metadata for {obs_name!r} and key {key!r}."
                        )
            metadata[obs_name] = values
        return metadata

    def _init_encoders(self) -> None:
        if self._batch_key is None:
            self._batch_codes = np.zeros(self.n_obs, dtype=np.int64)
            self._batch_encoder = None
        else:
            batch_values = np.array(
                [
                    str(self._metadata[obs_name][self._batch_key])
                    for obs_name in self._global_obs_names
                ],
                dtype=object,
            )
            self._batch_encoder = LabelEncoder().fit(batch_values)
            self._batch_codes = self._batch_encoder.transform(batch_values).astype(np.int64)

        if self._categorical_covariate_keys is None:
            self._categorical_covariate_encoders = []
            self._categorical_covariates = None
        else:
            self._categorical_covariate_encoders = []
            cat_covs = np.zeros(
                (self.n_obs, len(self._categorical_covariate_keys)),
                dtype=np.int64,
            )
            for cov_idx, key in enumerate(self._categorical_covariate_keys):
                values = np.array(
                    [str(self._metadata[obs_name][key]) for obs_name in self._global_obs_names],
                    dtype=object,
                )
                encoder = LabelEncoder().fit(values)
                self._categorical_covariate_encoders.append(encoder)
                cat_covs[:, cov_idx] = encoder.transform(values)
            self._categorical_covariates = cat_covs

        if self._continuous_covariate_keys is None:
            self._continuous_covariates = None
        else:
            self._continuous_covariates = np.asarray(
                [
                    [
                        float(self._metadata[obs_name][key])
                        for key in self._continuous_covariate_keys
                    ]
                    for obs_name in self._global_obs_names
                ],
                dtype=np.float32,
            )

    def _create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool,
        batch_size: int | None = None,
    ) -> DataLoader:
        if batch_size is None:
            batch_size = self._batch_size
        num_workers = getattr(self, "_num_workers", 0)
        dataloader_kwargs = {
            "collate_fn": self._collate_fn,
            "num_workers": num_workers,
            "pin_memory": getattr(self, "_pin_memory", False),
            "persistent_workers": (
                getattr(self, "_persistent_workers", False) if num_workers > 0 else False
            ),
        }
        if num_workers > 0 and isinstance(dataset, _MappedCollectionDataset):
            dataloader_kwargs["worker_init_fn"] = dataset.torch_worker_init_fn
        prefetch_factor = getattr(self, "_prefetch_factor", None)
        if num_workers > 0 and prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        if dist.is_available() and dist.is_initialized():
            sampler = BatchDistributedSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=self._drop_last,
                drop_dataset_tail=self._drop_dataset_tail,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                **dataloader_kwargs,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **dataloader_kwargs,
        )

    def _fetch_modality_tensor(
        self,
        source: _CollectionBackedAnnData | None,
        obs_names: np.ndarray,
        sparse: bool,
    ) -> torch.Tensor:
        if source is None:
            return torch.zeros((len(obs_names), 0), dtype=torch.float32)
        matrix = source.fetch_rows(obs_names, densify=not sparse)
        if sparse and issparse(matrix):
            csr = matrix.tocsr()
            return torch.sparse_csr_tensor(
                crow_indices=torch.from_numpy(csr.indptr.astype(np.int64, copy=False)),
                col_indices=torch.from_numpy(csr.indices.astype(np.int64, copy=False)),
                values=torch.from_numpy(csr.data.astype(np.float32, copy=False)),
                size=csr.shape,
            )
        if issparse(matrix):
            matrix = matrix.toarray()
        return torch.from_numpy(np.asarray(matrix, dtype=np.float32))

    def _build_mapped_indices(self, source: _CollectionBackedAnnData | None) -> np.ndarray:
        if source is None:
            return np.array([], dtype=np.int64)
        offsets = np.cumsum([0, *source._artifact_n_obs[:-1]], dtype=np.int64)
        return np.asarray(
            [
                offsets[artifact_idx] + row_idx
                for artifact_idx, row_idx in (
                    source.obs_to_location[obs_name] for obs_name in self._global_obs_names
                )
            ],
            dtype=np.int64,
        )

    def _collate_mapped_atac_only_batch(
        self, batch: list[dict[str, object]]
    ) -> dict[str, torch.Tensor | None]:
        indices = np.asarray([item["_indices"] for item in batch], dtype=np.int64)
        if self._sparse_atac:
            rows = [item["X"] for item in batch]
            if any(issparse(row) for row in rows):
                atac_rows = []
                for row in rows:
                    if issparse(row):
                        atac_rows.append(row.tocsr())
                    else:
                        atac_rows.append(
                            csr_matrix(np.asarray(row, dtype=np.float32).reshape(1, -1))
                        )
                atac_csr = vstack(atac_rows, format="csr")
            else:
                atac_csr = csr_matrix(np.asarray(rows, dtype=np.float32))
            atac_tensor = torch.sparse_csr_tensor(
                crow_indices=torch.from_numpy(atac_csr.indptr.astype(np.int64, copy=False)),
                col_indices=torch.from_numpy(atac_csr.indices.astype(np.int64, copy=False)),
                values=torch.from_numpy(atac_csr.data.astype(np.float32, copy=False)),
                size=atac_csr.shape,
            )
        else:
            atac_matrix = np.asarray(
                [
                    (
                        row.toarray().ravel()
                        if issparse(row := item["X"])
                        else np.asarray(row).ravel()
                    )
                    for item in batch
                ],
                dtype=np.float32,
            )
            atac_tensor = torch.from_numpy(atac_matrix)
        return {
            REGISTRY_KEYS.X_KEY: torch.zeros((len(indices), 0), dtype=torch.float32),
            REGISTRY_KEYS.ATAC_X_KEY: atac_tensor,
            "atac_X": atac_tensor,
            REGISTRY_KEYS.BATCH_KEY: torch.from_numpy(self._batch_codes[indices][:, None]),
            REGISTRY_KEYS.LABELS_KEY: torch.zeros((len(indices), 1), dtype=torch.int64),
            REGISTRY_KEYS.INDICES_KEY: torch.from_numpy(indices[:, None]),
            REGISTRY_KEYS.CAT_COVS_KEY: (
                torch.from_numpy(self._categorical_covariates[indices])
                if self._categorical_covariates is not None
                else None
            ),
            REGISTRY_KEYS.CONT_COVS_KEY: (
                torch.from_numpy(self._continuous_covariates[indices])
                if self._continuous_covariates is not None
                else None
            ),
        }

    def _collate_fn(
        self, batch_indices: list[int] | list[dict[str, object]]
    ) -> dict[str, torch.Tensor | None]:
        if batch_indices and isinstance(batch_indices[0], dict):
            return self._collate_mapped_atac_only_batch(batch_indices)
        indices = np.asarray(batch_indices, dtype=np.int64)
        obs_names = self._global_obs_names[indices]
        rna_tensor = self._fetch_modality_tensor(
            self._rna_source, obs_names, sparse=self._sparse_rna
        )
        atac_tensor = self._fetch_modality_tensor(
            self._atac_source, obs_names, sparse=self._sparse_atac
        )
        batch = {
            REGISTRY_KEYS.X_KEY: rna_tensor,
            REGISTRY_KEYS.ATAC_X_KEY: atac_tensor,
            # Keep the explicit `atac_X` alias for the custom-dataloader contract alongside the
            # canonical REGISTRY_KEYS.ATAC_X_KEY (`atac`) key used internally by MULTIVAE.
            "atac_X": atac_tensor,
            REGISTRY_KEYS.BATCH_KEY: torch.from_numpy(self._batch_codes[indices][:, None]),
            REGISTRY_KEYS.LABELS_KEY: torch.zeros((len(indices), 1), dtype=torch.int64),
            REGISTRY_KEYS.INDICES_KEY: torch.from_numpy(indices[:, None]),
            REGISTRY_KEYS.CAT_COVS_KEY: (
                torch.from_numpy(self._categorical_covariates[indices])
                if self._categorical_covariates is not None
                else None
            ),
            REGISTRY_KEYS.CONT_COVS_KEY: (
                torch.from_numpy(self._continuous_covariates[indices])
                if self._continuous_covariates is not None
                else None
            ),
        }
        return batch


class TileDBDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for training scVI models from SOMA data

    Wraps a `tiledbsoma_ml.ExperimentDataset` to stream the results of a SOMA
    `ExperimentAxisQuery`, exposing a `DataLoader` to generate tensors ready for scVI model
    training. Also handles deriving the scVI batch label as a tuple of obs columns.
    """

    @dependencies("tiledbsoma")
    def __init__(
        self,
        query: soma.ExperimentAxisQuery,
        *args,
        batch_column_names: list[str] | None = None,
        batch_labels: list[str] | None = None,
        label_keys: list[str] | None = None,
        unlabeled_category: str | None = "Unknown",
        sample_key: list[str] | None = None,
        train_size: float | None = 1.0,
        split_seed: int | None = None,
        dataloader_kwargs: dict[str, Any] | None = None,
        accelerator: str = "auto",
        device: int | str = "auto",
        model_name: str = "SCVI",
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """
        Args:

        query: tiledbsoma.ExperimentAxisQuery
                        Defines the desired result set from a SOMA Experiment.
        *args, **kwargs:
        Additional arguments passed through to `tiledbsoma_ml.ExperimentDataset`.
        batch_column_names: List[str], optional
        List of obs column names, the tuple of which defines the scVI batch label
        (not to be confused with a batch of training data).
        batch_labels: List[str], optional
        List of possible values of the batch label, for mapping to label tensors. By default,
        this will be derived from the unique labels in the given query results (given
        `batch_column_names`), making the label mapping depend on the query. The `batch_labels`
        attribute in the `TileDBDataModule` used for training may be saved and here restored in
        another instance for a different query. That ensures the label mapping will be correct
        for the trained model, even if the second query doesn't return examples of every
        training batch label.
        label_keys
            List of obs column names concatenated to form the label column.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to set up CZI datamodule with scvi.
        %(param_sample_key)s
        train_size
            Fraction of data to use for training.
        split_seed
            Seed for data split.
        dataloader_kwargs: dict, optional
        %(param_accelerator)s
        %(param_device)s
        model_name
            The SCVI-Tools Model we are running
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Keyword arguments passed to `tiledbsoma_ml.experiment_dataloader()`, e.g. `num_workers`.
        """
        super().__init__()
        self.query = query
        self.dataset_args = args
        self.dataset_kwargs = kwargs
        self.dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}
        self.train_size = train_size
        self.split_seed = split_seed
        self.model_name = model_name

        # deal with labels if needed
        self.unlabeled_category = unlabeled_category
        self.label_keys = label_keys
        self.labels_colsep = "//"
        self.label_colname = "_scvi_labels"
        self.labels = None
        self.label_encoder = None
        self.labels_ = None
        self.sample_key = sample_key
        self.sample_colsep = "//"
        self.sample_colname = "_scvi_sample"
        self.samples = None
        self.sample_encoder = None
        self.samples_ = None
        self._categorical_covariate_keys = categorical_covariate_keys
        self._continuous_covariate_keys = continuous_covariate_keys
        self.categ_cov_colsep = "//"
        self._categorical_covariate_colname = "_scvi_cat_cov"

        # deal with batches
        self.batch_column_names = batch_column_names
        self.batch_colsep = "//"
        self.batch_colname = "_scvi_batch"
        # prepare LabelEncoder for the scVI batch label:
        #   1. read obs DataFrame for the whole query result set
        #   2. add scvi_batch column
        #   3. fit LabelEncoder to the scvi_batch column's unique values
        if batch_labels is None:
            cols_sel = (
                self.batch_column_names
                if self.label_keys is None
                else self.batch_column_names + self.label_keys
            )
            cols_sel = (
                cols_sel
                if self._categorical_covariate_keys is None
                else cols_sel + self._categorical_covariate_keys
            )
            cols_sel = (
                cols_sel
                if self._continuous_covariate_keys is None
                else cols_sel + self._continuous_covariate_keys
            )

            obs_df = self.query.obs(column_names=cols_sel).concat().to_pandas()
            obs_df = obs_df[cols_sel]
            self._add_batch_col(obs_df, inplace=True)
            batch_labels = obs_df[self.batch_colname].unique()
        self.batch_labels = batch_labels
        self.batch_encoder = LabelEncoder().fit(self.batch_labels)

        if label_keys is not None:
            obs_label_df = self.query.obs(column_names=self.label_keys).concat().to_pandas()
            obs_label_df = obs_label_df[self.label_keys]
            self._add_label_col(obs_label_df, inplace=True)
            labels = obs_label_df[self.label_colname].unique()
            self.labels = labels
            self.label_encoder = LabelEncoder().fit(self.labels)
            self.labels_ = obs_label_df[self.label_colname].values

        if sample_key is not None:
            obs_sample_df = self.query.obs(column_names=self.sample_key).concat().to_pandas()
            obs_sample_df = obs_sample_df[self.sample_key]
            self._add_sample_col(obs_sample_df, inplace=True)
            samples = obs_sample_df[self.sample_colname].unique()
            self.samples = samples
            self.sample_encoder = LabelEncoder().fit(self.samples)
            self.samples_ = obs_sample_df[self.sample_colname].values
        self.n_obs_per_sample = torch.tensor([])

        if categorical_covariate_keys is not None:
            obs_categ_cov_df = (
                self.query.obs(column_names=self._categorical_covariate_keys).concat().to_pandas()
            )
            obs_categ_cov_df = obs_categ_cov_df[self._categorical_covariate_keys]
            self._add_categ_cov_col(obs_categ_cov_df, inplace=True)
            categ_cov = obs_categ_cov_df[self._categorical_covariate_colname].unique()
            self.categ_cov = categ_cov
            self.categ_cov_encoder = LabelEncoder().fit(self.categ_cov)
            self.categ_cov_ = obs_categ_cov_df[self._categorical_covariate_colname].values

        _, _, self.device = parse_device_args(
            accelerator=accelerator, devices=device, return_device="torch"
        )

    @dependencies("tiledbsoma_ml")
    def setup(self, stage: str | None = None) -> None:
        # Instantiate the ExperimentDataset with the provided args and kwargs.
        from tiledbsoma_ml import ExperimentDataset

        cols_sel = (
            self.batch_column_names
            if self.label_keys is None
            else self.batch_column_names + self.label_keys
        )
        cols_sel = cols_sel if self.sample_key is None else cols_sel + self.sample_key
        cols_sel = (
            cols_sel
            if self._categorical_covariate_keys is None
            else cols_sel + self._categorical_covariate_keys
        )
        cols_sel = (
            cols_sel
            if self._continuous_covariate_keys is None
            else cols_sel + self._continuous_covariate_keys
        )

        self.train_dataset = ExperimentDataset(
            self.query,
            *self.dataset_args,
            obs_column_names=cols_sel,
            **self.dataset_kwargs,
        )

        if self.validation_size > 0.0:
            datapipes = self.train_dataset.random_split(
                self.train_size, self.validation_size, seed=self.split_seed
            )
            self.train_dataset = datapipes[0]
            self.val_dataset = datapipes[1]
        else:
            self.val_dataset = None

    @dependencies("tiledbsoma_ml")
    def train_dataloader(self) -> DataLoader:
        from tiledbsoma_ml import experiment_dataloader

        return experiment_dataloader(
            self.train_dataset,
            **self.dataloader_kwargs,
        )

    @dependencies("tiledbsoma_ml")
    def val_dataloader(self) -> DataLoader:
        from tiledbsoma_ml import experiment_dataloader

        if self.val_dataset is not None:
            return experiment_dataloader(
                self.val_dataset,
                **self.dataloader_kwargs,
            )
        else:
            pass

    def _add_batch_col(self, obs_df: pd.DataFrame, inplace: bool = False):
        # synthesize a new column for obs_df by concatenating the self.batch_column_names columns
        if not inplace:
            obs_df = obs_df.copy()
        obs_df[self.batch_colname] = (
            obs_df[self.batch_column_names].astype(str).agg(self.batch_colsep.join, axis=1)
        )
        if self.labels is not None:
            obs_df[self.label_colname] = (
                obs_df[self.label_keys].astype(str).agg(self.labels_colsep.join, axis=1)
            )
        if self._categorical_covariate_keys is not None:
            obs_df[self._categorical_covariate_colname] = (
                obs_df[self._categorical_covariate_keys]
                .astype(str)
                .agg(self.categ_cov_colsep.join, axis=1)
            )
        return obs_df

    def _add_label_col(self, obs_label_df: pd.DataFrame, inplace: bool = False):
        # synthesize a new column for obs_label_df by concatenating
        # the self.batch_column_names columns
        if not inplace:
            obs_label_df = obs_label_df.copy()
        obs_label_df[self.label_colname] = (
            obs_label_df[self.label_keys].astype(str).agg(self.labels_colsep.join, axis=1)
        )
        return obs_label_df

    def _add_sample_col(self, obs_sample_df: pd.DataFrame, inplace: bool = False):
        # synthesize a new column for obs_label_df by concatenating
        # the self.batch_column_names columns
        if not inplace:
            obs_sample_df = obs_sample_df.copy()
        obs_sample_df[self.sample_colname] = (
            obs_sample_df[self.sample_key].astype(str).agg(self.sample_colsep.join, axis=1)
        )
        return obs_sample_df

    def _add_categ_cov_col(self, obs_categ_cov_df: pd.DataFrame, inplace: bool = False):
        # synthesize a new column for obs_label_df by concatenating
        # the self.batch_column_names columns
        if not inplace:
            obs_categ_cov_df = obs_categ_cov_df.copy()
        obs_categ_cov_df[self._categorical_covariate_colname] = (
            obs_categ_cov_df[self._categorical_covariate_keys]
            .astype(str)
            .agg(self.categ_cov_colsep.join, axis=1)
        )
        return obs_categ_cov_df

    def on_before_batch_transfer(
        self,
        batch,
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor | None]:
        # DataModule hook: transform the ExperimentDataset data batch
        # (X: ndarray, obs_df: DataFrame)
        # into X & batch variable tensors for scVI (using batch_encoder on scvi_batch)
        batch_X, batch_obs = batch
        self._add_batch_col(batch_obs, inplace=True)
        return {
            "X": torch.from_numpy(batch_X).float(),
            "batch": torch.from_numpy(
                self.batch_encoder.transform(batch_obs[self.batch_colname])
            ).unsqueeze(1)
            if self.batch_column_names is not None
            else None,
            "labels": torch.from_numpy(
                self.label_encoder.transform(batch_obs[self.label_colname])
            ).unsqueeze(1)
            if self.label_keys is not None
            else torch.empty(0),
            "extra_categorical_covs": torch.cat(
                [
                    torch.from_numpy(
                        self.categ_cov_encoder.transform(
                            batch_obs[self._categorical_covariate_colname]
                        )
                    ).unsqueeze(1)
                ],
                dim=1,
            )
            if self._categorical_covariate_keys is not None
            else None,
            "extra_continuous_covs": torch.cat(
                [
                    torch.from_numpy(batch_obs[k].values).float().unsqueeze(1)
                    for k in self._continuous_covariate_keys
                ],
                dim=1,
            )
            if self._continuous_covariate_keys is not None
            else None,
            "sample": torch.from_numpy(
                self.sample_encoder.transform(batch_obs[self.sample_colname])
            ).unsqueeze(1)
            if self.sample_key is not None
            else torch.empty(0),
        }

    # scVI code expects these properties on the DataModule:

    @property
    def unlabeled_category(self) -> str:
        """String assigned to unlabeled cells."""
        if not hasattr(self, "_unlabeled_category"):
            raise AttributeError("`unlabeled_category` not set.")
        return self._unlabeled_category

    @unlabeled_category.setter
    def unlabeled_category(self, value: str | None):
        if not (value is None or isinstance(value, str)):
            raise ValueError("`unlabeled_category` must be a string or None.")
        self._unlabeled_category = value

    @property
    def split_seed(self) -> int:
        """Seed for data split."""
        if not hasattr(self, "_split_seed"):
            raise AttributeError("`split_seed` not set.")
        return self._split_seed

    @split_seed.setter
    def split_seed(self, value: int | None):
        if value is not None and not isinstance(value, int):
            raise ValueError("`split_seed` must be an integer.")
        self._split_seed = value or 0

    @property
    def train_size(self) -> float:
        """Fraction of data to use for training."""
        if not hasattr(self, "_train_size"):
            raise AttributeError("`train_size` not set.")
        return self._train_size

    @train_size.setter
    def train_size(self, value: float | None):
        if value is not None and not isinstance(value, float):
            raise ValueError("`train_size` must be a float.")
        elif value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("`train_size` must be between 0.0 and 1.0.")
        self._train_size = value or 1.0

    @property
    def validation_size(self) -> float:
        """Fraction of data to use for validation."""
        if not hasattr(self, "_train_size"):
            raise AttributeError("`validation_size` not available.")
        return 1.0 - self.train_size

    @property
    def n_obs(self) -> int:
        return len(self.query.obs_joinids())

    @property
    def n_vars(self) -> int:
        return len(self.query.var_joinids())

    @property
    def n_batch(self) -> int:
        return len(self.batch_encoder.classes_)

    @property
    def n_labels(self) -> int:
        if self.label_keys is not None:
            return len(self.labels_mapping)
        else:
            return 0

    @property
    def labels_mapping(self) -> list:
        if self.label_keys is not None:
            combined = np.concatenate((self.label_encoder.classes_, [self.unlabeled_category]))
            unique_values, idx = np.unique(combined, return_index=True)
            unique_values = unique_values[np.argsort(idx)]
            return unique_values.astype(object)

    @property
    def n_samples(self) -> int:
        if self.sample_key is not None:
            return len(self.samples_mapping)
        else:
            return 0

    @property
    def samples_mapping(self) -> list:
        if self.sample_key is not None:
            unique_values, idx = np.unique(self.sample_encoder.classes_, return_index=True)
            unique_values = unique_values[np.argsort(idx)]
            return unique_values.astype(object)

    @property
    def extra_categorical_covs(self) -> dict:
        if self._categorical_covariate_keys is None:
            out = {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_categorical_covs": 0},
            }
        else:
            mapping = dict(
                zip(
                    self._categorical_covariate_keys,
                    [self.categ_cov_encoder.classes_],
                    strict=False,
                )
            )
            out = {
                "data_registry": {"attr_key": "_scvi_extra_categorical_covs", "attr_name": "obsm"},
                "state_registry": {
                    "field_keys": self._categorical_covariate_keys,
                    "mapping": mapping,
                    "n_cats_per_key": [len(mapping[map]) for map in mapping.keys()],
                },
                "summary_stats": {
                    "n_extra_categorical_covs": len(self._categorical_covariate_keys)
                },
            }
        return out

    @property
    def extra_continuous_covs(self) -> dict:
        if self._continuous_covariate_keys is None:
            out = {
                "data_registry": {},
                "state_registry": {},
                "summary_stats": {"n_extra_continuous_covs": 0},
            }
        else:
            out = {
                "data_registry": {"attr_key": "_scvi_extra_continuous_covs", "attr_name": "obsm"},
                "state_registry": {
                    "columns": np.array(self._continuous_covariate_keys, dtype=object)
                },
                "summary_stats": {"n_extra_continuous_covs": len(self._continuous_covariate_keys)},
            }
        return out

    @property
    def registry(self) -> dict:
        features_names = list(
            self.query.var_joinids().tolist() if self.query is not None else range(self.n_vars)
        )
        return {
            "scvi_version": scvi.__version__,
            "model_name": self.model_name,
            "setup_args": {
                "layer": None,
                "batch_key": self.batch_colname,
                "labels_key": self.label_keys[0] if self.label_keys is not None else "label",
                "size_factor_key": None,
                "categorical_covariate_keys": self._categorical_covariate_keys,
                "continuous_covariate_keys": self._continuous_covariate_keys,
            },
            "field_registries": {
                "X": {
                    "data_registry": {"attr_name": "X", "attr_key": None},
                    "state_registry": {
                        "n_obs": self.n_obs,
                        "n_vars": self.n_vars,
                        "column_names": [str(i) for i in features_names],
                    },
                    "summary_stats": {"n_vars": self.n_vars, "n_cells": self.n_obs},
                },
                "batch": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_batch"},
                    "state_registry": {
                        "categorical_mapping": self.batch_labels,
                        "original_key": "batch",
                    },
                    "summary_stats": {"n_batch": self.n_batch},
                },
                "labels": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_labels"},
                    "state_registry": {
                        "categorical_mapping": self.labels_mapping,
                        "original_key": self.label_keys[0]
                        if self.label_keys is not None
                        else "label",
                        "unlabeled_category": self.unlabeled_category,
                    },
                    "summary_stats": {"n_labels": self.n_labels},
                },
                "ind_x": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_indices"},
                    "state_registry": {},
                    "summary_stats": {},
                },
                "sample": {
                    "data_registry": {"attr_name": "obs", "attr_key": "_scvi_sample"},
                    "state_registry": {
                        "categorical_mapping": self.samples,
                        "original_key": self.sample_colname,
                    },
                    "n_obs_per_sample": {"n_obs_per_sample": self.n_obs_per_sample},
                    "summary_stats": {"n_sample": self.n_samples},
                },
                "size_factor": {"data_registry": {}, "state_registry": {}, "summary_stats": {}},
                "extra_categorical_covs": self.extra_categorical_covs,
                "extra_continuous_covs": self.extra_continuous_covs,
            },
            "setup_method_name": "setup_datamodule",
        }

    def inference_dataloader(self):
        """Dataloader for inference with `on_before_batch_transfer` applied."""
        dataloader = self.train_dataloader()
        return self._InferenceDataloader(dataloader, self.on_before_batch_transfer)

    class _InferenceDataloader:
        """Wrapper to apply `on_before_batch_transfer` during iteration."""

        def __init__(self, dataloader, transform_fn):
            self.dataloader = dataloader
            self.transform_fn = transform_fn

        def __iter__(self):
            for batch in self.dataloader:
                yield self.transform_fn(batch, dataloader_idx=None)

        def __len__(self):
            return len(self.dataloader)
