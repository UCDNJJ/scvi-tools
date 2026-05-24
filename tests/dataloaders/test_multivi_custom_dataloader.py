from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

import scvi
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid
from scvi.dataloaders import MultiVIMappedCollectionDataModule
from scvi.dataloaders import _custom_dataloaders as custom_dataloaders
from scvi.utils import dependencies


@dataclass
class _FakeArtifact:
    path: str

    def load(self):
        from anndata import read_h5ad

        return read_h5ad(self.path)


class _FakeArtifacts:
    def __init__(self, artifacts):
        self._artifacts = artifacts

    def all(self):
        return self._artifacts


class _FakeCollection:
    def __init__(self, paths: list[str]):
        self.artifacts = _FakeArtifacts([_FakeArtifact(path) for path in paths])


def _shutdown_workers(dataloader) -> None:
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        dataloader._iterator = None


def _prepare_h5ad_for_write(adata):
    adata = adata.copy()
    adata.obs_names = pd.Index(np.asarray(adata.obs_names.astype(str), dtype=object), dtype=object)
    adata.var_names = pd.Index(np.asarray(adata.var_names.astype(str), dtype=object), dtype=object)
    for column in adata.obs.columns:
        if is_string_dtype(adata.obs[column]) or isinstance(
            adata.obs[column].dtype, pd.CategoricalDtype
        ):
            adata.obs[column] = pd.Series(
                np.asarray(adata.obs[column].astype(str), dtype=object),
                index=adata.obs.index,
                dtype=object,
            )
    return adata


def _write_collection_parts(tmp_path, prefix: str, adatas):
    paths = []
    for idx, adata in enumerate(adatas):
        path = tmp_path / f"{prefix}_{idx}.h5ad"
        _prepare_h5ad_for_write(adata).write_h5ad(path)
        paths.append(str(path))
    return _FakeCollection(paths)


def _make_multivi_collections(
    tmp_path,
    fully_paired: bool,
    sparse_atac: bool = True,
    sparse_rna: bool = False,
):
    mdata = synthetic_iid(
        batch_size=6,
        n_batches=2,
        n_genes=12,
        n_regions=8,
        n_proteins=4,
        return_mudata=True,
    )
    order = np.array([7, 1, 10, 3, 6, 0, 8, 2, 11, 4, 9, 5])
    obs_names = np.array([f"cell_{i:02d}" for i in order], dtype=object)

    rna = mdata.mod["rna"][order].copy()
    atac = mdata.mod["accessibility"][order].copy()
    if sparse_rna:
        rna.X = csr_matrix(rna.X.toarray() if hasattr(rna.X, "toarray") else np.asarray(rna.X))
    if sparse_atac:
        atac.X = csr_matrix(atac.X.toarray() if hasattr(atac.X, "toarray") else np.asarray(atac.X))
    for adata in (rna, atac):
        adata.obs_names = obs_names
        adata.obs["batch"] = np.where(
            np.arange(adata.n_obs) < adata.n_obs // 2,
            "batch_0",
            "batch_1",
        )
        adata.obs["site"] = np.where(np.arange(adata.n_obs) % 2 == 0, "site_a", "site_b")
        adata.obs["score"] = np.linspace(0.0, 1.0, adata.n_obs)

    if fully_paired:
        rna_parts = [rna[:6].copy(), rna[6:].copy()]
        atac_parts = [atac[:6].copy(), atac[6:].copy()]
        missing_rna = set()
        missing_atac = set()
    else:
        rna_parts = [rna[[0, 1, 3, 4, 6]].copy(), rna[[7, 9, 11]].copy()]
        atac_parts = [atac[[0, 2, 5, 6]].copy(), atac[[7, 8, 10, 11]].copy()]
        missing_rna = set(obs_names) - set(
            np.concatenate([part.obs_names.to_numpy() for part in rna_parts])
        )
        missing_atac = set(obs_names) - set(
            np.concatenate([part.obs_names.to_numpy() for part in atac_parts])
        )

    rna_collection = _write_collection_parts(tmp_path, "rna", rna_parts)
    atac_collection = _write_collection_parts(tmp_path, "atac", atac_parts)
    return rna_collection, atac_collection, sorted(set(obs_names)), missing_rna, missing_atac


def _make_sparse_collection_source(tmp_path):
    adata_1 = AnnData(
        X=csr_matrix(np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)),
    )
    adata_1.obs_names = np.array(["cell_0", "cell_1", "cell_2"], dtype=object)
    adata_1.var_names = np.array(["var_0", "var_1", "var_2"], dtype=object)
    adata_1.obs["batch"] = "batch_0"

    adata_2 = AnnData(
        X=csr_matrix(np.array([[4, 0, 0], [0, 5, 0], [0, 0, 6]], dtype=np.float32)),
    )
    adata_2.obs_names = np.array(["cell_3", "cell_4", "cell_5"], dtype=object)
    adata_2.var_names = np.array(["var_0", "var_1", "var_2"], dtype=object)
    adata_2.obs["batch"] = "batch_1"

    collection = _write_collection_parts(tmp_path, "sparse", [adata_1, adata_2])
    return custom_dataloaders._CollectionBackedAnnData(collection, obs_columns=["batch"])


def _create_and_train_multivi_model(datamodule, modality_weights: str = "equal"):
    model = scvi.model.MULTIVI(
        adata=None,
        registry=datamodule.registry,
        n_latent=5,
        n_hidden=16,
        modality_weights=modality_weights,
        encode_covariates=True,
    )
    model.train(
        datamodule=datamodule,
        max_epochs=1,
        batch_size=4,
        adversarial_mixing=True,
    )
    return model


@pytest.mark.dataloader
@dependencies("lamindb")
def test_multivi_custom_dataloader_scans_local_h5ad_metadata_without_backed_read(
    tmp_path, monkeypatch
):
    (
        rna_collection,
        atac_collection,
        obs_names,
        _,
        _,
    ) = _make_multivi_collections(tmp_path, fully_paired=True)
    expected_var_names = np.asarray(
        rna_collection.artifacts.all()[0].load().var_names,
        dtype=object,
    )
    expected_region_names = np.asarray(
        atac_collection.artifacts.all()[0].load().var_names,
        dtype=object,
    )

    read_h5ad_calls = []
    original_read_h5ad = custom_dataloaders.ad.read_h5ad

    def _tracking_read_h5ad(*args, **kwargs):
        read_h5ad_calls.append(kwargs.get("backed"))
        return original_read_h5ad(*args, **kwargs)

    monkeypatch.setattr(custom_dataloaders.ad, "read_h5ad", _tracking_read_h5ad)

    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=rna_collection,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=len(obs_names),
        shuffle=False,
        categorical_covariate_keys=["site"],
        continuous_covariate_keys=["score"],
    )

    assert datamodule.n_obs == len(obs_names)
    assert np.array_equal(datamodule.var_names, expected_var_names)
    assert np.array_equal(datamodule.region_names, expected_region_names)
    assert read_h5ad_calls == []

    batch = next(iter(datamodule.inference_dataloader(batch_size=len(obs_names))))
    assert batch[REGISTRY_KEYS.X_KEY].shape == (len(obs_names), len(expected_var_names))
    assert batch[REGISTRY_KEYS.ATAC_X_KEY].shape == (
        len(obs_names),
        len(expected_region_names),
    )
    assert any(call == "r" for call in read_h5ad_calls)


@pytest.mark.dataloader
@dependencies("lamindb")
def test_multivi_custom_dataloader_registry_and_batch_shapes(tmp_path):
    (
        rna_collection,
        atac_collection,
        obs_names,
        missing_rna,
        missing_atac,
    ) = _make_multivi_collections(tmp_path, fully_paired=False)
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=rna_collection,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=len(obs_names),
        shuffle=False,
        categorical_covariate_keys=["site"],
        continuous_covariate_keys=["score"],
    )

    assert datamodule.n_obs == len(obs_names)
    assert datamodule.n_vars == 12
    assert datamodule.n_regions == 8
    assert datamodule.n_batch == 2
    assert datamodule.n_cats_per_cov == [2]
    assert datamodule.registry["field_registries"][REGISTRY_KEYS.ATAC_X_KEY]["summary_stats"] == {
        "n_atac": 8
    }

    batch = next(iter(datamodule.inference_dataloader(batch_size=len(obs_names))))
    assert batch[REGISTRY_KEYS.INDICES_KEY].squeeze(-1).tolist() == list(range(len(obs_names)))
    assert batch[REGISTRY_KEYS.X_KEY].shape == (len(obs_names), 12)
    assert batch[REGISTRY_KEYS.ATAC_X_KEY].shape == (len(obs_names), 8)
    assert batch[REGISTRY_KEYS.CAT_COVS_KEY].shape == (len(obs_names), 1)
    assert batch[REGISTRY_KEYS.CONT_COVS_KEY].shape == (len(obs_names), 1)

    obs_to_pos = {name: idx for idx, name in enumerate(obs_names)}
    for obs_name in missing_rna:
        assert batch[REGISTRY_KEYS.X_KEY][obs_to_pos[obs_name]].sum().item() == 0
    for obs_name in missing_atac:
        assert batch[REGISTRY_KEYS.ATAC_X_KEY][obs_to_pos[obs_name]].sum().item() == 0


@pytest.mark.dataloader
@pytest.mark.parametrize("fully_paired", [True, False])
@pytest.mark.parametrize("modality_weights", ["equal", "universal"])
@dependencies("lamindb")
def test_multivi_custom_dataloader_train(tmp_path, fully_paired: bool, modality_weights: str):
    rna_collection, atac_collection, _, _, _ = _make_multivi_collections(tmp_path, fully_paired)
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=rna_collection,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=4,
        shuffle=False,
        categorical_covariate_keys=["site"],
        continuous_covariate_keys=["score"],
    )

    model = _create_and_train_multivi_model(datamodule, modality_weights=modality_weights)
    assert model.module is not None
    assert model.module.n_input_genes == datamodule.n_vars
    assert model.module.n_input_regions == datamodule.n_regions
    assert model.module.n_input_proteins == 0

    latent = model.get_latent_representation(
        dataloader=datamodule.inference_dataloader(batch_size=4)
    )
    assert latent.shape == (datamodule.n_obs, 5)


@pytest.mark.dataloader
@pytest.mark.parametrize(
    ("rna_only", "expected_n_obs", "expected_x_shape", "expected_atac_shape"),
    [
        (True, 8, (8, 12), (8, 0)),
        (False, 8, (8, 0), (8, 8)),
    ],
)
@dependencies("lamindb")
def test_multivi_custom_dataloader_single_modality(
    tmp_path,
    rna_only,
    expected_n_obs,
    expected_x_shape,
    expected_atac_shape,
):
    (
        rna_collection,
        atac_collection,
        _,
        _,
        _,
    ) = _make_multivi_collections(tmp_path, fully_paired=False)
    present_collection = rna_collection if rna_only else atac_collection
    present_obs_names = sorted(
        {
            obs_name
            for artifact in present_collection.artifacts.all()
            for obs_name in artifact.load().obs_names.astype(str)
        }
    )
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=present_collection if rna_only else None,
        atac_collection=None if rna_only else present_collection,
        batch_key="batch",
        batch_size=4,
        shuffle=False,
        categorical_covariate_keys=["site"],
        continuous_covariate_keys=["score"],
    )

    assert datamodule.n_obs == expected_n_obs
    assert datamodule.n_vars == (12 if rna_only else 0)
    assert datamodule.n_regions == (0 if rna_only else 8)
    assert np.array_equal(
        datamodule._global_obs_names,
        np.asarray(present_obs_names, dtype=object),
    )
    assert datamodule.registry["field_registries"][REGISTRY_KEYS.X_KEY]["state_registry"][
        "column_names"
    ].shape == (12 if rna_only else 0,)
    assert datamodule.registry["field_registries"][REGISTRY_KEYS.ATAC_X_KEY]["state_registry"][
        "column_names"
    ].shape == (0 if rna_only else 8,)

    batch = next(iter(datamodule.inference_dataloader(batch_size=expected_n_obs)))
    assert batch[REGISTRY_KEYS.X_KEY].shape == expected_x_shape
    assert batch[REGISTRY_KEYS.ATAC_X_KEY].shape == expected_atac_shape

    model = _create_and_train_multivi_model(datamodule)
    assert model.module.n_input_genes == datamodule.n_vars
    assert model.module.n_input_regions == datamodule.n_regions

    latent = model.get_latent_representation(
        dataloader=datamodule.inference_dataloader(batch_size=4)
    )
    assert latent.shape == (datamodule.n_obs, 5)


@pytest.mark.dataloader
@dependencies("lamindb")
def test_multivi_custom_dataloader_requires_collection():
    with pytest.raises(
        ValueError,
        match="At least one of `rna_collection` or `atac_collection` must be provided.",
    ):
        MultiVIMappedCollectionDataModule(
            rna_collection=None,
            atac_collection=None,
        )


@pytest.mark.dataloader
@dependencies("lamindb")
def test_multivi_custom_dataloader_no_extra_cat_covs_uses_batch_and_trains(tmp_path):
    rna_collection, atac_collection, _, _, _ = _make_multivi_collections(
        tmp_path, fully_paired=True
    )
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=rna_collection,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=4,
        shuffle=False,
        categorical_covariate_keys=None,
    )

    assert datamodule.n_cats_per_cov == []
    model = _create_and_train_multivi_model(datamodule)
    assert model.is_trained
    assert model.module.n_batch == 2
    assert list(model.module.n_cats_per_cov) == []
    latent = model.get_latent_representation(
        dataloader=datamodule.inference_dataloader(batch_size=4)
    )
    assert latent.shape == (datamodule.n_obs, 5)


@pytest.mark.dataloader
@dependencies("lamindb")
def test_multivi_custom_dataloader_propagates_worker_kwargs(tmp_path):
    rna_collection, atac_collection, _, _, _ = _make_multivi_collections(
        tmp_path, fully_paired=True
    )
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=rna_collection,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )

    dataloader = datamodule.train_dataloader()
    assert dataloader.num_workers == 2
    assert dataloader.pin_memory is True
    assert dataloader.persistent_workers is True
    assert dataloader.prefetch_factor == 3
    _shutdown_workers(dataloader)


def test_multivi_custom_dataloader_ddp_uses_batch_sampler(monkeypatch):
    class _TrackingIndexDataset(custom_dataloaders._IndexDataset):
        def __init__(self, indices):
            super().__init__(indices)
            self.seen_index_types = []

        def __getitem__(self, index: int) -> int:
            self.seen_index_types.append(type(index))
            return super().__getitem__(index)

    class _DummyBatchDistributedSampler:
        def __init__(self, dataset, batch_size, **kwargs):
            self.dataset = dataset
            self.batch_size = batch_size

        def __iter__(self):
            yield [0, 1]

        def __len__(self):
            return 1

    monkeypatch.setattr(custom_dataloaders.dist, "is_available", lambda: True)
    monkeypatch.setattr(custom_dataloaders.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        custom_dataloaders,
        "BatchDistributedSampler",
        _DummyBatchDistributedSampler,
    )

    datamodule = MultiVIMappedCollectionDataModule.__new__(MultiVIMappedCollectionDataModule)
    datamodule._batch_size = 2
    datamodule._drop_last = False
    datamodule._drop_dataset_tail = False
    datamodule._num_workers = 2
    datamodule._pin_memory = True
    datamodule._persistent_workers = True
    datamodule._prefetch_factor = 4
    datamodule._collate_fn = lambda batch_indices: batch_indices

    dataset = _TrackingIndexDataset(np.arange(2, dtype=np.int64))
    dataloader = datamodule._create_dataloader(dataset, shuffle=False)

    assert isinstance(dataloader.batch_sampler, _DummyBatchDistributedSampler)
    assert dataloader.num_workers == 2
    assert dataloader.pin_memory is True
    assert dataloader.persistent_workers is True
    assert dataloader.prefetch_factor == 4
    sampled_batch = next(iter(dataloader.batch_sampler))
    assert sampled_batch == [0, 1]
    _ = [dataset[idx] for idx in sampled_batch]
    assert dataset.seen_index_types == [int, int]


def test_multivi_custom_dataloader_non_ddp_uses_main_process_loader(monkeypatch):
    monkeypatch.setattr(custom_dataloaders.dist, "is_available", lambda: False)
    monkeypatch.setattr(custom_dataloaders.dist, "is_initialized", lambda: False)

    datamodule = MultiVIMappedCollectionDataModule.__new__(MultiVIMappedCollectionDataModule)
    datamodule._batch_size = 2
    datamodule._persistent_workers = True
    datamodule._collate_fn = lambda batch_indices: batch_indices

    dataset = custom_dataloaders._IndexDataset(np.arange(4, dtype=np.int64))
    dataloader = datamodule._create_dataloader(dataset, shuffle=False)

    assert dataloader.num_workers == 0


@pytest.mark.dataloader
@dependencies("lamindb")
def test_fetch_rows_returns_sparse_when_densify_false(tmp_path):
    source = _make_sparse_collection_source(tmp_path)
    obs_names = np.array(["cell_4", "cell_0", "cell_3", "cell_1"], dtype=object)
    sparse_matrix = source.fetch_rows(obs_names, densify=False)
    dense_matrix = source.fetch_rows(obs_names, densify=True)

    assert issparse(sparse_matrix)
    assert sparse_matrix.shape == (len(obs_names), source.n_vars)
    np.testing.assert_allclose(sparse_matrix.toarray(), dense_matrix)


@pytest.mark.dataloader
@dependencies("lamindb")
def test_fetch_rows_dense_when_densify_true(tmp_path):
    source = _make_sparse_collection_source(tmp_path)
    obs_names = np.array(["cell_4", "cell_0", "cell_3", "cell_1"], dtype=object)
    dense_matrix = source.fetch_rows(obs_names, densify=True)
    assert isinstance(dense_matrix, np.ndarray)


@pytest.mark.dataloader
@dependencies("lamindb")
def test_fetch_rows_sparse_handles_missing_obs_names(tmp_path):
    source = _make_sparse_collection_source(tmp_path)
    obs_names = np.array(["cell_0", "missing_cell", "cell_5", "another_missing"], dtype=object)
    sparse_matrix = source.fetch_rows(obs_names, densify=False)
    dense = sparse_matrix.toarray()

    assert issparse(sparse_matrix)
    np.testing.assert_allclose(dense[1], np.zeros(source.n_vars, dtype=np.float32))
    np.testing.assert_allclose(dense[3], np.zeros(source.n_vars, dtype=np.float32))


@pytest.mark.dataloader
@dependencies("lamindb")
def test_collate_fn_emits_sparse_atac_tensor(tmp_path):
    rna_collection, atac_collection, obs_names, _, _ = _make_multivi_collections(
        tmp_path, fully_paired=False
    )
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=rna_collection,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=len(obs_names),
        shuffle=False,
        sparse_atac=True,
    )

    batch = next(iter(datamodule.inference_dataloader(batch_size=len(obs_names))))
    atac_tensor = batch[REGISTRY_KEYS.ATAC_X_KEY]
    indices = batch[REGISTRY_KEYS.INDICES_KEY].squeeze(-1).numpy()
    batch_obs_names = datamodule._global_obs_names[indices]
    expected = datamodule._atac_source.fetch_rows(batch_obs_names, densify=True)

    assert atac_tensor.layout == torch.sparse_csr
    assert atac_tensor.is_sparse_csr
    assert batch["atac_X"] is atac_tensor
    assert atac_tensor.shape == (len(obs_names), datamodule.n_regions)
    np.testing.assert_allclose(atac_tensor.to_dense().numpy(), expected)


@pytest.mark.dataloader
@dependencies("lamindb")
def test_collate_fn_emits_dense_atac_when_sparse_atac_false(tmp_path):
    rna_collection, atac_collection, obs_names, _, _ = _make_multivi_collections(
        tmp_path, fully_paired=False
    )
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=rna_collection,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=len(obs_names),
        shuffle=False,
        sparse_atac=False,
    )

    batch = next(iter(datamodule.inference_dataloader(batch_size=len(obs_names))))
    atac_tensor = batch[REGISTRY_KEYS.ATAC_X_KEY]
    assert not atac_tensor.is_sparse
    assert not atac_tensor.is_sparse_csr


@pytest.mark.dataloader
@dependencies("lamindb")
def test_multivi_custom_dataloader_sparse_atac_smoke_train(tmp_path):
    _, atac_collection, _, _, _ = _make_multivi_collections(tmp_path, fully_paired=False)
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=None,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=4,
        shuffle=False,
        sparse_atac=True,
    )
    model = scvi.model.MULTIVI(adata=None, registry=datamodule.registry, modality_weights="equal")
    model.train(datamodule=datamodule, max_epochs=1, batch_size=4)
    assert model.module is not None
