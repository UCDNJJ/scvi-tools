from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import scvi
from scvi import REGISTRY_KEYS
from scvi.data import synthetic_iid
from scvi.dataloaders import MultiVIMappedCollectionDataModule
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


def _write_collection_parts(tmp_path, prefix: str, adatas):
    paths = []
    for idx, adata in enumerate(adatas):
        path = tmp_path / f"{prefix}_{idx}.h5ad"
        adata.write_h5ad(path)
        paths.append(str(path))
    return _FakeCollection(paths)


def _make_multivi_collections(tmp_path, fully_paired: bool):
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
        parallel=False,
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
        parallel=False,
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
        parallel=False,
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
        parallel=False,
        categorical_covariate_keys=None,
    )

    assert datamodule.n_cats_per_cov == []
    model = _create_and_train_multivi_model(datamodule)
    assert model.is_trained
    assert model.module.n_batch == 2
    assert model.module.n_cats_per_cov == []
    latent = model.get_latent_representation(
        dataloader=datamodule.inference_dataloader(batch_size=4)
    )
    assert latent.shape == (datamodule.n_obs, 5)


@pytest.mark.dataloader
@pytest.mark.parametrize(
    ("rna_collection_name", "atac_collection_name", "expected_x_shape", "expected_atac_shape"),
    [
        ("rna_collection", "atac_collection", (4, 12), (4, 8)),
        ("rna_collection", None, (4, 12), (4, 0)),
        (None, "atac_collection", (4, 0), (4, 8)),
    ],
)
@dependencies("lamindb")
def test_multivi_custom_dataloader_parallel_workers_use_lazy_handles(
    tmp_path,
    rna_collection_name,
    atac_collection_name,
    expected_x_shape,
    expected_atac_shape,
):
    rna_collection, atac_collection, _, _, _ = _make_multivi_collections(
        tmp_path, fully_paired=False
    )
    collections = {
        "rna_collection": rna_collection,
        "atac_collection": atac_collection,
        None: None,
    }
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=collections[rna_collection_name],
        atac_collection=collections[atac_collection_name],
        batch_key="batch",
        batch_size=4,
        shuffle=False,
        parallel=True,
        parallel_cpu_count=1,
        categorical_covariate_keys=["site"],
        continuous_covariate_keys=["score"],
    )

    dataloader = datamodule.inference_dataloader(batch_size=4)
    assert dataloader.num_workers == 1
    assert dataloader.persistent_workers is True
    for source in datamodule._sources:
        assert all(adata is None for adata in source._adatas)

    batch = next(iter(dataloader))
    assert batch[REGISTRY_KEYS.X_KEY].shape == expected_x_shape
    assert batch[REGISTRY_KEYS.ATAC_X_KEY].shape == expected_atac_shape
    for source in datamodule._sources:
        assert all(adata is None for adata in source._adatas)

    _shutdown_workers(dataloader)


@pytest.mark.dataloader
@dependencies("lamindb")
def test_multivi_custom_dataloader_parallel_false_keeps_single_process_fetching(tmp_path):
    rna_collection, atac_collection, _, _, _ = _make_multivi_collections(
        tmp_path, fully_paired=True
    )
    datamodule = MultiVIMappedCollectionDataModule(
        rna_collection=rna_collection,
        atac_collection=atac_collection,
        batch_key="batch",
        batch_size=4,
        shuffle=False,
        parallel=False,
        categorical_covariate_keys=["site"],
        continuous_covariate_keys=["score"],
    )

    dataloader = datamodule.inference_dataloader(batch_size=4)
    assert dataloader.num_workers == 0
    assert dataloader.persistent_workers is False
    for source in datamodule._sources:
        assert all(adata is None for adata in source._adatas)

    batch = next(iter(dataloader))
    assert batch[REGISTRY_KEYS.X_KEY].shape == (4, 12)
    assert batch[REGISTRY_KEYS.ATAC_X_KEY].shape == (4, 8)
    for source in datamodule._sources:
        assert any(adata is not None for adata in source._adatas)

    datamodule.close()
    for source in datamodule._sources:
        assert all(adata is None for adata in source._adatas)
