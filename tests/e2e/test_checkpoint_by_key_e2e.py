# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end tests for ``save_checkpoint_by_key``, and for ``save_checkpoint`` staying full.

Every storage unit writes its own checkpoint file from the node it runs on, so
``TQ_CHECKPOINT_TEST_ROOT`` must point at a filesystem shared by the whole
cluster. A node-local path such as pytest's ``tmp_path`` fails on a multi-node
cluster with ``FileNotFoundError``.

Run with:
    pytest tests/e2e/test_checkpoint_by_key_e2e.py -v
"""

import json
import os
import pickle
import shutil
import uuid
from pathlib import Path

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

import transfer_queue as tq

os.environ["RAY_DEDUP_LOGS"] = "0"

_NUM_STORAGE_UNITS = 2
_SU_SUBDIR = "simple_storage"
_SU_INFO_FILE = "storage_unit_info.json"
_MANIFEST_FILE = "selection_manifest.json"
_DEFAULT_CHECKPOINT_ROOT = "/apdcephfs_hldy/share_303541817/tq_checkpoint_tests"

_TQ_CONFIG = OmegaConf.create(
    {
        "controller": {"polling_mode": True},
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "total_storage_size": 200,
                "num_data_storage_units": _NUM_STORAGE_UNITS,
            },
        },
    }
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_init():
    if not ray.is_initialized():
        ray.init(namespace="TestCheckpointByKeyE2E")
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def tq_system(ray_init):
    tq.init(_TQ_CONFIG)
    yield
    tq.close()


@pytest.fixture
def controller(tq_system):
    return ray.get_actor("TransferQueueController", namespace="transfer_queue")


@pytest.fixture(autouse=True)
def cleanup_partitions(controller):
    yield
    try:
        for pid in ray.get(controller.list_partitions.remote()):
            ray.get(controller.clear_partition.remote(pid))
    except Exception:
        pass


@pytest.fixture(scope="module")
def shared_root():
    root = Path(os.environ.get("TQ_CHECKPOINT_TEST_ROOT", _DEFAULT_CHECKPOINT_ROOT)) / uuid.uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def case_dir(shared_root, request):
    case = shared_root / request.node.name.replace("/", "_")
    case.mkdir(parents=True, exist_ok=True)
    yield case
    shutil.rmtree(case, ignore_errors=True)


@pytest.fixture
def checkpoint_dir(case_dir):
    return case_dir / "checkpoint"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row_input_ids(row: int) -> torch.Tensor:
    """Deterministic per-row payload so a restored row can be traced to its key."""
    return torch.tensor([row * 10, row * 10 + 1, row * 10 + 2])


def _put_rows(partition_id: str, keys: list[str]) -> torch.Tensor:
    """Put one row per key with a per-row payload and an ``idx`` tag."""
    input_ids = torch.stack([_row_input_ids(row) for row in range(len(keys))])
    tq.kv_batch_put(
        keys=keys,
        partition_id=partition_id,
        fields=TensorDict(
            {"input_ids": input_ids, "attention_mask": torch.ones(len(keys), 3)},
            batch_size=len(keys),
        ),
        tags=[{"idx": row} for row in range(len(keys))],
    )
    return input_ids


def _assert_rows_equal(actual: torch.Tensor, expected_rows: list[torch.Tensor]) -> None:
    """Compare a retrieved field row by row.

    TransferQueue returns a batched field as a nested tensor, which ``torch.equal``
    cannot consume directly.
    """
    actual_rows = list(actual.unbind()) if actual.is_nested else list(actual)
    assert len(actual_rows) == len(expected_rows)
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        assert torch.equal(actual_row, expected_row)


def _keys_mapping(controller, partition_id: str) -> dict[str, int]:
    snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
    return dict(snapshot.keys_mapping)


def _load_su_states(checkpoint_dir) -> list[dict]:
    """Read every storage-unit pickle written by a checkpoint."""
    su_dir = checkpoint_dir / _SU_SUBDIR
    with open(su_dir / _SU_INFO_FILE) as f:
        su_info = json.load(f)
    states = []
    for entry in su_info:
        path = su_dir / f"su_{entry['position']}_{entry['storage_unit_id']}.pkl"
        with open(path, "rb") as f:
            states.append(pickle.load(f))
    return states


# ---------------------------------------------------------------------------
# selective save → load roundtrip
# ---------------------------------------------------------------------------


class TestSaveByKeyRoundtrip:
    def test_only_selected_keys_are_restored(self, tq_system, checkpoint_dir, controller):
        # Define test data
        partition_id = "p_sel_basic"
        keys = [f"k{i}" for i in range(6)]
        selected = ["k1", "k4"]

        # Put
        input_ids = _put_rows(partition_id, keys)
        original_mapping = _keys_mapping(controller, partition_id)

        # Save only the selected keys
        tq.save_checkpoint_by_key(checkpoint_dir, selected, partition_id)

        # Wipe
        ray.get(controller.clear_partition.remote(partition_id))
        assert ray.get(controller.list_partitions.remote()) == []

        # Load
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state: exactly the selected keys survive
        restored_mapping = _keys_mapping(controller, partition_id)
        assert sorted(restored_mapping) == sorted(selected)

        # Check loaded state: global indexes are preserved, so storage routing is stable
        for key in selected:
            assert restored_mapping[key] == original_mapping[key]

        # Check loaded state: payloads round-trip
        retrieved = tq.kv_batch_get(keys=selected, partition_id=partition_id, select_fields=["input_ids"])
        _assert_rows_equal(retrieved["input_ids"], [input_ids[keys.index(key)] for key in selected])

    def test_tags_preserved_for_selected_keys(self, tq_system, checkpoint_dir, controller):
        # Define test data
        partition_id = "p_sel_tags"
        keys = [f"t{i}" for i in range(5)]
        selected = ["t0", "t3"]

        # Put + Save + Wipe + Load
        _put_rows(partition_id, keys)
        tq.save_checkpoint_by_key(checkpoint_dir, selected, partition_id)
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state: each selected row keeps its own tag
        snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
        for key in selected:
            gidx = snapshot.keys_mapping[key]
            assert snapshot.custom_meta[gidx]["idx"] == keys.index(key)

    def test_selection_spans_every_storage_unit(self, tq_system, checkpoint_dir, controller):
        # Define test data
        partition_id = "p_sel_su"
        keys = [f"s{i}" for i in range(8)]

        # Put
        _put_rows(partition_id, keys)
        mapping = _keys_mapping(controller, partition_id)

        # Rows route to storage units by global_index % num_units, and global indexes
        # are not necessarily contiguous, so pick one key per unit from the real mapping.
        key_per_unit: dict[int, str] = {}
        for key in keys:
            key_per_unit.setdefault(mapping[key] % _NUM_STORAGE_UNITS, key)
        assert len(key_per_unit) == _NUM_STORAGE_UNITS, "put did not cover every storage unit"
        selected = sorted(key_per_unit.values(), key=keys.index)

        # Save
        tq.save_checkpoint_by_key(checkpoint_dir, selected, partition_id)

        # Check saved state: every unit is marked selective and writes exactly its one row
        su_dir = checkpoint_dir / _SU_SUBDIR
        with open(su_dir / _SU_INFO_FILE) as f:
            su_info = json.load(f)
        assert all(entry["selective"] for entry in su_info)
        assert all(entry["saved_rows"] == 1 for entry in su_info)

        selected_indexes = {mapping[key] for key in selected}
        saved_indexes = set()
        for state in _load_su_states(checkpoint_dir):
            assert state["selective"] is True
            saved_indexes |= set(state["active_keys"])
            for values in state["field_data"].values():
                assert set(values) <= selected_indexes
        assert saved_indexes == selected_indexes

        # Check loaded state after a wipe
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_checkpoint(checkpoint_dir)
        retrieved = tq.kv_batch_get(keys=selected, partition_id=partition_id, select_fields=["input_ids"])
        _assert_rows_equal(retrieved["input_ids"], [_row_input_ids(keys.index(key)) for key in selected])

    def test_other_partitions_are_excluded(self, tq_system, checkpoint_dir, controller):
        # Define test data
        selected_partition = "p_sel_target"
        other_partition = "p_sel_other"

        # Put into two partitions
        _put_rows(selected_partition, ["a0", "a1", "a2"])
        _put_rows(other_partition, ["b0", "b1"])

        # Save one key from one partition
        tq.save_checkpoint_by_key(checkpoint_dir, ["a1"], selected_partition)

        # Wipe both
        for pid in (selected_partition, other_partition):
            ray.get(controller.clear_partition.remote(pid))

        # Load
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state: the untouched partition is not resurrected
        assert ray.get(controller.list_partitions.remote()) == [selected_partition]
        assert sorted(_keys_mapping(controller, selected_partition)) == ["a1"]

    def test_duplicate_keys_are_deduplicated(self, tq_system, checkpoint_dir, controller):
        # Define test data
        partition_id = "p_sel_dup"
        _put_rows(partition_id, ["d0", "d1", "d2"])

        # Save with a repeated key
        tq.save_checkpoint_by_key(checkpoint_dir, ["d1", "d1", "d2"], partition_id)

        # Check saved state: the repeat collapses to a single row
        with open(checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["selection"]["num_keys"] == 2
        assert meta["selection"]["num_produced_rows"] == 2

        # Check loaded state
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_checkpoint(checkpoint_dir)
        assert sorted(_keys_mapping(controller, partition_id)) == ["d1", "d2"]

    def test_single_key_accepts_a_bare_string(self, tq_system, checkpoint_dir, controller):
        # Define test data
        partition_id = "p_sel_str"
        _put_rows(partition_id, ["x0", "x1"])

        # Save with keys passed as a plain string rather than a sequence
        tq.save_checkpoint_by_key(checkpoint_dir, "x1", partition_id)

        # Check loaded state
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_checkpoint(checkpoint_dir)
        assert sorted(_keys_mapping(controller, partition_id)) == ["x1"]

    def test_non_tensor_fields_round_trip(self, tq_system, checkpoint_dir, controller):
        # Define test data
        partition_id = "p_sel_nontensor"
        keys = ["n0", "n1", "n2"]
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=TensorDict(
                {
                    "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]),
                    "text": NonTensorStack("alpha", "beta", "gamma"),
                },
                batch_size=len(keys),
            ),
            tags=[{} for _ in keys],
        )

        # Save + Wipe + Load
        tq.save_checkpoint_by_key(checkpoint_dir, ["n2"], partition_id)
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state
        retrieved = tq.kv_batch_get(keys=["n2"], partition_id=partition_id, select_fields=["input_ids", "text"])
        _assert_rows_equal(retrieved["input_ids"], [torch.tensor([5, 6])])
        assert list(retrieved["text"]) == ["gamma"]


# ---------------------------------------------------------------------------
# checkpoint metadata and manifest
# ---------------------------------------------------------------------------


class TestSaveByKeyMetadata:
    def test_metadata_records_the_selection(self, tq_system, checkpoint_dir):
        # Define test data
        partition_id = "p_sel_meta"
        _put_rows(partition_id, [f"m{i}" for i in range(4)])

        # Save
        tq.save_checkpoint_by_key(checkpoint_dir, ["m0", "m2"], partition_id, metadata={"step": 7})

        # Check saved state
        with open(checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["storage_saved"] is True
        assert meta["user_metadata"]["step"] == 7
        assert meta["selection"] == {
            "partition_id": partition_id,
            "num_keys": 2,
            "num_produced_rows": 2,
        }

    def test_manifest_lists_fields_per_key(self, tq_system, checkpoint_dir):
        # Define test data
        partition_id = "p_sel_manifest"
        _put_rows(partition_id, ["f0", "f1", "f2"])

        # Save
        tq.save_checkpoint_by_key(checkpoint_dir, ["f0", "f2"], partition_id)

        # Check saved state
        with open(checkpoint_dir / _MANIFEST_FILE) as f:
            manifest = json.load(f)
        assert manifest["partition_id"] == partition_id
        assert sorted(manifest["keys"]) == ["f0", "f2"]
        for fields in manifest["keys"].values():
            assert fields == ["attention_mask", "input_ids"]

    def test_full_checkpoint_writes_no_selection(self, tq_system, checkpoint_dir, controller):
        # Define test data
        partition_id = "p_full"
        keys = ["g0", "g1", "g2"]
        input_ids = _put_rows(partition_id, keys)

        # Save without a selection
        tq.save_checkpoint(checkpoint_dir)

        # Check saved state: no selection block, no manifest, full storage dump
        with open(checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)
        assert "selection" not in meta
        assert not (checkpoint_dir / _MANIFEST_FILE).exists()
        su_dir = checkpoint_dir / _SU_SUBDIR
        with open(su_dir / _SU_INFO_FILE) as f:
            su_info = json.load(f)
        assert all(sorted(entry) == ["position", "storage_unit_id"] for entry in su_info)

        # Check loaded state: every key comes back
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_checkpoint(checkpoint_dir)
        assert sorted(_keys_mapping(controller, partition_id)) == keys
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["input_ids"])
        _assert_rows_equal(retrieved["input_ids"], list(input_ids))


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


class TestSaveByKeyErrors:
    def test_empty_keys_raises(self, tq_system, checkpoint_dir):
        _put_rows("p_err_nokeys", ["e0"])
        with pytest.raises(ValueError, match="must not be empty"):
            tq.save_checkpoint_by_key(checkpoint_dir, [], "p_err_nokeys")

    def test_full_save_rejects_a_key_selection(self, tq_system, checkpoint_dir):
        """save_checkpoint keeps its original signature; selection lives in its own function."""
        _put_rows("p_err_nopid", ["e0"])
        with pytest.raises(TypeError):
            tq.save_checkpoint(checkpoint_dir, keys=["e0"], partition_id="p_err_nopid")

    def test_unknown_key_raises_and_leaves_no_directory(self, tq_system, case_dir):
        # Define test data
        partition_id = "p_err_key"
        _put_rows(partition_id, ["e0", "e1"])
        ck = case_dir / "ck"

        # Save a key that was never put
        with pytest.raises(RuntimeError, match="keys not found"):
            tq.save_checkpoint_by_key(ck, ["e0", "missing"], partition_id)

        # Check saved state: no partial directory left behind
        assert not ck.exists()
        assert not (case_dir / "ck.tmp").exists()

    def test_unknown_partition_raises(self, tq_system, checkpoint_dir):
        _put_rows("p_err_part", ["e0"])
        with pytest.raises(RuntimeError, match="does not exist"):
            tq.save_checkpoint_by_key(checkpoint_dir, ["e0"], "p_never_created")

    def test_live_partition_survives_a_selective_save(self, tq_system, checkpoint_dir, controller):
        """A selective save must snapshot, not mutate, the controller's live partition."""
        # Define test data
        partition_id = "p_sel_nonmutating"
        keys = [f"l{i}" for i in range(4)]
        input_ids = _put_rows(partition_id, keys)
        before = _keys_mapping(controller, partition_id)

        # Save a subset
        tq.save_checkpoint_by_key(checkpoint_dir, ["l1"], partition_id)

        # Check live state: untouched rows are still readable with their original indexes
        assert _keys_mapping(controller, partition_id) == before
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["input_ids"])
        _assert_rows_equal(retrieved["input_ids"], list(input_ids))
