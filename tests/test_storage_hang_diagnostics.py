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

"""Tests for the storage-unit hang diagnostics: endpoint resolution and the post-mortem hold."""

import logging
from unittest.mock import patch

import zmq

from transfer_queue.metrics import TQMetricsExporter
from transfer_queue.storage.managers import simple_storage_manager as ssm
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo


def _manager_with_units(**units: ZMQServerInfo) -> AsyncSimpleStorageManager:
    """Build a manager carrying only the state the diagnostics helpers read."""
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_manager_id = "TQ_STORAGE_test"
    manager.storage_unit_infos = dict(units)
    return manager


def _server_info(unit_id: str, ip: str, port: int) -> ZMQServerInfo:
    return ZMQServerInfo(role=Role.STORAGE, id=unit_id, ip=ip, ports={"put_get_socket": port})


def test_describe_storage_unit_returns_ip_and_port():
    """A registered unit resolves to ip:port so a timeout names a reachable endpoint."""
    manager = _manager_with_units(unit_a=_server_info("unit_a", "10.0.0.7", 5555))

    assert manager._describe_storage_unit("unit_a") == "10.0.0.7:5555"


def test_describe_storage_unit_unregistered_does_not_raise():
    """An unknown unit must degrade to a readable string, never mask the original failure."""
    manager = _manager_with_units()

    description = manager._describe_storage_unit("missing_unit")

    assert "endpoint unknown" in description


def test_hold_for_postmortem_disabled_by_default_does_not_sleep():
    """With the env var unset (production default) the hold must be a no-op."""
    with patch.object(ssm, "TQ_HANG_HOLD_SECONDS", 0), patch.object(ssm.time, "sleep") as sleep:
        ssm.hold_for_postmortem("TQ_STORAGE_test", "unit_a", "10.0.0.7:5555", "get")

        sleep.assert_not_called()


def test_hold_for_postmortem_sleeps_for_configured_seconds():
    """When enabled the hold sleeps exactly the configured duration before returning."""
    with patch.object(ssm, "TQ_HANG_HOLD_SECONDS", 120), patch.object(ssm.time, "sleep") as sleep:
        ssm.hold_for_postmortem("TQ_STORAGE_test", "unit_a", "10.0.0.7:5555", "get")

        sleep.assert_called_once_with(120)


def _exporter_probing(side_effect) -> TQMetricsExporter:
    """Build an exporter whose socket layer is replaced by ``side_effect``."""
    exporter = TQMetricsExporter.__new__(TQMetricsExporter)
    exporter._zmq_sockets = {}
    exporter._unresponsive_storage_units = set()
    exporter._get_or_create_socket = lambda su_id, su_info: side_effect()
    return exporter


def test_metrics_probe_timeout_logs_once_while_unit_stays_hung(caplog):
    """A hung unit must warn on the transition, not once per 10s collection cycle."""

    def always_timeout():
        raise zmq.error.Again()

    exporter = _exporter_probing(always_timeout)
    info = _server_info("unit_a", "10.0.0.7", 5555)

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            assert exporter._query_storage_unit(info, "unit_a") is None

    timeouts = [r for r in caplog.records if "querying metrics" in r.message]
    assert len(timeouts) == 1, f"expected one warning across five probes, got {len(timeouts)}"
    assert "10.0.0.7:5555" in timeouts[0].message


def test_metrics_probe_recovery_is_logged(caplog):
    """Recovery must be visible, otherwise a one-shot warning reads as still-broken."""
    state = {"fail": True}

    class _Socket:
        def send_multipart(self, frames):
            if state["fail"]:
                raise zmq.error.Again()

        def recv_multipart(self, copy=False):
            return ZMQMessage.create(
                request_type=ZMQRequestType.METRICS_RESPONSE,
                sender_id="unit_a",
                body={"storage_unit_id": "unit_a"},
            ).serialize()

    exporter = _exporter_probing(_Socket)
    info = _server_info("unit_a", "10.0.0.7", 5555)

    with caplog.at_level(logging.WARNING):
        assert exporter._query_storage_unit(info, "unit_a") is None
        state["fail"] = False
        assert exporter._query_storage_unit(info, "unit_a") is not None

    assert any("answering metrics probes again" in r.message for r in caplog.records)
    assert exporter._unresponsive_storage_units == set()
