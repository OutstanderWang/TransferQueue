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

"""Tests that the socket timeout is resolved per call rather than at import.

``with_zmq_socket`` used to receive the timeout as a value, so the module-level constant
was read once when the module first loaded. Anything that set the env var later, or a test
that patched the constant, was silently ignored: a request asking for 1s still waited the
400s the env held at import. These pin the callable form that fixes it.
"""

import asyncio
from unittest.mock import patch

import pytest
import zmq

from transfer_queue.storage.managers import simple_storage_manager as ssm
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQServerInfo, with_zmq_socket

UNIT_ID = "TQ_STORAGE_UNIT_x"


class _Owner:
    """Minimal owner exposing what the decorator reads."""

    def __init__(self, port: int) -> None:
        self.storage_manager_id = "TQ_STORAGE_test"
        self.storage_unit_infos = {
            UNIT_ID: ZMQServerInfo(role=Role.STORAGE, id=UNIT_ID, ip="127.0.0.1", ports={"put_get_socket": port})
        }
        self.zmq_context = zmq.asyncio.Context()


def _decorate(timeout):
    """Build a decorated coroutine that reports the socket's RCVTIMEO."""

    @with_zmq_socket(
        "put_get_socket",
        get_identity=lambda self: self.storage_manager_id,
        get_peer=lambda self, target: self.storage_unit_infos[target],
        get_context=lambda self: self.zmq_context,
        resolve_target=lambda args, kwargs: kwargs.get("target_storage_unit"),
        timeout=timeout,
    )
    async def report(self, *, target_storage_unit, socket=None):
        return socket.getsockopt(zmq.RCVTIMEO), socket.getsockopt(zmq.SNDTIMEO)

    return report


@pytest.fixture
def owner():
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    instance = _Owner(port)
    try:
        yield instance
    finally:
        router.close(linger=0)
        instance.zmq_context.term()
        context.term()


def test_int_timeout_is_applied_to_the_socket(owner):
    """The plain form must keep working for callers that pass a constant."""
    report = _decorate(7)

    rcv, snd = asyncio.run(report(owner, target_storage_unit=UNIT_ID))

    assert (rcv, snd) == (7000, 7000)


def test_callable_timeout_is_resolved_at_call_time(owner):
    current = {"value": 5}
    report = _decorate(lambda: current["value"])

    first, _ = asyncio.run(report(owner, target_storage_unit=UNIT_ID))
    current["value"] = 11
    second, _ = asyncio.run(report(owner, target_storage_unit=UNIT_ID))

    assert (first, second) == (5000, 11000), "a callable timeout must be re-read for each socket"


def test_none_timeout_leaves_the_socket_blocking(owner):
    report = _decorate(None)

    rcv, snd = asyncio.run(report(owner, target_storage_unit=UNIT_ID))

    assert (rcv, snd) == (-1, -1)


def test_patching_the_manager_constant_reaches_the_socket(owner):
    """Guards the production wiring: both storage-unit decorators pass a callable.

    Without this the constant is frozen at import, which is what made a 1s timeout in a
    test wait the full 400s the environment happened to hold.
    """
    with patch.object(ssm, "TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT", 3):
        report = _decorate(lambda: ssm.TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT)
        rcv, _ = asyncio.run(report(owner, target_storage_unit=UNIT_ID))

    assert rcv == 3000
