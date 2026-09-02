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

"""Regression tests for the storage-unit worker thread surviving per-request failures.

A worker thread that dies leaves the unit unable to answer anything while its process and
bound ROUTER socket stay up, so every client blocks until its own recv timeout regardless of
how high that timeout is set. These tests drive the real ``_worker_loop`` over an inproc
socket pair and assert that a poisoned request never takes the thread down with it.
"""

import threading
from unittest.mock import patch

import pytest
import zmq

from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType

# Long enough that a healthy worker always answers, short enough that a dead one fails fast.
RECV_TIMEOUT_MS = 5000

# @ray.remote replaces the class with an ActorClass wrapper, so reach through it to the real
# class in order to construct an instance without starting Ray.
_StorageUnitClass = SimpleStorageUnit.__ray_metadata__.modified_class


class _WorkerHarness:
    """Run ``SimpleStorageUnit._worker_loop`` against an inproc DEALER, with no Ray actor.

    ``SimpleStorageUnit`` is a Ray actor whose ``__init__`` binds sockets and spawns threads,
    so it is bypassed via ``__new__``: these tests are about the loop's exception handling,
    not about actor startup.
    """

    def __init__(self, storage_unit_size: int | None = None):
        from transfer_queue.storage.simple_storage import StorageUnitData

        self.unit = _StorageUnitClass.__new__(_StorageUnitClass)
        self.unit.storage_unit_id = "TQ_STORAGE_UNIT_test0001"
        self.unit.storage_unit_size = storage_unit_size
        self.unit.storage_data = StorageUnitData(storage_unit_size)
        self.unit._shutdown_event = threading.Event()
        self.unit._metrics = None

        self.context = zmq.Context()
        self.unit.zmq_context = self.context
        self.unit._node_ip = "127.0.0.1"
        self.address = "inproc://worker_resilience_test"

        # ROUTER stands in for the proxy frontend so replies route back by identity.
        self.client = self.context.socket(zmq.ROUTER)
        self.client.bind(self.address)

        self.worker_socket = self.context.socket(zmq.DEALER)
        self.worker_socket.setsockopt(zmq.IDENTITY, b"worker")
        self.worker_socket.connect(self.address)

        self.poller = zmq.Poller()
        self.poller.register(self.worker_socket, zmq.POLLIN)

        self.thread = threading.Thread(
            target=self.unit._worker_loop,
            args=(self.worker_socket, self.poller, _NullMonitor()),
            daemon=True,
        )
        self.thread.start()

    def request(self, request_type, body: dict) -> ZMQMessage | None:
        """Send one request and return the decoded reply, or None if the worker never answers."""
        msg = ZMQMessage.create(request_type=request_type, sender_id="test_client", body=body)
        self.send_raw(msg.serialize())
        return self.recv_reply()

    def send_raw(self, frames: list[bytes]) -> None:
        """Deliver frames to the worker using the production framing.

        In production a ROUTER/DEALER proxy sits in front of the worker and prepends the
        originating client's identity, so the worker always sees [client_identity, msg...].
        This harness talks to the worker directly, so it adds that frame itself.
        """
        self.client.send_multipart([b"worker", b"client_identity"] + list(frames))

    def recv_reply(self) -> ZMQMessage | None:
        """Return the next decoded reply, or None if the worker does not answer in time."""
        if not self.client.poll(RECV_TIMEOUT_MS):
            return None
        frames = self.client.recv_multipart(copy=False)
        # Strip the ROUTER-added worker identity and the client identity the worker echoes back.
        return ZMQMessage.deserialize(frames[2:])

    def close(self) -> None:
        self.unit._shutdown_event.set()
        self.thread.join(timeout=5)
        self.client.close(linger=0)
        if not self.worker_socket.closed:
            self.worker_socket.close(linger=0)
        self.context.term()


class _NullMonitor:
    """Stand-in for IntervalPerfMonitor/TQMetricsExporter; measure() must be a no-op context."""

    def measure(self, op_type: str):
        from contextlib import nullcontext

        return nullcontext()


@pytest.fixture
def harness():
    h = _WorkerHarness()
    yield h
    h.close()


def test_undecodable_request_keeps_worker_alive(harness):
    """An undecodable frame must not kill the worker: the next valid request still answers."""
    harness.send_raw([b"not a serialized ZMQMessage"])
    # The worker replies with an error here; drain it so the next assertion reads its own reply.
    harness.recv_reply()

    reply = harness.request(ZMQRequestType.GET_METRICS, {})

    assert reply is not None, (
        "worker stopped answering after an undecodable request; a dead worker thread is exactly "
        "the failure that makes clients hang until their recv timeout"
    )
    assert reply.request_type == ZMQRequestType.METRICS_RESPONSE
    assert harness.thread.is_alive()


def test_handler_exception_returns_error_and_keeps_worker_alive(harness):
    """A handler raising must yield a PUT_GET_ERROR reply, not a silent hang."""
    with patch.object(harness.unit, "_handle_get", side_effect=RuntimeError("boom")):
        reply = harness.request(ZMQRequestType.GET_DATA, {"global_indexes": [0], "fields": ["x"]})

        assert reply is not None, "handler exception left the caller with no reply"
        assert reply.request_type == ZMQRequestType.PUT_GET_ERROR
        assert "boom" in reply.body["message"]

    # The unit must still serve traffic once the failing handler is restored.
    assert harness.request(ZMQRequestType.GET_METRICS, {}) is not None
    assert harness.thread.is_alive()


def test_send_failure_keeps_worker_alive(harness):
    """A failed reply send strands one caller but must not end the worker thread."""
    original_send = harness.worker_socket.send_multipart
    calls = {"n": 0}

    def fail_first_send(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise zmq.ZMQError("send failed")
        return original_send(*args, **kwargs)

    with patch.object(harness.worker_socket, "send_multipart", side_effect=fail_first_send):
        assert harness.request(ZMQRequestType.GET_METRICS, {}) is None

    assert harness.thread.is_alive(), "a failed send killed the worker thread"
    assert harness.request(ZMQRequestType.GET_METRICS, {}) is not None


def test_unknown_operation_returns_operation_error(harness):
    """An unroutable operation must get an explicit error reply rather than be dropped."""
    reply = harness.request(ZMQRequestType.HANDSHAKE, {})

    assert reply is not None
    assert reply.request_type == ZMQRequestType.PUT_GET_OPERATION_ERROR
