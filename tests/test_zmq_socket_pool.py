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

"""Tests for ZMQSocketPool.

Sockets used to be created and closed per request. The pool reuses them, which is only
safe because a lease is exclusive and a socket that did not complete a clean send/recv is
discarded: replies carry no request id (see ZMQMessage.create), so a reply left in flight
by a timed-out or cancelled request would be read by the next user of that socket as its
own. test_timed_out_socket_is_not_reused pins exactly that.
"""

import asyncio
import threading

import pytest
import zmq
import zmq.asyncio

import transfer_queue.utils.zmq_utils as zmq_utils
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQServerInfo, ZMQSocketPool


class _Peer:
    """A ROUTER that echoes one reply per request, optionally after a delay."""

    def __init__(self, delay_first_reply: float = 0.0):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.info = ZMQServerInfo(role=Role.STORAGE, id="peer_0", ip="127.0.0.1", ports={"put_get_socket": port})
        self._delay_first_reply = delay_first_reply
        self._replies = 0
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            if not dict(poller.poll(50)):
                continue
            identity, request = self.socket.recv_multipart()
            if self._replies == 0 and self._delay_first_reply:
                # Reply late enough that the requester has already timed out, leaving this
                # reply in flight -- the poisoned-socket scenario.
                time_left = self._delay_first_reply
                while time_left > 0 and self.running:
                    sleep = min(0.05, time_left)
                    threading.Event().wait(sleep)
                    time_left -= sleep
            self._replies += 1
            self.socket.send_multipart([identity, b"reply-to-" + request])

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.socket.close(linger=0)
        self.context.term()


@pytest.fixture
def peer():
    p = _Peer()
    yield p
    p.stop()


def _idle_sockets(pool: ZMQSocketPool) -> list:
    """Every socket currently parked in the pool, across all owners and buckets."""
    return [s for buckets in pool._idle.values() for bucket in buckets.values() for s in bucket]


async def _round_trip(pool, peer_info, payload=b"req", timeout=None):
    with pool.lease(peer_info, "put_get_socket", timeout=timeout) as sock:
        await sock.send_multipart([payload])
        return (await sock.recv_multipart())[0]


@pytest.mark.asyncio
async def test_socket_is_reused_across_requests(peer):
    """Sequential requests to one peer must share a single socket."""
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")

    for i in range(10):
        assert await _round_trip(pool, peer.info, f"req{i}".encode()) == f"reply-to-req{i}".encode()

    idle = _idle_sockets(pool)
    assert len(idle) == 1, "each request opened its own socket instead of reusing one"

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_timed_out_socket_is_not_reused():
    """A timed-out request must not leave its socket -- or its late reply -- in the pool.

    Regression guard for the core hazard: with the socket pooled, the *next* request would
    receive the previous request's reply, silently attributing one response to another.
    """
    # Reply to the first request only after its 1s timeout has expired, so the reply is
    # still in flight when the socket would otherwise be handed to the next caller.
    peer = _Peer(delay_first_reply=2.0)
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")
    try:
        with pytest.raises(zmq.error.Again):
            await _round_trip(pool, peer.info, b"first", timeout=1)

        assert _idle_sockets(pool) == [], "a timed-out socket was returned to the pool"

        # The late reply to "first" must not surface as the answer to "second".
        assert await _round_trip(pool, peer.info, b"second", timeout=10) == b"reply-to-second"
    finally:
        pool.close()
        ctx.destroy(linger=0)
        peer.stop()


@pytest.mark.asyncio
async def test_cancelled_lease_discards_socket(peer):
    """Cancellation mid-recv poisons the socket: asyncio.gather cancels siblings routinely."""
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")
    leased = []

    async def never_answered():
        with pool.lease(peer.info, "put_get_socket") as sock:
            leased.append(sock)
            await asyncio.sleep(60)  # cancelled here, after the lease was handed out

    task = asyncio.create_task(never_answered())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert leased and leased[0].closed
    assert _idle_sockets(pool) == []

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_failed_lease_discards_socket(peer):
    """Any exception in the body poisons the socket, not just timeouts."""
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")

    with pytest.raises(RuntimeError):
        with pool.lease(peer.info, "put_get_socket") as sock:
            await sock.send_multipart([b"req"])
            raise RuntimeError("handler blew up")

    assert _idle_sockets(pool) == []

    pool.close()
    ctx.destroy(linger=0)


def test_sockets_are_not_reused_across_event_loops(peer):
    """A socket bound to a finished loop must never be handed to another one.

    pyzmq rebinds an async socket to whatever loop it next sees; one bound to a *closed*
    loop is the "Bad file descriptor / SIGABRT" failure this keying exists to prevent.
    """
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")
    leased = []

    async def lease_twice(tag):
        # Twice per loop, so a socket IS reused within a loop -- which is what makes the
        # cross-loop comparison below meaningful rather than trivially true.
        for i in range(2):
            with pool.lease(peer.info, "put_get_socket") as sock:
                leased.append(sock)
                await sock.send_multipart([f"{tag}{i}".encode()])
                await sock.recv_multipart()

    asyncio.run(lease_twice("a"))
    first = [s for s in leased]
    asyncio.run(lease_twice("b"))
    second = [s for s in leased if s not in first]

    assert len({id(s) for s in first}) == 1, "a socket should be reused within one loop"
    assert len({id(s) for s in second}) == 1
    assert not ({id(s) for s in first} & {id(s) for s in second}), "a socket crossed event loops"

    pool.close()
    ctx.destroy(linger=0)


def test_finished_loop_releases_its_sockets(peer):
    """A finished loop's sockets must be closed, not left parked bound to a dead loop.

    ``asyncio.run()`` per call -- the pattern the client docstrings show -- creates one loop
    per call. A pooled async socket keeps its own loop referenced, so these cannot be reaped
    by garbage collection; the pool evicts finished owners on the next lease instead.
    """
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")
    leased = []

    async def once():
        with pool.lease(peer.info, "put_get_socket") as sock:
            leased.append(sock)
            await sock.send_multipart([b"q"])
            await sock.recv_multipart()

    for _ in range(5):
        asyncio.run(once())

    assert len(leased) == 5, "each fresh loop needs its own socket"
    # The last loop's socket is still parked (nothing has leased since), but every earlier
    # one must have been closed rather than accumulating.
    assert all(sock.closed for sock in leased[:-1]), "a finished loop left an open socket behind"
    assert len(_idle_sockets(pool)) == 1
    assert len(pool._idle) == 1, "finished owners must be evicted, not accumulated"

    pool.close()
    ctx.destroy(linger=0)


def test_pooled_identities_are_unique_across_pools(peer):
    """Two pools with the same owner_id must not collide on the wire.

    A ROUTER silently drops a second peer claiming an identity it already has, so colliding
    identities would blackhole one process's traffic. Client ids are pid-derived and pids
    repeat across nodes, so owner_id alone cannot carry uniqueness.
    """
    ctx = zmq.asyncio.Context()
    # Same owner_id, as two processes on different nodes with equal pids would produce.
    a, b = ZMQSocketPool(ctx, "TransferQueueClient_1234"), ZMQSocketPool(ctx, "TransferQueueClient_1234")

    async def identity_of(pool):
        with pool.lease(peer.info, "put_get_socket") as sock:
            return sock.getsockopt(zmq.IDENTITY)

    first, second = asyncio.run(identity_of(a)), asyncio.run(identity_of(b))
    assert first != second, "two pools minted the same ZMQ identity"

    a.close()
    b.close()
    ctx.destroy(linger=0)


def test_connect_failure_does_not_leak_a_socket(peer):
    """A socket is nobody's responsibility until it reaches a lease, so _connect closes it."""
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")
    bad = ZMQServerInfo(role=Role.STORAGE, id="bad", ip="127.0.0.1", ports={"put_get_socket": -1})

    created = []
    original = zmq_utils.create_zmq_socket

    def spy(*args, **kwargs):
        sock = original(*args, **kwargs)
        created.append(sock)
        return sock

    zmq_utils.create_zmq_socket = spy
    try:
        with pytest.raises(zmq.ZMQError):
            with pool.lease(bad, "put_get_socket"):
                pass
    finally:
        zmq_utils.create_zmq_socket = original

    assert created and created[0].closed, "a socket that failed to connect was left open"

    pool.close()
    ctx.destroy(linger=0)
    ctx.destroy(linger=0)


def test_sync_caller_can_lease(peer):
    """The metrics collector leases from a plain thread, with no event loop running."""
    ctx = zmq.Context()
    pool = ZMQSocketPool(ctx, "metrics_collector")

    for i in range(3):
        with pool.lease(peer.info, "put_get_socket", timeout=5) as sock:
            sock.send_multipart([f"m{i}".encode()])
            assert sock.recv_multipart()[0] == f"reply-to-m{i}".encode()

    assert len(_idle_sockets(pool)) == 1

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_pool_size_is_a_soft_cap(peer):
    """Concurrency above maxsize still gets sockets; only the steady state is bounded."""
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", maxsize=2)

    results = await asyncio.gather(*[_round_trip(pool, peer.info, f"c{i}".encode()) for i in range(8)])
    assert len(results) == 8, "a burst beyond maxsize must not be refused or blocked"
    assert len(_idle_sockets(pool)) == 2, "excess sockets must be closed on return, not parked"

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_unknown_socket_name_is_reported(peer):
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")

    with pytest.raises(RuntimeError, match="not configured"):
        with pool.lease(peer.info, "no_such_socket"):
            pass

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_close_is_idempotent_and_survives_dead_context(peer):
    """Teardown ordering is not guaranteed, so close() must tolerate a destroyed context."""
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner")
    await _round_trip(pool, peer.info)

    pool.close()
    assert _idle_sockets(pool) == []
    pool.close()  # twice

    ctx.destroy(linger=0)
    pool.close()  # after the context is gone
