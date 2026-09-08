"""Accept-queue instrumentation for the storage-unit ROUTER socket.

A get request was observed leaving the manager (TCP counted the bytes and the peer
ACKed them) while the unit's worker never saw it: its GET_DATA counter did not move
and it kept polling. The suspected drop point is the listen socket's accept queue,
which ZMQ leaves at its default backlog of 100 and which the kernel empties
silently (``tcp_abort_on_overflow=0``), so neither end raises an error.

This module samples the kernel's own view of that queue so the guess becomes a
measurement:

* ``Recv-Q`` on a listening socket is the number of established-but-not-yet-accepted
  connections, i.e. the live queue depth.
* ``ListenOverflows`` / ``ListenDrops`` count queue-full events machine-wide.
* ``sk_drops`` counts drops charged to one specific socket.

Sampling runs in a thread because the queue drains in milliseconds; a value read
after a hang has already returned to zero, which is exactly what earlier
post-mortem inspection saw.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)

_RECVQ_RE = re.compile(r"^LISTEN\s+(\d+)\s+(\d+)\s+\S*:(\d+)\s", re.M)
_DROPS_RE = re.compile(r"\bd(\d+)\b")


@dataclass
class AcceptQueueSample:
    """One reading of the accept queue for a single listening port."""

    timestamp: float
    recv_q: int
    backlog: int
    sk_drops: int
    listen_overflows: int
    listen_drops: int

    @property
    def utilization(self) -> float:
        """Queue depth as a fraction of the configured backlog."""
        return self.recv_q / self.backlog if self.backlog else 0.0


@dataclass
class AcceptQueueStats:
    """Peak and delta summary over a sampling window."""

    port: int
    samples: int = 0
    peak_recv_q: int = 0
    peak_utilization: float = 0.0
    backlog: int = 0
    first_sample: AcceptQueueSample | None = None
    last_sample: AcceptQueueSample | None = None
    peak_history: list[tuple[float, int]] = field(default_factory=list)

    @property
    def sk_drops_delta(self) -> int:
        if self.first_sample is None or self.last_sample is None:
            return 0
        return self.last_sample.sk_drops - self.first_sample.sk_drops

    @property
    def overflow_delta(self) -> int:
        if self.first_sample is None or self.last_sample is None:
            return 0
        return self.last_sample.listen_overflows - self.first_sample.listen_overflows

    def describe(self) -> str:
        return (
            f"port={self.port} samples={self.samples} backlog={self.backlog} "
            f"peak_recv_q={self.peak_recv_q} peak_util={self.peak_utilization:.1%} "
            f"sk_drops_delta={self.sk_drops_delta} listen_overflow_delta={self.overflow_delta}"
        )


def _read_listen_socket(port: int) -> tuple[int, int, int] | None:
    """Return (recv_q, backlog, sk_drops) for the listening socket on ``port``."""
    try:
        out = subprocess.run(["ss", "-lntm"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"accept-probe: ss failed: {exc}")
        return None

    for match in _RECVQ_RE.finditer(out):
        recv_q, backlog, found_port = (int(g) for g in match.groups())
        if found_port != port:
            continue
        # skmem lives on the continuation line right after the match.
        tail = out[match.end() : match.end() + 400]
        drops_match = _DROPS_RE.search(tail.split("\n")[1] if "\n" in tail else "")
        return recv_q, backlog, int(drops_match.group(1)) if drops_match else 0
    return None


def _read_listen_overflows() -> tuple[int, int]:
    """Return machine-wide (ListenOverflows, ListenDrops) from /proc/net/netstat."""
    try:
        with open("/proc/net/netstat") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return 0, 0

    for i, line in enumerate(lines):
        if not line.startswith("TcpExt:") or "ListenOverflows" not in line:
            continue
        keys = line.split()
        values = lines[i + 1].split()
        try:
            return (
                int(values[keys.index("ListenOverflows")]),
                int(values[keys.index("ListenDrops")]),
            )
        except (ValueError, IndexError):
            return 0, 0
    return 0, 0


def sample_accept_queue(port: int) -> AcceptQueueSample | None:
    """Take one accept-queue reading for ``port``, or None if it cannot be read."""
    listen = _read_listen_socket(port)
    if listen is None:
        return None
    recv_q, backlog, sk_drops = listen
    overflows, drops = _read_listen_overflows()
    return AcceptQueueSample(
        timestamp=time.time(),
        recv_q=recv_q,
        backlog=backlog,
        sk_drops=sk_drops,
        listen_overflows=overflows,
        listen_drops=drops,
    )


class AcceptQueueProbe:
    """Sample one listening port's accept queue from a background thread.

    Args:
        port: Listening port to watch.
        owner_id: Identifier used in log lines (the storage unit id).
        interval_s: Seconds between samples. Keep well under a second; the queue
            drains in milliseconds and a slower cadence misses the burst entirely.
        warn_utilization: Log a warning the first time depth reaches this fraction
            of the backlog, so a near-miss is visible before any drop happens.
    """

    def __init__(
        self,
        port: int,
        owner_id: str,
        interval_s: float = 0.1,
        warn_utilization: float = 0.5,
    ) -> None:
        self.port = port
        self.owner_id = owner_id
        self.interval_s = interval_s
        self.warn_utilization = warn_utilization
        self.stats = AcceptQueueStats(port=port)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"AcceptQueueProbe-{self.owner_id}", daemon=True)
        self._thread.start()
        logger.info(f"[{self.owner_id}]: accept-queue probe started on port {self.port} (interval={self.interval_s}s)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info(f"[{self.owner_id}]: accept-queue probe stopped. {self.stats.describe()}")

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = sample_accept_queue(self.port)
            if sample is not None:
                self._record(sample)
            self._stop.wait(self.interval_s)

    def _record(self, sample: AcceptQueueSample) -> None:
        stats = self.stats
        stats.samples += 1
        stats.backlog = sample.backlog or stats.backlog
        if stats.first_sample is None:
            stats.first_sample = sample
        stats.last_sample = sample

        if sample.recv_q > stats.peak_recv_q:
            stats.peak_recv_q = sample.recv_q
            stats.peak_utilization = sample.utilization
            stats.peak_history.append((sample.timestamp, sample.recv_q))

        # A drop charged to this socket is the direct evidence the guess needs, so it
        # is reported at error level with the depth that produced it.
        if stats.first_sample is not None and sample.sk_drops > stats.first_sample.sk_drops:
            logger.error(
                f"[{self.owner_id}]: accept queue dropped a connection on port {self.port}. "
                f"recv_q={sample.recv_q}/{sample.backlog} sk_drops={sample.sk_drops} "
                f"(+{sample.sk_drops - stats.first_sample.sk_drops} since probe start). "
                f"A silently dropped connection leaves the client in ESTABLISHED with no "
                f"reply; raise ZMQ_BACKLOG above {sample.backlog}."
            )

        if not self._warned and sample.utilization >= self.warn_utilization:
            self._warned = True
            logger.warning(
                f"[{self.owner_id}]: accept queue on port {self.port} reached "
                f"{sample.recv_q}/{sample.backlog} ({sample.utilization:.0%}); "
                f"connections are queueing faster than they are accepted."
            )
