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

"""Tests for the accept-queue probe.

A silently dropped connection leaves the client in ESTABLISHED with no reply and the
unit's worker idle, which is what post-mortem inspection of a hang actually showed.
The probe turns that guess into a measurement, so these tests pin the arithmetic it
reports and the levels it logs at.
"""

from transfer_queue.utils.accept_probe import (
    AcceptQueueProbe,
    AcceptQueueSample,
    AcceptQueueStats,
    _read_listen_overflows,
    sample_accept_queue,
)


def _sample(recv_q: int, backlog: int = 100, sk_drops: int = 0, overflows: int = 0) -> AcceptQueueSample:
    return AcceptQueueSample(
        timestamp=0.0,
        recv_q=recv_q,
        backlog=backlog,
        sk_drops=sk_drops,
        listen_overflows=overflows,
        listen_drops=overflows,
    )


def _probe() -> AcceptQueueProbe:
    return AcceptQueueProbe(port=34513, owner_id="TQ_STORAGE_UNIT_test", interval_s=0.01)


def test_utilization_is_depth_over_backlog():
    assert _sample(50, backlog=100).utilization == 0.5


def test_zero_backlog_reports_zero_utilization_not_zero_division():
    """A socket read before bind reports backlog 0; that must not raise."""
    assert _sample(5, backlog=0).utilization == 0.0


def test_full_queue_reports_full_utilization():
    assert _sample(100, backlog=100).utilization == 1.0


def test_empty_stats_report_zero_deltas():
    stats = AcceptQueueStats(port=1234)

    assert stats.sk_drops_delta == 0
    assert stats.overflow_delta == 0


def test_drop_delta_is_last_minus_first():
    """Deltas are what matter: the absolute counters carry the whole uptime's history."""
    stats = AcceptQueueStats(port=1234)
    stats.first_sample = _sample(0, sk_drops=23, overflows=1775)
    stats.last_sample = _sample(0, sk_drops=31, overflows=1790)

    assert stats.sk_drops_delta == 8
    assert stats.overflow_delta == 15


def test_describe_names_port_and_deltas():
    stats = AcceptQueueStats(port=34513, backlog=100, peak_recv_q=97)
    stats.first_sample = _sample(0, sk_drops=23)
    stats.last_sample = _sample(0, sk_drops=24)

    text = stats.describe()

    assert "port=34513" in text
    assert "peak_recv_q=97" in text
    assert "sk_drops_delta=1" in text


def test_peak_tracks_highest_depth_not_latest():
    """The burst is the signal; sampling after it drains reads zero."""
    probe = _probe()

    probe._record(_sample(10))
    probe._record(_sample(97))
    probe._record(_sample(3))

    assert probe.stats.peak_recv_q == 97


def test_first_sample_is_retained_as_delta_baseline():
    probe = _probe()

    probe._record(_sample(1, sk_drops=23))
    probe._record(_sample(2, sk_drops=25))

    assert probe.stats.first_sample.sk_drops == 23
    assert probe.stats.sk_drops_delta == 2


def test_drop_increase_is_logged_at_error(caplog):
    """A drop is the direct evidence, so it must not be buried at debug level."""
    probe = _probe()
    probe._record(_sample(0, sk_drops=23))

    with caplog.at_level("ERROR"):
        probe._record(_sample(100, sk_drops=24))

    assert "accept queue dropped a connection" in caplog.text


def test_near_full_queue_warns_once(caplog):
    """Repeating the warning every 100ms would flood the log of a 512-node job."""
    probe = _probe()

    with caplog.at_level("WARNING"):
        probe._record(_sample(60))
        probe._record(_sample(70))

    assert caplog.text.count("reached") == 1


def test_no_warning_below_threshold():
    probe = _probe()

    probe._record(_sample(10))

    assert probe._warned is False


def test_listen_overflows_returns_two_non_negative_ints():
    """Reads the real host, so assert shape rather than a specific value."""
    overflows, drops = _read_listen_overflows()

    assert overflows >= 0
    assert drops >= 0


def test_sampling_an_unused_port_returns_none():
    assert sample_accept_queue(1) is None
