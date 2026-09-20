import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar.dedupe import item_id  # noqa: E402
from radar.models import RawItem  # noqa: E402
from radar.store import Store  # noqa: E402

NOW = dt.datetime(2026, 9, 21, 0, 0, tzinfo=dt.timezone.utc)


def make_item(title: str, url: str | None = None, tier: int = 3, hours_ago: float = 1,
              country: str = "ID", publisher: str = "Pub") -> RawItem:
    url = url or f"https://example.com/{abs(hash(title))}"
    return RawItem(
        id=item_id(url), url=url, title=title, publisher=publisher,
        published=NOW - dt.timedelta(hours=hours_ago), fetched_at=NOW,
        feed_name="test", country_hint=country, tier=tier,
    )


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


@pytest.fixture
def cfg():
    return {
        "dedupe": {"simhash_max_distance": 3, "jaccard_threshold": 0.6,
                   "event_merge_threshold": 0.75, "event_candidate_threshold": 0.3,
                   "event_lookback_days": 30},
        "fetch": {"window_hours": 48, "concurrency": 4, "timeout_seconds": 5},
        "feeds": [
            {"name": "F1", "country": "ID", "tier": 3, "query": "digital lending Indonesia"},
            {"name": "F2", "country": "PH", "tier": 1, "query": "site:bsp.gov.ph credit"},
        ],
    }
