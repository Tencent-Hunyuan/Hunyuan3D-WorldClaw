from worldclaw_oss.cache import ContentCache, cache_key
from worldclaw_oss.schemas import Stage
from worldclaw_oss.state import StateDB


def test_cache_key_is_order_independent(tmp_path):
    a = cache_key("p", "r", "m", "v", {"x": 1, "y": 2})
    b = cache_key("p", "r", "m", "v", {"y": 2, "x": 1})
    assert a == b
    cache = ContentCache(tmp_path)
    path = cache.put_bytes(a, ".bin", b"value")
    assert cache.get(a, "bin") == path


def test_state_db_persists_resume_state(tmp_path):
    db = StateDB(tmp_path / "state.sqlite3")
    db.initialize("run")
    db.start("run", Stage.INTENT)
    db.complete("run", Stage.INTENT, {"ok": True})
    reopened = StateDB(tmp_path / "state.sqlite3")
    assert reopened.status("run", Stage.INTENT) == "complete"
    assert reopened.attempts("run", Stage.INTENT) == 1
