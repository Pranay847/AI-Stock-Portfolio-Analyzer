"""Two vector DBs constructed at once must both succeed.

chromadb registers a PersistentClient's System in a shared class-level dict
before starting it, without a lock, so two clients created together for the
same path used to both fail -- which is what concurrent first requests after a
cold start do.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_concurrent_construction_on_one_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    from alpha_vantage_vector_db import AlphaVantageVectorDB

    path = str(tmp_path / "store")
    barrier = threading.Barrier(3)
    errors = []

    def build():
        barrier.wait()
        try:
            AlphaVantageVectorDB(db_path=path)
        except Exception as e:  # noqa: BLE001 - recorded for the assertion
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=build) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
