"""The embedding model must stay small and quick inside a capped container.

See _EMBEDDING_CALL_LOCK and _SingleThreadMiniLM in alpha_vantage_vector_db.py
for the measurements behind these rules.
"""
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2  # noqa: E402

import alpha_vantage_vector_db as vdb  # noqa: E402

MODEL_FILE = Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME / "model.onnx"
needs_model = pytest.mark.skipif(not MODEL_FILE.exists(), reason="ONNX MiniLM model not downloaded")


@needs_model
def test_session_uses_one_non_spinning_thread():
    options = vdb.shared_embedding_model().model.get_session_options()
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"


@needs_model
def test_vectors_match_chromadb_stock_session():
    import numpy as np

    texts = ["AAPL quote: price 200.00, change +1.2%", "News: margins expand as demand recovers. " * 20, "x"]
    ours = np.array(vdb.shared_embedding_model()(texts))
    stock = np.array(ONNXMiniLM_L6_V2()(texts))
    assert ours.shape == (3, 384)
    assert np.max(np.abs(ours - stock)) < 1e-5


class _RecordingModel:
    """Stands in for the ONNX model: records call sizes and overlap."""

    def __init__(self):
        self.sizes = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, texts):
        with self._lock:
            self.sizes.append(len(texts))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.002)  # long enough for other threads to try to overlap
        with self._lock:
            self.active -= 1
        return [[float(len(t)), 1.0] for t in texts]


def test_embeds_one_document_per_call_one_call_at_a_time():
    model = _RecordingModel()
    db = vdb.AlphaVantageVectorDB.__new__(vdb.AlphaVantageVectorDB)  # skip the API key and ChromaDB setup
    db.embedding_model = model
    batches = [[f"doc {i}-{j}" + "." * j for j in range(10)] for i in range(6)]
    results = [None] * len(batches)

    def run(i):
        results[i] = db.create_embeddings_batch(batches[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(batches))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert set(model.sizes) == {1}
    assert len(model.sizes) == 60
    assert model.max_active == 1
    # Each caller gets its own vectors back, in order.
    for batch, vectors in zip(batches, results):
        assert vectors == [[float(len(t)), 1.0] for t in batch]
