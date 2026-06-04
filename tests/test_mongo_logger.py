"""Unit tests for prediction logging helpers (no live MongoDB required)."""

from unittest.mock import MagicMock, patch

import pytest

from src.mongo_logger import PredictionLogger, try_create_logger


def test_try_create_logger_without_uri(monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    assert try_create_logger() is None


def test_try_create_logger_connection_failure(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    with patch("src.mongo_logger.PredictionLogger", side_effect=Exception("down")):
        assert try_create_logger() is None


def test_log_prediction_builds_document(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017")
    mock_coll = MagicMock()
    with patch("src.mongo_logger.MongoClient") as mock_client:
        mock_client.return_value.admin.command.return_value = True
        logger = PredictionLogger()
        logger.predictions = mock_coll

    logger.log_prediction(
        image_id="test.jpg",
        instances=[
            {
                "category_id": 1,
                "category_name": "lane",
                "score": 0.9,
                "bbox": [0, 0, 10, 10],
                "mask_b64": "x",
                "mask_shape": [100, 100],
            }
        ],
        inference_ms=12.5,
        model_version="mask2former_int8.onnx",
    )

    mock_coll.insert_one.assert_called_once()
    doc = mock_coll.insert_one.call_args[0][0]
    assert doc["image_id"] == "test.jpg"
    assert doc["mean_score"] == pytest.approx(0.9)
    assert len(doc["instances"]) == 1
    assert "mask_b64" not in doc["instances"][0]
