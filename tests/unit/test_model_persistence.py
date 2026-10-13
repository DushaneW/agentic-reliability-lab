from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from arl.evaluation.benchmark import load_benchmark
from arl.reliability.dataset import generate_dataset
from arl.reliability.features import FEATURE_NAMES
from arl.reliability.model import MODEL_FORMAT, ModelFormatError, ReliabilityModel


@pytest.fixture(scope="module")
def trained() -> ReliabilityModel:
    model = ReliabilityModel()
    model.fit(generate_dataset(load_benchmark("benchmarks/smoke"), seeds_per_profile=3))
    return model


def test_round_trip_gives_identical_predictions(trained: ReliabilityModel, tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    trained.save(path)
    loaded = ReliabilityModel.load(path)
    for scale in (0.0, 1.0, 3.5, 20.0):
        vector = [scale] * len(FEATURE_NAMES)
        assert loaded.predict_proba(vector) == trained.predict_proba(vector)


def test_saved_file_is_plain_json(trained: ReliabilityModel, tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    trained.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == MODEL_FORMAT
    assert data["feature_names"] == FEATURE_NAMES


def test_saving_an_untrained_model_fails() -> None:
    with pytest.raises(RuntimeError):
        ReliabilityModel().save("never-written.json")


def test_pickle_files_are_refused_and_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"

    class Evil:
        def __reduce__(self) -> tuple[object, ...]:
            return (marker.write_text, ("pwned",))

    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(ModelFormatError, match="pickle"):
        ReliabilityModel.load(path)
    assert not marker.exists()


def test_json_with_a_different_feature_set_is_refused(
    trained: ReliabilityModel, tmp_path: Path
) -> None:
    path = tmp_path / "m.json"
    trained.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["feature_names"] = data["feature_names"][:-1]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ModelFormatError, match="feature set"):
        ReliabilityModel.load(path)


def test_binary_file_with_a_json_name_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "sneaky.json"
    path.write_bytes(pickle.dumps({"a": 1}))
    with pytest.raises(ModelFormatError):
        ReliabilityModel.load(path)


@pytest.mark.parametrize(
    "content",
    ["not json", "[]", '{"format": "something-else"}', json.dumps({"format": MODEL_FORMAT})],
)
def test_malformed_files_raise_model_format_error(tmp_path: Path, content: str) -> None:
    path = tmp_path / "m.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ModelFormatError):
        ReliabilityModel.load(path)
