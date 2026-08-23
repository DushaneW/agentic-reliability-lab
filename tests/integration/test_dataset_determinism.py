from __future__ import annotations

from arl.reliability.dataset import generate_dataset


def test_dataset_generation_is_deterministic(sum_task) -> None:  # noqa: ANN001
    examples_a = generate_dataset([sum_task], seeds_per_profile=2)
    examples_b = generate_dataset([sum_task], seeds_per_profile=2)

    ids_a = [e.run_id for e in examples_a]
    ids_b = [e.run_id for e in examples_b]
    assert ids_a == ids_b  # run ids must not depend on uuid4() timing

    labels_a = [e.label_failure for e in examples_a]
    labels_b = [e.label_failure for e in examples_b]
    assert labels_a == labels_b

    features_a = [e.features.as_list() for e in examples_a]
    features_b = [e.features.as_list() for e in examples_b]
    assert features_a == features_b
