from pathlib import Path

import numpy as np

from uav_ican_3d.vision import (
    ImageBelief,
    associate_candidates,
    inspect_visdrone_split,
    mahalanobis_squared,
    temporary_bbox_center_reference,
)


VISDRONE_ROOT = Path(
    "/home/xwl/.cache/kagglehub/datasets/banuprasadb/visdrone-dataset/versions/1/VisDrone_Dataset"
)


def test_mahalanobis_distance_uses_full_covariance() -> None:
    belief = ImageBelief(np.array([10.0, 20.0]), np.array([[4.0, 1.0], [1.0, 9.0]]))
    candidates = np.array([[10.0, 20.0], [12.0, 23.0]])
    distances = mahalanobis_squared(candidates, belief)
    expected = np.array([0.0, [2.0, 3.0] @ np.linalg.solve(belief.covariance_uv, [2.0, 3.0])])
    assert np.allclose(distances, expected)


def test_association_rejects_all_candidates_outside_gate() -> None:
    belief = ImageBelief(np.zeros(2), np.eye(2))
    result = associate_candidates(np.array([[20.0, 20.0], [-30.0, 10.0]]), belief, 0.95)
    assert result.selected_index is None


def test_bbox_center_is_explicit_temporary_reference() -> None:
    centers = temporary_bbox_center_reference(np.array([[0.0, 10.0, 20.0, 30.0]]))
    assert np.allclose(centers, [[10.0, 20.0]])


def test_local_visdrone_adapter_reads_vehicle_candidates() -> None:
    frames, candidates = inspect_visdrone_split(
        VISDRONE_ROOT, "VisDrone2019-DET-val", {3, 4, 5, 8}, 2
    )
    assert frames == 2
    assert candidates > 0

