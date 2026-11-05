import json
from pathlib import Path

import numpy as np

# all embeddings live in one file keyed by sound name; to add a sound, add one key there
embeddings_path = Path(__file__).parent / "embeddings.json"
embeddings = {name: np.array(vector) for name, vector in json.loads(embeddings_path.read_text()).items()}


def cosine_similarity(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
    """Returns cos theta between two vectors."""
    return float(np.dot(first_vector, second_vector) / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector)))


def cosine_to_degrees(cosine: float) -> float:
    """Returns the angle in degrees for a given cosine."""
    # float rounding can push cosine slightly outside [-1, 1], where arccos returns nan
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


# scores depend on the set and are unreliable for small N. raw cosine is inflated by a shared offset across embeddings.
mean_vector = np.mean(list(embeddings.values()), axis=0)


def print_similarity_matrix(title: str, vectors_by_name: dict[str, np.ndarray]) -> None:
    """Prints an all-pairs table where each cell is "cosine / angle in degrees".

    Args:
        title: Heading printed above the table.
        vectors_by_name: Vectors to compare, keyed by sound name.
    """
    names = list(vectors_by_name)
    row_label_width = max(len(name) for name in names)
    cell_width = 14
    print(f"\n{title}")
    # column headers are truncated so the table fits a terminal; full names are in the row labels
    print(" " * row_label_width + "".join(f"{name[:cell_width - 1]:>{cell_width}}" for name in names))
    for row_name in names:
        cells = []
        for column_name in names:
            cosine = cosine_similarity(vectors_by_name[row_name], vectors_by_name[column_name])
            cells.append(f"{f'{cosine:.3f}/{cosine_to_degrees(cosine):.0f}°':>{cell_width}}")
        print(f"{row_name:<{row_label_width}}" + "".join(cells))


print_similarity_matrix("cos / angle", embeddings)
print_similarity_matrix(
    "centered cos / angle", {name: vector - mean_vector for name, vector in embeddings.items()}
)
