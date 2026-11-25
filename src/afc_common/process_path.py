"""Pure instruction-local process-point path convention shared by readers."""
from __future__ import annotations

import math

PROCESS_MAPPING_RULES_VERSION = "afc.process_mapping.2"


def uniform_path_boundaries(length_mm, point_count):
    """Return n+1 bounds; source point k owns ((k-1)L/n, kL/n]."""
    length = float(length_mm)
    if not math.isfinite(length) or length < 0 or type(point_count) is not int or point_count < 1:
        raise ValueError("工艺行均分需要非负有限NC段长和正整数点数")
    # L*n/n can differ from L by one ULP; only internal bounds need arithmetic.
    internal = tuple(length * index / point_count for index in range(1, point_count))
    return (0.0, *internal, length)
