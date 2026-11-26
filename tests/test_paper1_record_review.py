from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "verification"))
import pytest
from review_paper1_record_evidence import record_decisions, bind_steady_process, PROCESS_MAPPING_RULES_VERSION


def evidence():
    record = dict(record_id="r", no_operator_pause=True, spacing_mm=.5)
    numeric = dict(record_id="r", file_sha256="measurement", scan_status="completed", samples=5,
                   invalid_samples=0, invalid_feed_samples=0, duplicate_conflicts={},
                   observed_end_marker_candidates=[dict(raw_minus_one=True)])
    aligned = dict(record_id="r", measurement_sha256="measurement", nc_sha256="program",
                   selected_candidate=dict(line_offset=1), parser_issues=[])
    prepared = dict(record_id="r", source_files=dict(measurement=dict(sha256="measurement"), nc=dict(sha256="program")),
                    process_segments=[dict(endpoint_difference_mm=0.)], process_contact_conflict_positions=0,
                    process_mapping_rules_version=PROCESS_MAPPING_RULES_VERSION,
                    source_process_points=4, canonical_process_rows=4,
                    process_source_indices_preserved=True, process_point_order_unique=True,
                    mapped_positive_segments_strictly_increasing=True, mapped_segment_endpoints_match_nc=True,
                    process_geometry_conflict_positions=0,
                    cutting_contact_points=4, candidate_aligned_contact_points=3, max_accepted_alignment_error_mm=.05)
    return [record, numeric, aligned, prepared]


def test_missing_points_do_not_become_fabricated_full_coverage():
    result = record_decisions(*evidence())
    assert result["complete"] and result["alignment_reliable"]
    assert "coverage" not in result


def test_wrong_program_or_unbound_evidence_cannot_pass():
    args = evidence()
    args[0]["recollection_pending"] = True
    result = record_decisions(*args)
    assert not any(result[k] for k in ("complete", "alignment_reliable", "process_aligned"))
    args = evidence()
    args[2]["measurement_sha256"] = "different"
    result = record_decisions(*args)
    assert not any(result[k] for k in ("complete", "alignment_reliable", "process_aligned"))


def test_completeness_and_alignment_are_separate_evidence():
    args = evidence()
    args[1]["observed_end_marker_candidates"][0]["raw_minus_one"] = False
    assert not record_decisions(*args)["complete"]
    assert record_decisions(*args)["alignment_reliable"]
    args = evidence()
    args[3]["max_accepted_alignment_error_mm"] = .11
    assert not record_decisions(*args)["alignment_reliable"]
    args[3]["process_segments"][0]["endpoint_difference_mm"] = .6
    assert not record_decisions(*args)["process_aligned"]


def test_observed_post_cut_stop_can_bound_cutting_without_claiming_m30():
    args = evidence()
    args[1]["observed_end_marker_candidates"][0]["raw_minus_one"] = False
    args[0]["post_cut_stop_observed"] = True
    result = record_decisions(*args)
    assert result["complete"]
    assert not result["terminal_nc_marker_observed"]


def test_duplicate_geometry_does_not_pass_uniform_mapping_review():
    args = evidence()
    args[3]["process_contact_conflict_positions"] = 2
    args[3]["process_geometry_conflict_positions"] = 2
    result = record_decisions(*args)
    assert not result["process_aligned"]
    assert result["process_contact_conflict_positions"] == 2


def test_author_facts_and_inputs_are_not_rewritten():
    args = evidence()
    args[0]["no_operator_pause"] = None
    before = deepcopy(args)
    assert not record_decisions(*args)["complete"]
    assert args == before


@pytest.mark.parametrize("key,value", [
    ("process_mapping_rules_version", "old"),
    ("canonical_process_rows", 3),
    ("process_source_indices_preserved", False),
    ("process_point_order_unique", False),
    ("mapped_positive_segments_strictly_increasing", False),
    ("mapped_segment_endpoints_match_nc", False),
])
def test_old_mapping_or_lost_points_cannot_pass(key, value):
    args = evidence()
    args[3][key] = value
    assert not record_decisions(*args)["process_aligned"]


def test_steady_binding_preserves_unknown_quality_and_raw_source():
    record = dict(record_id="r", nc="nc", process="raw", process_version="raw", program_version="p", no_alarm=None)
    entry = dict(canonical_process="canonical", canonical_sha256="canonical", evidence="proof", evidence_sha256="proof")
    evidence = dict(process_mapping_rules_version=PROCESS_MAPPING_RULES_VERSION, source_points_preserved=True,
                    program_version="p", nc_sha256="nc", source_process_sha256="raw")
    bind_steady_process(record, entry, evidence, hash_file=lambda p: p)
    assert record["process"] == "canonical" and record["process_version"] == "canonical"
    assert record["source_process"]["path"] == "raw" and record["no_alarm"] is None
    assert record["steady_lines"] == [] and record["process_line_offset"] == 0
    with pytest.raises(ValueError, match="not bound"):
        bind_steady_process(record, entry, evidence, hash_file=lambda p: p)
