import json

from cheap_routing_mapping import analyze_experiment_root, effective_slot_indices


def test_effective_slot_indices_uses_count_and_weight_thresholds():
    active = effective_slot_indices(
        [6, 0, 4, 0],
        [0.55, 0.05, 0.35, 0.05],
        min_count_fraction=0.10,
        min_mean_weight=0.10,
    )
    assert active == [0, 2]


def test_build_question_verdicts_detects_two_slot_collapse(tmp_path):
    root = create_trace_tree_for_collapse_case(tmp_path)
    summaries, verdicts = analyze_experiment_root(
        root,
        min_count_fraction=0.10,
        min_mean_weight=0.10,
    )

    assert len(summaries) == 2
    assert len(verdicts) == 1
    assert verdicts[0].answer is True
    assert verdicts[0].slot_2_is_fully_used is True
    assert verdicts[0].larger_slot_counts_collapse_to_two is True


def create_trace_tree_for_collapse_case(tmp_path):
    slots_2_dir = tmp_path / "specialized_core" / "nobalance" / "slots_2" / "core_traces"
    slots_4_dir = tmp_path / "specialized_core" / "nobalance" / "slots_4" / "core_traces"
    slots_2_dir.mkdir(parents=True, exist_ok=True)
    slots_4_dir.mkdir(parents=True, exist_ok=True)

    write_trace(
        slots_2_dir / "delay32_specialized_core_seed42.json",
        [
            make_entry("alice", "color", [0.90, 0.10], [0.92, 0.08]),
            make_entry("bob", "color", [0.15, 0.85], [0.20, 0.80]),
            make_entry("carol", "color", [0.88, 0.12], [0.86, 0.14]),
            make_entry("dave", "color", [0.10, 0.90], [0.12, 0.88]),
        ],
    )
    write_trace(
        slots_4_dir / "delay32_specialized_core_seed42.json",
        [
            make_entry("alice", "color", [0.80, 0.05, 0.05, 0.10], [0.75, 0.05, 0.05, 0.15]),
            make_entry("bob", "color", [0.15, 0.05, 0.05, 0.75], [0.10, 0.05, 0.05, 0.80]),
            make_entry("carol", "color", [0.82, 0.04, 0.04, 0.10], [0.78, 0.04, 0.04, 0.14]),
            make_entry("dave", "color", [0.10, 0.05, 0.05, 0.80], [0.12, 0.05, 0.05, 0.78]),
        ],
    )
    return tmp_path


def write_trace(path, entries):
    payload = {
        "trace_entries": entries,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_entry(query_name, query_field, write_weights, read_weights):
    slot_count = len(write_weights)
    return {
        "query_name": query_name,
        "query_field": query_field,
        "slot_norms": write_weights,
        "slot_routing_weights": write_weights,
        "slot_read_weights": read_weights,
        "core_state": [[0.0] for _ in range(slot_count)],
    }
