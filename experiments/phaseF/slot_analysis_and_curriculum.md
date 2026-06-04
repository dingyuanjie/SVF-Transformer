# Phase F: Slot Analysis And Curriculum Recall

## Goal

- Determine whether `Persistent Core` slots actually specialize.
- Replace all-or-nothing recall tasks with staged curriculum levels.

## Slot Analysis

- Script: `analyze_core_slots.py`
- Inputs: `core_traces/*.json`
- Outputs:
  - `slot_usage_entropy`
  - `dominant_slot_fraction`
  - `mean_offdiag_cosine`
  - `mean_slot_norm_std`
  - `mean_core_norm`

### Suggested command

```bash
python analyze_core_slots.py --trace-dir outputs/experiments/phaseE_multientity_recall_formal/core_traces --output-dir outputs/experiments/phaseF_slot_analysis
```

## Curriculum Recall

- Script: `run_phase_f_curriculum.py`
- Levels:
  - `level1`: 2 entities, delay 512
  - `level2`: 2 entities, delay 1024
  - `level3`: 3 entities, delay 1024
  - `level4`: 4 entities, delay 2048

### Suggested command

```bash
python run_phase_f_curriculum.py --levels level1 level2 level3 level4 --variants baseline persistent_core core_dynamics svf --seeds 42 43 44 --save-core-traces --output-root outputs/experiments/phaseF_curriculum_formal
```

## Main Questions

- Do slot similarities remain close to `1.0`, indicating collapse?
- Does curriculum training reveal a usable capability boundary?
- Does `core_dynamics` improve slot diversity compared with `persistent_core`?
