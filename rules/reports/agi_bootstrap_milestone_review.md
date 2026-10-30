# AGI Bootstrap Milestone Review

Date: 2026-10-29

## Artifact review (completed-milestone outputs)

| Milestone | Artifact(s) | Status | Notes |
|---|---|---|---|
| Baseline contract outputs | `rules/reports/agi_bootstrap_baseline_v1.json` | ✅ Valid | Contract report generated; 24-case combined suite across 3 profiles. |
| Invariant curriculum outputs | `rules/reports/agi_bootstrap_invariant_curriculum_v1.json` | ⚠️ Critical regression | `all_gates_green=false`; halted on `invariant-core-finality` (`lt-heavy-core-all-valid-solves`: 3/4). |
| Metarule expansion validation | `rules/validate_metarule_integration.py` | ✅ Pass | Validation script passes. |
| SAR objective validation | `rules/validate_sar_objective.py` | ✅ Pass | Symbolic-gate coupling behavior validated. |
| Teacher-ingestion validation | `rules/validate_teacher_ingestion.py` | ✅ Pass | Treatment-arm ingestion + proposal-only lifecycle validated. |
| A/B efficiency report | `rules/reports/agi_bootstrap_ab_efficiency_report.json` (+ control/treatment JSONs) | ✅ Valid | Quality deltas are flat; compute delta near parity. |
| Curriculum widening report | `rules/reports/agi_bootstrap_curriculum_widening_v1.json` | ✅ Valid | `all_gates_green=true` across 5 slices, including non-GSM transfer slices. |

## Exit-criteria evaluation

### 1) Progress toward 100x efficiency target
**Not achieved.**

- A/B macro delta (`treatment - control`):  
  - `solve_rate_mean`: `0.0`  
  - `symbolic_valid_rate_mean`: `0.0`  
  - `valid_solves_per_sample_mean`: `0.0`  
  - `wall_time_sec_total_sum`: `+0.097s` (near-parity, not a reduction signal)
- Best observed `wall_time_sec_per_valid_solve` values are ~`1.14–1.51s` depending on slice/profile, indicating incremental (single-digit to low-multiple) movement, not two orders of magnitude.

### 2) Symbolic validity preservation
**Partially achieved.**

- Preserved strongly in widening for hybrid and LT-heavy on transfer slices:
  - `tinystories-state-tracking-compact`: hybrid `3/3` valid solves, LT-heavy `3/3`
  - `mixed-narrative-symbolic-compact`: hybrid `3/3`, LT-heavy `3/3`
- Regression present in compact invariant gate run:
  - `invariant-core-finality`: LT-heavy valid solves `3/4` (blocking gate failure)
- ILP-heavy symbolic validity remains unstable in some widening contexts (e.g., mixed symbolic-text: `0.333` symbolic-valid-rate, monitor gate only).

### 3) Transfer to at least one non-GSM slice
**Achieved.**

- Non-GSM slices passed with `all_gates_green=true` in widening report:
  - `tinystories-state-tracking-compact`
  - `mixed-narrative-symbolic-compact`

## Quantitative metrics observed

- Baseline contract (`agi_bootstrap_baseline_v1.json`, combined 24-case):
  - LT-heavy: solve `0.833`, symbolic-valid `0.958`, sec/valid `1.896`
  - Hybrid: solve `1.000`, symbolic-valid `1.000`, sec/valid `1.487`
  - ILP-heavy: solve `1.000`, symbolic-valid `0.708`, sec/valid `1.508`
- A/B compact slice (`agi_bootstrap_ab_efficiency_report.json`):
  - Macro quality metrics unchanged between control and treatment.
  - Total wall-time delta across all profiles: `+0.097s`.
- Widening (`agi_bootstrap_curriculum_widening_v1.json`):
  - `all_gates_green=true`
  - 5/5 slices executed and green.

## Blockers / risks

1. **Critical invariant regression:** compact invariant contract is not green due to LT-heavy core-finality miss (`3/4`), preventing clean “all milestones green” claim.
2. **Efficiency gap:** no demonstrated trajectory toward the 100x objective from current A/B evidence.
3. **ILP-heavy symbolic stability risk:** low symbolic-valid-rate on mixed symbolic-text slice (currently non-blocking) may threaten wider transfer reliability.

## Recommended next milestone actions

1. **Fix invariant regression first:** root-cause LT-heavy miss in `invariant-core-finality` and restore 4/4 valid solves.
2. **Introduce explicit 100x ledger:** track `cost_per_valid_solve` with fixed hardware/runtime budget and mandatory relative-improvement targets per milestone.
3. **Expand efficiency A/B breadth:** rerun A/B on widening slices (not only compact combined slice) with stable seeds and repeated trials.
4. **Harden symbolic transfer gates:** raise ILP-heavy mixed-slice symbolic gate from monitor to blocking once consistently above floor.
