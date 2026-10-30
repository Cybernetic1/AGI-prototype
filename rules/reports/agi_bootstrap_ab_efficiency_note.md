# AGI Bootstrap A/B Efficiency (compact slice)

- Slice: 8 mixed finality+template families, 3 profiles, max_cycles=2.
- Control: transfer metarules OFF, SAR OFF, teacher ingestion OFF.
- Treatment: transfer metarules ON + SAR ON + teacher ingestion treatment arm ON.

| Profile | Solve rate C→T | Symbolic valid rate C→T | Valid solves/sample C→T | Wall time total C→T (s) |
|---|---:|---:|---:|---:|
| lt-heavy | 0.750 → 0.750 | 1.000 → 1.000 | 0.750 → 0.750 | 12.63 → 12.76 |
| hybrid | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 12.04 → 11.97 |
| ilp-heavy | 1.000 → 1.000 | 0.750 → 0.750 | 0.750 → 0.750 | 8.53 → 8.58 |

- Finding: quality metrics were unchanged on this compact slice; treatment remained near compute-parity (macro wall-time delta +0.097s across all profiles).
- Teacher ingestion smoke signal: accepted 3/4 proposals (dropped_by_volume=1).
