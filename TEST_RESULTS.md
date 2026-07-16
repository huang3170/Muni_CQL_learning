# Validation Results

The v4 spread-relative update was validated with:

- Python compilation across the full codebase.
- 12 unit tests, including spread scaling, absolute-cap masking, event ordering, fill capacity and greedy-policy leakage controls.
- Offline CQL smoke training with the 36-action spread-ratio grid.
- End-to-end simulator-online smoke training with generated replay data.
- Standalone RL-versus-greedy evaluation on held-out generated replay episodes.

These are software checks only. They do not validate market realism or economic performance of the placeholder simulator models.
