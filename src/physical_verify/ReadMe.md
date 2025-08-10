
## File Tree
- `static_qpos_analysis.py`
- `static_parameter_identify_effective.ipynb`
- `static_parameter_identify_all.ipynb`

## Run
1. Run `static_qpos_analysis.py` to analysis the static states of the robotic leg, both for effective and all data.
2. Run `static_parameter_identify_effective.ipynb`
3. Run `static_parameter_identify_all.ipynb`

## Static
python static_qpos_analysis.py
cd static
python scripts/sysid_static_effective.py

## dynamic
dynamic_sysid.ipynb
sensitivity_analysis_with_baseline.ipynb
trajectory_track_triangle.ipynb