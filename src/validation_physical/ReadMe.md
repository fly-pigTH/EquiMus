# Validation of Physical Models

## File Tree

```text
.
├── dynamic
│   ├── data
│   │   ├── ToCenter_2025-02-20_21-06-20_A2B_data.csv: [Physical trajectory.]
│   │   └── TriStatic_2025-01-21_19-26-44_A2B_data.csv: [Physical trajectory, tracking 3 steps pressure steps.]
│   ├── dynamic_sysid.ipynb
│   ├── figure
│   │   ├── sensitivity_analysis.pdf
│   │   ├── sensitivity_analysis.png
│   │   └── tri_trajectory.png
│   ├── log
│   │   ├── baseline_analysis.csv
│   │   ├── sensitivity_analysis.csv
│   │   └── triangle_error.csv
│   ├── scripts
│   ├── sensitivity_analysis_with_baseline.ipynb
│   ├── trajectory_track_triangle.ipynb
│   └── video
│       └── output_video.mp4
├── ReadMe.md
├── static
│   ├── data
│   │   └── real_static_state
│   ├── figure
│   ├── log
│   │   ├── regression_pressure_error_summary(effective).csv
│   │   ├── RMSE_error_summary(all).csv
│   │   ├── RMSE_error_summary(effective).csv
│   │   ├── simulated_static_state_error_all.csv
│   │   ├── simulated_static_state_error_effective.csv
│   │   └── sysid_parameters.csv
│   ├── scripts
│   │   └── sysid_static_effective.py
│   └── video
└── static_qpos_analysis.py
```

## Run
1. Run `./static_qpos_analysis.py` to analysis the static states of the robotic leg, both for effective and all data.
2. Run `./static/scripts/sysid_static_effective.py`to identified the effective static parameters.
3. Run `./dynamic/dynamic_sysid.ipynb` to identify the effective dynamic parameters.
4. Run `./dynamic/sensitivity_analysis_with_baseline.ipynb` to perform sensitivity analysis and baseline experiments.
5. Run `./dynamic/trajectory_track_triangle.ipynb` to analyze trajectory tracking.

## Results
- Sensitivity Analysis
![alt text](dynamic/figure/sensitivity_analysis.png)

- Trajectory Tracking
![alt text](dynamic/figure/tri_trajectory.png)

