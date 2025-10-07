# Morphology Generalization Validation

## How to Run

1. **Theory Part**: Explore the theoretical foundation in the following notebook:
```bash
./theory_2DOF/main_2DOF.ipynb
```

2. **Build 3-DOF Model with the EquiMus Method**:  
From the repository root, execute:
```bash
cd src/validation_simulation/
python ./morphology/model_build/main.py
```

3. **Build 3-DOF Model with the Analytical Model (SymPy)**:  
Open and run `main.ipynb` to derive the systematic dynamics and simulate the response.

4. **Analyze Results**:  
Compare the results of the two models by running:
```bash
python analysis.py
```

## Results

- Simulation result figures are stored in the `figure/` directory.
- RMSE values for the two models are logged in `./morphology/test_data/rmse_log.csv`.
- The 3-DOF validation result is visualized in the following image:

![Joint Angle Comparison](./figure/joint_angle_comparison.jpg)
Step response of the 3-DOF model, showing the comparison between the EquiMus method and the analytical model.
