# Morphology Generalization Validation

## Run
- theory part: run `./theory_2DOF/main_2DOF.ipynb`
- build 3 DOF Model with our EquiMus method
~~~bash
python ./morphology/model_build/main.py
~~~
- build 3 DOF Model with analytical model with SymPy

run `main.ipynb` (Derivation of the systematic dynamics, and simulate the response).

~~~bash
python analysis.py  # see the results of two models
~~~

## Results
- `figure/` contains the figures generated from the simulation results.
- `./morphology/test_data/rmse_log.csv` contains the RMSE of the two models.