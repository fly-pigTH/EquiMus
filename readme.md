# EquiMus: Musculoskeletal Equivalent Dynamic Modeling and Simulation for Rigid-soft Hybrid Robots with Linear Elastic Actuators

## About This Work
Leveraging the full potential of soft robots relies heavily on dynamic modeling and control, which remains challenging due to their complex constitutive relationships and real-world operational scenarios. Bio-inspired musculoskeletal robots, which integrate rigid skeletons with soft actuators, combine the advantages of heavy load-bearing capacity and inherent flexibility. Although actuation dynamics has been studied through experimental methods and surrogate models, accurate and effective modeling and simulation still pose a significant challenge when soft actuators are applied at a large scale, especially in hybrid rigid-soft robots with continuously distributed mass, kinematic loops and diverse motion modes.

To address this issue, this study introduces EquiMus, a musculoskeletal equivalent dynamic modeling and MuJoCo-based simulation for rigid-soft hybrid robots with linear elastic actuators. The equivalence and effectiveness are proven in detail and examined through simulated and real experiments on a bionic robotic leg. Based on the energy-equivalent model and simulation, we do some explorations in model-based and data-driven control algorithms including reinforcement learning.

## Run Demo
To run the demo, execute the following command in the terminal:
```bash
# show help message
$ python demo/demo.py -h
# run the simulation of 2DOF model or 3DOF model on Mac/Win/Linux
$ mjpython demo/demo.py # if the os is Mac
$ python demo/demo.py # if the os is Windows
```

If running successfully, you should see the simulation window pop up, which is the interactive viewer of [MuJoCo passive-viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

| ![3DOF](./demo/figure/resized//demo_3DOF.png) | ![2DOF](./demo/figure/resized//demo_2DOF.png) |
|:--:|:--:|
| **(a) Morphology with 3DOF** | **(b) Morphology with 2DOF** |

In the interactive viewer, you can see the simulation of the 2DOF and 3DOF models. And following [the MuJoCo conventions of the GUI interaction](https://www.youtube.com/watch?v=P83tKA1iz2Y), you can interact with the models using the mouse and keyboard.

| ![Body labels](./demo/figure/resized//demo_2DOF_body.png) | ![Geom labels](./demo/figure/resized//demo_2DOF_geom.png) | ![Joint labels](./demo/figure/resized//demo_2DOF_joint.png) |
|:--:|:--:|:--:|
| **(a) Body labels** | **(b) Geom labels** | **(c) Joint labels** |


## Project structure
> obey the flow of the manuscript.
- [x] models: MuJoCo `XML` files
- [x] src
  - [x] theory (omit temporarily, mabe used the old version of the `theory` part in the paper.)
    - [x] [doc](src/theory/doc.md) remains empty.
  - [x] [validation_simulation](src/validation_simulation/ReadMe.md)
    - [x] [static](src/validation_simulation/static/ReadMe.md)
    - [x] [dynamic](src/validation_simulation/dynamic/ReadMe.md)
    - [x] [morphology](src/validation_simulation/morphology/ReadMe.md)
  - [x] [validation_physical](src/validation_physical/ReadMe.md)
    - [x] static
    - [x] dynamic
  - [x] application (show the potential of our method)
    - [x] [PID_AutoTuning](src/application/PID_AutoTuning/ReadMe.md)
    - [x] [RL_BallKicking](src/application/RL_BallKicking/ReadMe.md)
- [x] utils
  - [x] experiment.py
  - [x] experiment_topology.py
- [x] ReadMe.md

## Requirements

## TODO
- A demo to show the EquiMus method
