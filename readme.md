# EquiMus: Musculoskeletal Equivalent Dynamic Modeling and Simulation for Rigid-soft Hybrid Robots with Linear Elastic Actuators

## About This Work
Leveraging the full potential of soft robots relies heavily on dynamic modeling and control, which remains challenging due to their complex constitutive relationships and real-world operational scenarios. Bio-inspired musculoskeletal robots, which integrate rigid skeletons with soft actuators, combine the advantages of heavy load-bearing capacity and inherent flexibility. Although actuation dynamics has been studied through experimental methods and surrogate models, accurate and effective modeling and simulation still pose a significant challenge when soft actuators are applied at a large scale, especially in hybrid rigid-soft robots with continuously distributed mass, kinematic loops and diverse motion modes.

To address this issue, this study introduces EquiMus, a musculoskeletal equivalent dynamic modeling and MuJoCo-based simulation for rigid-soft hybrid robots with linear elastic actuators. The equivalence and effectiveness are proven in detail and examined through simulated and real experiments on a bionic robotic leg. Based on the energy-equivalent model and simulation, we do some explorations in model-based and data-driven control algorithms including reinforcement learning.

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
