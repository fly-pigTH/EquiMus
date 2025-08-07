# EquiMus: Musculoskeletal Equivalent Dynamic Modeling and Simulation for Rigid-soft Hybrid Robots with Linear Elastic Actuators
> Struggling to Perfect

## About This Work
Leveraging the full potential of soft robots relies heavily on dynamic modeling and control, which remains challenging due to their complex constitutive relationships and real-world operational scenarios. Bio-inspired musculoskeletal robots, which integrate rigid skeletons with soft actuators, combine the advantages of heavy load-bearing capacity and inherent flexibility. Although actuation dynamics has been studied through experimental methods and surrogate models, accurate and effective modeling and simulation still pose a significant challenge when soft actuators are applied at a large scale, especially in hybrid rigid-soft robots with continuously distributed mass, kinematic loops and diverse motion modes.

To address this issue, this study introduces EquiMus, a musculoskeletal equivalent dynamic modeling and MuJoCo-based simulation for rigid-soft hybrid robots with linear elastic actuators. The equivalence and effectiveness are proven in detail and examined through simulated and real experiments on a bionic robotic leg. Based on the energy-equivalent model and simulation, we do some explorations in model-based and data-driven control algorithms including reinforcement learning.


---

## WorkFlow
1. data-src-log-src(post process)
2. Experiments
   1. `auto_record.py`: auto record in a table
      1. 为了解决导入问题，采用模块化运行方式
      ```bash
      python -m src.run
      ```
      这样做太麻烦，目前直接认为代码层很低
   2. 

## Structures
1. check multi
2. install in the Win 11 Omen!

MusGym

In the future, we will develop this repo into a more general-purpose tool, MUSEUMS.
update real control.
fight