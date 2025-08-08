# EquiMus: Musculoskeletal Equivalent Dynamic Modeling and Simulation for Rigid-soft Hybrid Robots with Linear Elastic Actuators
> Struggling to Perfect

## About This Work
Leveraging the full potential of soft robots relies heavily on dynamic modeling and control, which remains challenging due to their complex constitutive relationships and real-world operational scenarios. Bio-inspired musculoskeletal robots, which integrate rigid skeletons with soft actuators, combine the advantages of heavy load-bearing capacity and inherent flexibility. Although actuation dynamics has been studied through experimental methods and surrogate models, accurate and effective modeling and simulation still pose a significant challenge when soft actuators are applied at a large scale, especially in hybrid rigid-soft robots with continuously distributed mass, kinematic loops and diverse motion modes.

To address this issue, this study introduces EquiMus, a musculoskeletal equivalent dynamic modeling and MuJoCo-based simulation for rigid-soft hybrid robots with linear elastic actuators. The equivalence and effectiveness are proven in detail and examined through simulated and real experiments on a bionic robotic leg. Based on the energy-equivalent model and simulation, we do some explorations in model-based and data-driven control algorithms including reinforcement learning.

## Project structure
> obey the flow of the manuscript.
- [ ] models: MuJoCo `XML` files
- [ ] src
  - [ ] theory (omit temporarily, mabe used the old version of the `theory` part in the paper.)
    - [ ] [doc](src/theory/doc.md)
  - [x] ❌ Validation (on the correctnes of our method)
  - [ ] validation_simulation
    - [ ] static
    - [ ] dynamic
    - [ ] topology
  - [ ] validation_physical
    - [ ] static
    - [ ] dynamic
  - [ ] application (show the potential of our method)
    - [ ] PID_AutoTuning (run with Geek), 这部分缺少优化的seed，最终结果有区别。需要修复minimize的seed～
    - [ ] RL_BallKicking (run with Geek)
- [ ] utils
- [ ] ReadMe.md

## Design Rule

1. 模块封装-测试-main函数
2. 数据存储
   1. 为了ReadMe引用方便，全部采用固定命名，复现时会直接覆盖全部数据文件
3. 实验逻辑：生成数据（比如生成模型-仿真-优化...）——数据分析
   1. ❌ 考虑main中传入args来进行单模块调试
   2. ✅ 多个模块分别开发，main中进行最终集成，作为用户使用接口
   3. 数据文件一键清洁 bash


## TODO
- [ ] Project structure design.
- [ ] Finish all experiments
- [ ] Test on Windows platform
- [ ] Homepage of the whole project

## WorkFlow
1. data-src-log-src(post process)
2. Experiments
   1. `auto_record.py`: auto record in a table
      1. 为了解决导入问题，采用模块化运行方式
      ```bash
      python -m src.run
      ```
      这样做太麻烦，目前直接认为代码层很低



## Learn

### 使用utils

~~~python
import rootpath
ROOT_DIR = rootpath.detect()   # Get the root directory of the project (.git)
sys.path.append(str(Path(ROOT_DIR)))
from utils.experiment import MujocoExperiment
~~~

### Internal Import

~~~python
--a.py
--b.py
-main.py: 希望调用a，但是a内部调用了b（使用相对路径，import b）
目前暂时将ab在的目录加入到了sys.path
[TODO] 更好的方法：直接在main中处理？
~~~

### Use datetime to create the logfile

~~~python
dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
folder_path = CURRENT_DIR / "video" / dt_str
os.makedirs(folder_path, exist_ok=False)  # 如果文件名称冲突，报错!
~~~



## Structures
1. check multi
2. install in the Win 11 Omen!

MusGym

In the future, we will develop this repo into a more general-purpose tool, MUSEUMS.
update real control.
fight