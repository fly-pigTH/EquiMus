# 仿真静力学实验
设置一个测试参数（待标定的静力学参数组），对比【理论静力学模型】和【Mujoco等效模型】，【Matlab 模型】需要调用（这里直接忽略吧，动力学部分再调用，相当于和【理论静力学模型】平行的关系）

- 简化：采用真实模型 + fixed数据，静力学参数可变，随机化检验
- 已经验证可复现性！

## Run
python main.py
python video/conj_video.py

## data and log