import mujoco

path_dog1101 = "/Users/flypig/Documents/Coding/MujocoLearn/1101/1101/mydog110.xml"
path_dog24 = "/Users/flypig/Documents/Coding/MujocoLearn/v2_4/urdf/dog2_4singleLeg.xml"
# 加载 XML 模型
model = mujoco.MjModel.from_xml_path(path_dog24)  # 替换为实际 XML 文件路径
data = mujoco.MjData(model)

# 遍历并获取各 body 的质量
for i in range(model.nbody):
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    body_mass = model.body_mass[i]
    print(f"{body_name}: {body_mass:.3f}")