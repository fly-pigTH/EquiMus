import mujoco
import numpy as np

def get_site_distance(model, data, site1_name, site2_name):
    # 获取两个site的位置
    site1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site1_name)
    site2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site2_name)
    
    site1_pos = data.site_xpos[site1_id]
    site2_pos = data.site_xpos[site2_id]
    
    # 计算欧氏距离
    distance = np.linalg.norm(site1_pos - site2_pos)
    return distance

# 使用示例
model = mujoco.MjModel.from_xml_path("dog2_4singleLeg.xml")
data = mujoco.MjData(model)

# 更新模拟状态
mujoco.mj_forward(model, data)

# 获取距离
distance = get_site_distance(model, data, "RB_MAA_START", "RB_MAA_END")
distance2 = get_site_distance(model, data, "RB_BAA_START", "RB_BAA_END")
print(f"Distance between sites: {distance}")
print(f"Distance between sites: {distance2}")
