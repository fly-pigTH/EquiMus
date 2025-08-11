# build random mj model accordingly
# final version 

import mediapy as media
import random
import math
import numpy as np
import mujoco as mj
from itertools import accumulate
import matplotlib.pyplot as plt
import mediapy as media
from tqdm import tqdm
import time
import pandas as pd
# solve the actuator
import xml.etree.ElementTree as ET


import rootpath
import sys
ROOT_DIR = rootpath.detect()
from pathlib import Path

sys.path.insert(0, str(ROOT_DIR))
from utils.experiment import MujocoExperiment

CURRENT_DIR = Path(__file__).resolve().parent
EXP_DIR = CURRENT_DIR.parent

DEFAULT_TENDON_SIZE = 0.003
DEFAULT_TENDON_STIFFNESS = 0
DEFAULT_BALL_SIZE = 0.005     # for the mass point visualization

# Tool kit for Ipynb
class visual_tool():
    def render(model, data=None, height=300, camera=-1):
        if data is None:
            data = mj.mjData(model)
        with mj.Renderer(model, 480, 640) as renderer:
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera="closeup")
            media.show_image(renderer.render(), height=height)

    from IPython.display import clear_output
    clear_output()

    def print_xml(xml_string):
        import pygments
        from IPython.display import HTML, display
        formatter = pygments.formatters.HtmlFormatter()
        lexer = pygments.lexers.XmlLexer()
        highlighted = pygments.highlight(xml_string, lexer, formatter)
        display(HTML(f"<style>{formatter.get_style_defs()}</style>{highlighted}"))

def add_motors_to_spec(spec: mj.MjSpec, joints: list[str], 
                            ctrlrange=(-200, 200), gear=1.0, prefix="motor_") -> mj.MjSpec:
    """
    Add motor actuators to the MjSpec model by parsing and modifying the XML.

    Args:
        spec: An instance of MjSpec.
        joints: A list of joint names to be controlled.
        ctrlrange: Control range, e.g., (-200, 200).
        gear: Motor gear value.
        prefix: Prefix for naming the motors.

    Returns:
        A newly constructed MjSpec object containing the motor actuators.
    """
    root = ET.fromstring(spec.to_xml())

    # search the <actuator> node. If not exist, we will create one
    actuator_node = root.find("actuator")
    if actuator_node is None:
        actuator_node = ET.SubElement(root, "actuator")

    # add motor actuators
    for joint_name in joints:
        motor = ET.SubElement(actuator_node, "motor")
        motor.set("name", prefix + joint_name)
        motor.set("joint", joint_name)
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", f"{ctrlrange[0]} {ctrlrange[1]}")
        motor.set("gear", str(gear))

    # rebuild the spec from the modified_xml
    modified_xml = ET.tostring(root, encoding="unicode")
    return mj.MjSpec.from_string(modified_xml)

def add_muscle_to_spec(spec, muscle_cfg):
    # unpack
    name = muscle_cfg['name']
    mass = muscle_cfg['mass']
    stiffness = muscle_cfg['stiffness']
    damping = muscle_cfg['damping']
    rest_length = muscle_cfg['rest_length']
    init_length = muscle_cfg['init_length']
    # where it connects
    base_pos = muscle_cfg.get('base_pos')
    base_axisangle = muscle_cfg.get('base_axisangle')

    # preprocess
    uLink_mass, mLink_mass, lLink_mass = mass/6, mass*2/3, mass/6
    mSlideJoint_stiffness, lSlideJoint_stiffness = stiffness*2, stiffness*2
    mSlideJoint_damping, lSlideJoint_damping = damping*2, damping*2
    
    # NOTE: need to keep the axis of the muscle to be [0 0 -1] relative to the parent frame
    mLink_pos = [0, 0, -init_length/2]
    lLink_pos = [0, 0, -init_length/2]
    # NOTE: 'spring_ref' is the relative length of the muscle when it is at rest, to the init length
    init_relative_length = rest_length - init_length
    mSlideJoint_initRelaLength, lSlideJoint_initRelaLength = init_relative_length/2, init_relative_length/2

    # upper part
    _body = spec.add_body(name=f"{name}_uLink", pos=base_pos, axisangle=base_axisangle)
    print(f"base_axisangle: {base_axisangle}")
    _body.add_joint(name = f"{name}_uRotJoint",
        type=mj.mjtJoint.mjJNT_HINGE,
        pos = [0, 0, 0],
        axis = [1, 0, 0])
        # NOTE: we assume the motion is in the YOZ plane, so any rotate joint has a x-rotJoint
    _body.add_geom(name=f"{name}_uGeom",
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[DEFAULT_BALL_SIZE, 0, 0],
                mass = uLink_mass,
                contype = 0,
                conaffinity = 0)
    _body.add_site(name=f"{name}_uSite", pos = [0, 0, 0])

    # middle part
    _body = _body.add_body(name=f"{name}_mLink", pos=mLink_pos)
    # NOTE: need to keep the axis of the muscle to be [0 0 -1] relative to the parent frame
    _body.add_joint(name = f"{name}_mSlideJoint",
        type=mj.mjtJoint.mjJNT_SLIDE,
        pos = [0, 0, 0],
        axis = [0, 0, -1],      
        springref = mSlideJoint_initRelaLength,
        stiffness = mSlideJoint_stiffness,
        damping = mSlideJoint_damping,
        )
    _body.add_geom(name=f"{name}_mGeom",
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[DEFAULT_BALL_SIZE, 0, 0],
                mass = mLink_mass,
                # NOTE: ignore all collision in this theoretical verification
                contype = 0,
                conaffinity = 0)
    _body.add_site(name=f"{name}_mSite", pos = [0, 0, 0])

    # lower part
    _body = _body.add_body(name=f"{name}_lLink", pos=lLink_pos)
    _body.add_joint(name = f"{name}_lSlideJoint",
        type=mj.mjtJoint.mjJNT_SLIDE,
        pos = [0, 0, 0],
        axis = [0, 0, -1],
        springref = lSlideJoint_initRelaLength,
        stiffness = lSlideJoint_stiffness,
        damping = lSlideJoint_damping,
        )
    _body.add_geom(name=f"{name}_lGeom",
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[DEFAULT_BALL_SIZE, 0, 0],
                mass = lLink_mass,
                contype = 0,
                conaffinity = 0)
    _body.add_site(name=f"{name}_lSite", pos = [0, 0, 0])

def add_tendon_to_spec(spec, tendon_cfg):
    _tendon = spec.add_tendon(name=tendon_cfg['name'], stiffness=tendon_cfg['stiffness'], width=tendon_cfg['width'])
    _tendon.wrap_site(tendon_cfg['site1_name'])
    _tendon.wrap_site(tendon_cfg['site2_name'])

def add_equality_to_spec(spec, equality_cfg):
    ''' Add an equality constraint to the spec. Joint equality is used to enforce the equality of two joints. '''
    spec.add_equality(name=equality_cfg['name'], type=equality_cfg['type'], 
                      name1=equality_cfg['name1'], name2=equality_cfg['name2'], solimp=equality_cfg['solimp'])

def add_muscle_fromto(spec, start_body_name, end_body_name, muscle_cfg):
    # TODO: Base manually here
    start_body = spec.body(start_body_name)
    add_muscle_to_spec(start_body, muscle_cfg)

    tendon_cfg = {
        'name': f"tendon_{muscle_cfg['name']}",
        'width': DEFAULT_TENDON_SIZE,
        'stiffness': DEFAULT_TENDON_STIFFNESS,
        'site1_name': f"{muscle_cfg['name']}_uSite",
        'site2_name': f"{muscle_cfg['name']}_lSite",
    }
    add_tendon_to_spec(spec, tendon_cfg)

    equality_cfg = {
        'name': f"{muscle_cfg['name']}_equality",
        'type': mj.mjtEq.mjEQ_JOINT,
        'name1': f"{muscle_cfg['name']}_mSlideJoint",
        'name2': f"{muscle_cfg['name']}_lSlideJoint",
        'solimp': [0.95, 0.99, 0.00001, 0.1, 2],
    }
    add_equality_to_spec(spec, equality_cfg)

    spec.add_equality(name=f"_{muscle_cfg['name']}_equality_connect", type=mj.mjtEq.mjEQ_CONNECT, name1=f"{muscle_cfg['name']}_lLink", name2=end_body_name, solimp=[0.95, 0.99, 0.00001, 0.1, 2], objtype=mj.mjtObj.mjOBJ_BODY)

    muscle_jnt_names = [f"{muscle_cfg['name']}_mSlideJoint", f"{muscle_cfg['name']}_lSlideJoint"]
    spec_new = add_motors_to_spec(spec, muscle_jnt_names, ctrlrange=(-200, 200), gear=1.0, prefix='motor_')
    return spec_new

# fix the anchor
def set_anchor_to_zero(spec: mj.MjSpec):
    """ write all <connect> <anchor> to '0 0 0' """
    root = ET.fromstring(spec.to_xml())     # get the string
    # search for all connect labels
    for conn in root.findall(".//connect"):
        conn.set("anchor", "0 0 0")
    # transfer to xml, and finally in mj.spec
    xml_str_modified = ET.tostring(root, encoding='unicode')
    return mj.MjSpec.from_string(xml_str_modified)

if __name__ == "__main__":

    static_model = '''
    <mujoco model="v2_4">
        <compiler angle="radian"/>
        <option timestep="0.001" iterations="2000" gravity="0 0 -9.8" tolerance="1e-10"/>
        
        <default>
            <geom density="418"/>
        </default>

        <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
                rgb2=".2 .3 .4" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
        </asset>

        <worldbody>
            <light name="top" pos="0 0 5" diffuse="1 1 1" ambient="0.5 0.5 0.5"/>
            <camera name="closeup" pos="1.483 -0.033 0.580" xyaxes="0.040 0.999 -0.000 -0.129 0.005 0.992"/>
            <geom size="10 10 .01" type="plane" material="grid" friction="1 0.005 0.0001"/>

            <body name="Base" pos="0 0 0.8">
                <geom size="0.01 0.2 0.01" type="box" rgba="0.8 0 0 0.8" mass="0.155" contype="0" conaffinity="0"/>
            </body>
        </worldbody>

    </mujoco>
    '''

    spec = mj.MjSpec.from_string(static_model)
    base_body = spec.body('Base')

    # Step 1: build random bone-like rigid links (ofc, and bodies)
    link_num = 3 # ignore the base
    muscle_num = 3
    links_list = list(range(link_num)) + [-1]    # -1 represents base
    link_lengths = [0.25] * 3 + [0.4]  # assume base has -0.2->0.2
    link_masses = [0.086, 0.1033, 0.1033] + [0.1]  # assume base has 0.1
    cumsum = list(accumulate(link_lengths))
    links_origin_vertical = [0] + list(accumulate(link_lengths))[:-2] + [0]   # using -1 to get the base pos
    print(f"links_origin_vertical: {links_origin_vertical}")

    link_name_from_id = {
        -1: 'Base',
        0: 'link0_body',
        1: 'link1_body',
        2: 'link2_body',
        # 3: 'link3_body', 
    }

    for i in range(link_num):
        link_name = 'link' + str(i) + '_body'
        geom_name = 'link' + str(i) + '_geom'
        mass = link_masses[i]
        if i == 0: 
            pos = [0, 0, 0]         # For Link0, its parents link is the base body
            _body = base_body.add_body(name=link_name, pos=pos)
        else: 
            pos = [0, 0, -link_lengths[i-1]]
            parent_link_name = 'link' + str(i-1) + '_body'
            parent_link_body = spec.body(parent_link_name)
            _body = parent_link_body.add_body(name=link_name, pos=pos)
        # add geom
        _body.add_geom(name=geom_name, size=[0.005, 0, 0], type=mj.mjtGeom.mjGEOM_CAPSULE, rgba=[0.8, 0, 0, 0.8], mass=mass, fromto = [0, 0, 0, 0, 0, -link_lengths[i]], contype=0, conaffinity=0)
        # add joint
        _body.add_joint(name='link' + str(i) + '_joint', type=mj.mjtJoint.mjJNT_HINGE, pos=[0, 0, 0], axis=[1, 0, 0], range=[-1.57, 1.57])


    # Step 2: build random muscles to linked them

    factor_range = (0.5, 1.5)  # factor range for randomization
    random.uniform(*factor_range)

    posHorizonal_factor_range = (-0.2, 0.2)
    posVertical_factor_range = (0.2, 0.8)

    # base
    base_posHorizonal_factor_range = np.array([0.2, 0.8]) - 0.5
    base_posVertical_factor_range = (-0.2, 0.2)

    muscle_length_factor_range = (0.8, 1.2)  # factor range for muscle length randomization
    muscle_stiffness_factor_range = (0.8, 1.2)  # factor range for muscle stiffness randomization
    muscle_damping_factor_range = (0.8, 1.2)  # factor range for muscle damping randomization

    # set_value here
    d_1 = 0.06
    d_2 = 0.10
    b_1 = 0.21213
    b_2 = 0.1
    beta_1 = 8.13 / 180 * np.pi
    beta_2 = 30 / 180 * np.pi
    x_3 = 0.1
    y_3 = 0.1
    x_4 = 0.1
    y_4 = 0.05

    m_3, m_4 = 0.18648, 0.27266
    k_3, k_4 = 637.52/2, 631.6/2
    l_10, l_20 = 0.174, 0.2562
    # damping 
    c_3, c_4 = 22.68/2, 21.8/2

    lm_3 = 0.1033
    mm_3 = 0.18
    mk_3 = k_4
    mc_3 = c_4
    l_30 = 0.2549509756


    for muscle_id in range(muscle_num):

        # assume link0 and link1, muscle3
        # selected = sorted(random.sample(links_list, 2))
        # anchor_id0 = selected[0]
        # anchor_id1 = selected[1]

        # manual set here
        if muscle_id == 0:
            anchor_id0 = -1
            anchor_id1 = 0

            pos0_x = d_1
            pos0_y = 0
            pos1_x = b_1 * math.cos(math.pi/2-beta_1)
            pos1_y = b_1 * math.sin(math.pi/2-beta_1)


        elif muscle_id == 1: 
            anchor_id0 = -1
            anchor_id1 = 1

            pos0_x = -d_2
            pos0_y = 0
            pos1_x = b_2 * math.cos(math.pi/2+beta_2)
            pos1_y = b_2 * math.sin(math.pi/2+beta_2) + links_origin_vertical[anchor_id1]

        else:
            anchor_id0 = 1
            anchor_id1 = 2

            pos0_x = -y_3
            pos0_y = x_3 + links_origin_vertical[anchor_id0]
            pos1_x = -y_4
            pos1_y = x_4 + links_origin_vertical[anchor_id1]

        # if anchor_id0 == -1:
        #     pos0_x = random.uniform(*base_posHorizonal_factor_range) * link_lengths[anchor_id0]
        #     pos0_y = random.uniform(*base_posVertical_factor_range) * link_lengths[anchor_id0] + links_origin_vertical[anchor_id0]
        #     print(pos0_x, pos0_y)
        # else:
        #     pos0_x = random.uniform(*posHorizonal_factor_range) * link_lengths[anchor_id0]
        #     pos0_y = random.uniform(*posVertical_factor_range) * link_lengths[anchor_id0] + links_origin_vertical[anchor_id0]

        # pos1_x = random.uniform(*posHorizonal_factor_range) * link_lengths[anchor_id1]
        # pos1_y = random.uniform(*posVertical_factor_range) * link_lengths[anchor_id1] + links_origin_vertical[anchor_id1]

        
        print(f"DEBUG: body0: {link_name_from_id[anchor_id0]}, pos0: ({pos0_x}, {pos0_y})")
        print(f"DEBUG: body1: {link_name_from_id[anchor_id1]}, pos1: ({pos1_x}, {pos1_y})")


        init_length = math.sqrt((pos0_x - pos1_x)**2 + (pos0_y - pos1_y)**2)
        rest_length_list = [l_10, l_20, l_30]
        stiffness_list = [k_3, k_4, mk_3]
        damping_list = [c_3, c_4, mc_3]
        # rest_length = init_length * random.uniform(*muscle_length_factor_range)
        rest_length = rest_length_list[muscle_id]
        stiffness = stiffness_list[muscle_id]
        damping = damping_list[muscle_id]
        mass_list = [m_3, m_4, mm_3]
        mass = mass_list[muscle_id]

        # muscle_cfg = {
        #     'name': f"muscle{muscle_id}",
        #     'mass': 0.18648*random.uniform(*factor_range),
        #     'stiffness': 318.76*random.uniform(*factor_range),
        #     'damping': 11.34*random.uniform(*factor_range),
        #     'init_length': init_length,
        #     'rest_length': rest_length,
        #     'base_pos': [0, -pos0_x, -pos0_y],      # pos relative to the parent link origin
        #     'base_axisangle': [1, 0, 0, math.atan2(pos1_y-pos0_y, pos1_x-pos0_x)-math.pi/2],      # axisangle relative to the parent link origin
        # }
        muscle_cfg = {
            'name': f"muscle{muscle_id}",
            'mass': mass,
            'stiffness': stiffness,
            'damping': damping,
            'init_length': init_length,
            'rest_length': rest_length,
            'base_pos': [0, -pos0_x, -pos0_y-(-links_origin_vertical[anchor_id0])],      # pos relative to the parent link origin
            'base_axisangle': [1, 0, 0, math.atan2(pos1_y-pos0_y, pos1_x-pos0_x)-math.pi/2],      # axisangle relative to the parent link origin
        }


        spec = add_muscle_fromto(spec, link_name_from_id[anchor_id0], link_name_from_id[anchor_id1], muscle_cfg)

    spec = set_anchor_to_zero(spec)

    # Step 3: check the topology
    model = spec.compile()

    # save xml
    xml_script = spec.to_xml()
    with open(CURRENT_DIR / f'test.xml', 'w') as f:
        f.write(xml_script)


# load the test.xml and control
# topology randomization
class mjExperiment(object):
    def __init__(self, model_path, model_type="ideal_geom_swing"):
        self.model_path = model_path
        self.model = None   # wait for instantiation
        self.data = None
        self.height = 320
        self.width = 400
        self.framerate = 60  # (Hz)

    @staticmethod
    def calculate_length(theta1, theta2):
        a1, a2 = 0.25, 0.25
        b1, b2 = 0.21213, 0.1
        d1, d2 = 0.06, 0.10
        beta1, beta2 = np.radians(8.13), np.radians(30)

        C_x, C_y = b1 * np.cos(theta1 - beta1), b1 * np.sin(theta1 - beta1)
        D_x = a1 * np.cos(theta1) + b2 * np.cos(theta1 + theta2 + beta2)
        D_y = a1 * np.sin(theta1) + b2 * np.sin(theta1 + theta2 + beta2)

        l1 = np.hypot(d1 - C_x, 0-C_y)
        l2 = np.hypot(-d2 - D_x, 0-D_y)

        return l1, l2

    def calculate_bias(self, l10, l20):
        # ideal geom, with the initial qpos = [np.pi/2, 0]
        l1, l2 = mjExperiment.calculate_length(np.pi/2, 0)
        l1_rel = l10 - l1
        l2_rel = l20 - l2
        return l1_rel / 2, l2_rel / 2

    def apply_params(self, params):
        # do not need to apply params, for the XML is changed directly
        _model = mj.MjModel.from_xml_path(self.model_path)
        _data = mj.MjData(_model)
        self.model = _model
        self.data = _data
    
    @staticmethod
    def F1_func(t): return 0 if t < 10 else ((t//10)%3 + 1) * 2
    
    @staticmethod
    def F2_func(t): return 0 if t < 10 else (((t//10)+1)%3 + 1) * 2
    
    @staticmethod
    def F3_func(t): return 0 if t < 10 else (((t//10)+2)%3 + 1) * 2
    
    def run(self, params, time_step, duration, ifrender=True):
        self.apply_params(params)
        m, d = self.model, self.data
        # Reset state and time
        mj.mj_resetData(m, d)

        results = []
        frames = []
        valid, valid_last = True, True        # check if the angle exceeds the limit

        with mj.Renderer(m, self.height, self.width) as renderer:
            while d.time < duration:
                d.ctrl[:2] = mjExperiment.F1_func(d.time)
                d.ctrl[2:4] = mjExperiment.F2_func(d.time)
                d.ctrl[4:] = mjExperiment.F3_func(d.time)
                
                theta1 = d.qpos[0] + np.pi / 2
                theta2 = d.qpos[1] + 0
                theta3 = d.qpos[2] + 0
                    
                results.append((d.time, theta1, theta2, theta3))

                mj.mj_step(m, d)
                if len(frames) < d.time * self.framerate:   # assume the simulation is running much faster than the rendering
                    if ifrender:
                        renderer.update_scene(d, camera="closeup")
                        frames.append(renderer.render())

        time_sim, theta1_sim, theta2_sim, theta3_sim = np.array(results).T
        return time_sim, theta1_sim, theta2_sim, theta3_sim, frames, valid, valid_last

if __name__ == "__main__":

    path = CURRENT_DIR / "test.xml"
    experiment_instance = mjExperiment(str(path), model_type="ideal_geom_swing")

    # Model Parameter Set
    stiffness_MAA, stiffness_BAA = 318.76, 315.8
    l10, l20 = 0.174, 0.2562
    damping_MAA, damping_BAA = 11.34, 10.9
    s1, s2 = 0.000654, 0.000637
    s3 = 1
    c1_thigh, c2_calf = 0.03*0, 0.03*0
    P1 = 0/s1#50*1e3
    P2 = 0
    P3 = 0
    P1_prime = 10/s1#*1e3
    P2_prime = 5/s2#*1e3
    P3_prime = 1
    
    exp_num = 100
    np.random.seed(0)
    tic = time.time()

    params = {
        'stiffness_MAA': stiffness_MAA,
        'stiffness_BAA': stiffness_BAA,
        'l10': l10,
        'l20': l20,
        'damping_MAA': damping_MAA,
        'damping_BAA': damping_BAA,
        's1': s1,
        's2': s2,
        's3': s3,
        'c1_thigh': c1_thigh,
        'c2_calf': c2_calf,
        'P1': P1,
        'P2': P2,
        'P3': P3,
        'P1_prime': P1_prime,
        'P2_prime': P2_prime,
        'P3_prime': P3_prime
    }

    # Run the Experiments
    time_step = 10
    duration_exp = 40  # seconds
    time_sim, theta1_sim, theta2_sim, theta3_sim, frames, _, _ = experiment_instance.run(params, time_step=time_step, duration=duration_exp, ifrender=True)

    # show video
    # media.show_video(frames, fps=framerate)
    # media.write_video("./log/temp_experiment_valve_dynamics_temp0724_topology.mp4", frames, fps=experiment_instance.framerate)

    # Plot Results with dual y-axis for pressure states
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(time_sim, theta1_sim, label='Theta1', color='tab:blue')
    ax1.plot(time_sim, theta2_sim, label='Theta2', color='tab:orange')
    ax1.plot(time_sim, theta3_sim, label='Theta3', color='tab:red')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (radians)')
    ax1.set_title('Joint Angles and Valve Pressures Over Time')
    ax1.legend(loc='upper left')

    plt.show()

    # save for cross-validation of the sympy implementation
    data_traj = {
        'time': time_sim,
        'theta1': theta1_sim,
        'theta2': theta2_sim,
        'theta3': theta3_sim,
    }
    df = pd.DataFrame(data_traj)
    df.to_csv(EXP_DIR /'test_data'/'data_mj.csv', index=False)
