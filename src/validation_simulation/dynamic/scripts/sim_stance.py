# Simulated Verificaition, dynamics with STANCE PHASE
# This scripts generate the step response trajectory

import mediapy as media
import pandas as pd
import mediapy as media

import rootpath
import sys
from pathlib import Path

ROOT_DIR = rootpath.detect()
sys.path.append(str(Path(ROOT_DIR)))
from utils.experiment import MujocoExperiment

CURRENT_DIR = Path(__file__).resolve().parent
EXP_DIR = CURRENT_DIR.parent
VIDEO_DIR = EXP_DIR / "video"
DATA_DIR = EXP_DIR / "data"

path = Path(ROOT_DIR) / "models" / "SingleLeg_ideal_landing2DOF.xml"
experiment_instance = MujocoExperiment(str(path), model_type = "ideal_geom")
fixed_para = {
    'stiffness_MAA': 318.76, # 637.52 / 2,
    'stiffness_BAA': 315.8, # 631.6 / 2,
    'l10': 0.174,
    'l20': 0.2562,
    'damping_MAA': 11.34,
    'damping_BAA': 10.90,
    'c1_thigh': 0,
    'c2_calf': 0,
    's1': 1.0,
    's2': 1.0,
    'P1': 0,        # To be set
    'P2': 0,
    'P1_prime': 11.0,      
    'P2_prime': 20.0       # equal to F when s=1
}
exp_data_serializable = []
time_sim_, theta1_sim_, theta2_sim_, frames_, valid_, valid_last_ = experiment_instance.run(fixed_para, 0, 20, True)

# show
media.show_video(frames_, fps=60)
media.write_video(VIDEO_DIR / "stance.mp4", frames_, fps=60)

# save the data as csv
df = pd.DataFrame({
    'time': time_sim_,
    'theta1': theta1_sim_,
    'theta2': theta2_sim_,
    'valid': valid_,
    'valid_last': valid_last_
})
df.to_csv(DATA_DIR / "stance_exp_data.csv", index=False)