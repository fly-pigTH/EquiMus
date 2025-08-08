import datetime
# Down stream application: PID tuning
# utilizing MuJoCo Simulation and minimize/differential_evolution optimization
# result analysis version, with python script

import numpy as np
import mediapy as media
from tqdm import tqdm
import sys, datetime, math, mujoco, os
from scipy.stats import qmc
from scipy.optimize import minimize, differential_evolution
import pandas as pd

# Basic Path
import rootpath
from pathlib import Path

ROOT_DIR = rootpath.detect()   # Get the root directory of the project (.git)
CURRENT_DIR = Path(__file__).resolve().parent
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"CURRENT_DIR: {CURRENT_DIR}")

sys.path.append(str(Path(ROOT_DIR)))
from utils.experiment import MujocoExperiment


dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
folder_path = CURRENT_DIR / "video" / dt_str
os.makedirs(folder_path, exist_ok=False)  # 如果文件名称冲突，报错!