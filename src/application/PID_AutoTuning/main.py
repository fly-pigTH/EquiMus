from scripts.experiment import generate_PID_trajectories
from scripts.postprocess import analysis_data

if __name__ == "__main__":
    generate_PID_trajectories()
    analysis_data()