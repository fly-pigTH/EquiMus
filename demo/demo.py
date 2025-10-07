# Simulation rendering refer to https://mujoco.readthedocs.io/en/stable/python.html
import time, argparse, os
import mujoco
import mujoco.viewer

DURATION = 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dof", type=int, choices=[2, 3], default=3, help="choose 2 or 3 DOF model")
    parser.add_argument("--duration", type=float, default=1000, help="viewer duration in seconds")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(f'./src/validation_simulation/morphology/model_build/demo_{args.dof}DOF.xml')
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after DURATION wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < args.duration:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()

