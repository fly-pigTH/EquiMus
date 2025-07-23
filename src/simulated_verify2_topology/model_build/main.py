import mujoco as mj

spec = mj.MjSpec()




# Connect tendon to sites
thread = spec.add_tendon(name=f'{name}_thread_{x_dir}_{y_dir}',
                        limited=True,
                        range=[0, TENDON_LENGTH], width=0.01 )
thread.wrap_site(f'{name}_hook_{x_dir}_{y_dir}')
thread.wrap_site(f'{name}_anchor_{x_dir}_{y_dir}')