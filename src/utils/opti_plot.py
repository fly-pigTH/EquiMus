import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# 假设这是你的误差函数
def error(x):
    # 这里用随机数据模拟误差计算
    return np.sum(np.square(x - np.array([100, 100, 0.2, 0.3, 50, 50, 0.0005, 0.0006, 5, 5])))

# 绘图函数
def plot_results(xk, convergence, epi):
    # 清空当前图形
    plt.clf()
    
    # 绘制当前参数值
    plt.bar(range(len(xk)), xk, color='blue', alpha=0.6, label='Current Parameters')
    
    # 绘制目标参数值（假设目标值为 [100, 100, 0.2, 0.3, 50, 50, 0.0005, 0.0006, 5, 5]）
    target = [100, 100, 0.2, 0.3, 50, 50, 0.0005, 0.0006, 5, 5]
    plt.bar(range(len(target)), target, color='red', alpha=0.3, label='Target Parameters')
    
    # 设置图形属性
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.title(f'Epoch: {epi}, Convergence: {convergence:.4f}, MSE: {error(xk):.4f}')
    plt.legend()
    plt.grid(True)
    
    # 显示图形
    plt.pause(0.01)

# 优化回调函数
def callback(xk, convergence):
    global time_epi_start, epi
    epi += 1
    time_cost_epi = time.time() - time_epi_start
    time_epi_start = time.time()
    
    # 打印当前优化信息
    print(f"Epoch: {epi}, Time/epoch: {time_cost_epi:.2f}s, convergence={convergence:.4f}, MSE={error(xk):.4f}")
    print(f"k1={xk[0]:.4f}, k2={xk[1]:.4f}, l10={xk[2]:.4f}, l20={xk[3]:.4f}, c1={xk[4]:.4f}, c2={xk[5]:.4f}, s1={xk[6]:.8f}, s2={xk[7]:.8f}, c1_thigh={xk[8]:.4f}, c2_calf={xk[9]:.4f}")
    
    # 绘制当前优化结果
    plot_results(xk, convergence, epi)

if __name__== "__main__":
    exp_mode = "train"  # train or test

    if exp_mode == "train":
        global time_start, time_epi_start, epi
        time_start = time.time()
        time_epi_start = time.time()
        epi = 0

        # 使用进化算法进行参数优化
        bounds = [(0, 1000), (0, 1000), (0, 0.5), (0, 0.5), (0, 100), (0, 100), (0.000001, 0.01), (0.000001, 0.01), (0, 10), (0, 10)]  # 参数范围
        result = differential_evolution(error, bounds, strategy='best1bin', maxiter=10000, disp=False, callback=callback, popsize=10)
        
        # 打印优化结果
        k1_opt = result.x[0]
        k2_opt = result.x[1]
        l10_opt = result.x[2]
        l20_opt = result.x[3]
        c1_opt = result.x[4]
        c2_opt = result.x[5]
        s1_opt = result.x[6]
        s2_opt = result.x[7]
        print(f"Optimal k1={k1_opt:.4f}, k2={k2_opt:.4f}, l10={l10_opt:.4f}, l20={l20_opt:.4f}, c1={c1_opt:.4f}, c2={c2_opt:.4f}, s1={s1_opt:.8f}, s2={s2_opt:.8f}, c1_thigh={result.x[8]:.4f}, c2_calf={result.x[9]:.4f}, MSE={result.fun:.4f}")
        print(f"Error: {error([k1_opt, k2_opt, l10_opt, l20_opt, c1_opt, c2_opt, s1_opt, s2_opt])}")
        time_cost = time.time() - time_start
        print(f"Time Cost: {time_cost:.2f}s")
    
    elif exp_mode == "test":
        # 使用最优参数进行测试
        best_c1, best_c2 = 291.5757670770301, 98.3306672255234
        MSE_mat_overview = error([best_c1, best_c2], mode=exp_mode)
        print(MSE_mat_overview)