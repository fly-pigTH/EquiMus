import time
for i in range(10000):
    print(i)
    time.sleep(0.4)
    print("\033[H\033[J")  # 使用ANSI转义序列清除终端输出