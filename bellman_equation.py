import numpy as np

# 定义状态转移概率矩阵
transition_probabilities = np.array([
    [0, 1, 0.0, 0, 0.0],
    [0.3, 0.0, 0.3, 0.4, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])

# 定义奖励向量
rewards = np.array([0, 0, 10, 1000, -1000])

# 定义折扣因子
discount_factor = 0.95

# 初始化状态值函数
values = np.zeros(len(rewards))

# 迭代计算状态值函数
epsilon = 1e-6  # 定义收敛阈值
cnt=0
while True:
    prev_values = np.copy(values)
    for state in range(len(rewards)):
        values[state] = rewards[state] + discount_factor * np.sum(transition_probabilities[state] * values)
    if np.max(np.abs(prev_values - values)) < epsilon:
        break
    cnt+=1
    print(f'第{cnt}轮迭代结果:',values)

print("迭代法结果:")
print(values)

# 解析法求解状态值
values = np.linalg.inv(np.eye(5) - discount_factor * transition_probabilities).dot(rewards)

# 打印最终的状态值
print("解析法结果:")
print(values)