import numpy as np

# 定义状态、动作和奖励
states = [1, 2, 3]
actions = [1, 2, 3]  # 分别代表停留、向下个状态、向上个状态
rewards = {1: 1, 2: 10, 3: -10}

# 定义策略概率
policy = {1: 1/3, 2: 1/3, 3: 1/3}

# 定义折扣系数
gamma = 0.9

# 定义状态转移概率
transition_probs = {
    1:
    {1: {1: 0.5, 2: 0.2, 3: 0.3},
    2: {1: 0.3, 2: 0.5, 3: 0.2},
    3: {1: 0.1, 2: 0.3, 3: 0.6}},
    2:
    {1: {1: 0.5, 2: 0.1, 3: 0.4},
    2: {1: 0.2, 2: 0.6, 3: 0.2},
    3: {1: 0.2, 2: 0.2, 3: 0.6}},
    3:
    {1: {1: 0.5, 2: 0.4, 3: 0.1},
    2: {1: 0.1, 2: 0.5, 3: 0.4},
    3: {1: 0.3, 2: 0.1, 3: 0.6}}
}

# 初始化Q表格
Q = {(s, a): 0 for s in states for a in actions}

# 迭代计算Q函数，直到收敛
tolerance = 1e-6
cnt = 0
while True:
    cnt += 1
    Q_new = {(s, a): rewards[s]+ gamma * sum((transition_probs[s][a][s1] * sum((policy[a1]*Q[s1, a1])
                                                                               for a1 in actions)) for s1 in states) for (s, a) in Q}
    max_diff = max(abs(Q_new[s, a] - Q[s, a]) for (s, a) in Q)
    if max_diff < tolerance:
        break
    Q = Q_new
    print(f'第{cnt}轮迭代结果',Q)
