import numpy as np

q_table = np.zeros((5, 2))

def train_q_learning():
    global q_table

    alpha = 0.1
    gamma = 0.9
    episodes = 200

    for _ in range(episodes):
        state = np.random.randint(0, 5)

        for step in range(10):
            action = np.random.randint(0, 2)

            if action == 0:
                next_state = max(0, state - 1)
            else:
                next_state = min(4, state + 1)

            reward = 1 if next_state == 4 else 0

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state

def get_q_table():
    return q_table.tolist()