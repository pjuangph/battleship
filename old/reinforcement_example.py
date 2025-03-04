import numpy as np
import tensorflow as tf
import random
from collections import deque

# Define environment parameters
GRID_SIZE = 5
NUM_ACTIONS = 4  # Up, Down, Left, Right
EPISODES = 1000
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995  # Decay per episode
MIN_EPSILON = 0.01  # Min exploration rate
BATCH_SIZE = 32
MEMORY_SIZE = 2000  # Replay memory

# Goal position
GOAL = (4, 4)
REWARD_GOAL = 10
REWARD_EMPTY = 0
REWARD_WALL = -1

# Action mappings: [Up, Down, Left, Right]
ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

# Replay memory
memory = deque(maxlen=MEMORY_SIZE)

def get_next_position(state, action):
    """Returns the next position after taking an action."""
    x, y = state
    dx, dy = ACTIONS[action]
    new_x, new_y = x + dx, y + dy

    # Stay in bounds
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
        return (new_x, new_y), REWARD_EMPTY
    return (x, y), REWARD_WALL  # Hitting a wall

# Neural network for Q-learning
class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation="relu")
        self.dense2 = tf.keras.layers.Dense(24, activation="relu")
        self.output_layer = tf.keras.layers.Dense(NUM_ACTIONS, activation="linear")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

# Initialize Q-network
q_network = DQN()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

def get_action(state):
    """Epsilon-greedy policy for action selection."""
    if np.random.rand() < EPSILON:
        return random.choice(list(ACTIONS.keys()))  # Explore
    q_values = q_network(np.array([state], dtype=np.float32))
    return np.argmax(q_values.numpy())  # Exploit best action

# Training function
def train():
    global EPSILON
    if len(memory) < BATCH_SIZE:
        return  # Not enough samples to train

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, done_flags = zip(*batch)

    states = np.array(states, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    
    with tf.GradientTape() as tape:
        q_values = q_network(states)
        next_q_values = q_network(next_states)
        targets = q_values.numpy()
        
        for i in range(BATCH_SIZE):
            if done_flags[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + DISCOUNT_FACTOR * np.max(next_q_values[i])

        loss = tf.keras.losses.MeanSquaredError()(targets, q_values)

    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # Decay epsilon
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

# Training loop
for episode in range(EPISODES):
    state = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
    done = False
    step_count = 0

    while not done:
        action = get_action(state)
        next_state, reward = get_next_position(state, action)

        if next_state == GOAL:
            reward = REWARD_GOAL
            done = True

        memory.append((state, action, reward, next_state, done))
        train()
        state = next_state
        step_count += 1

    if episode % 100 == 0:
        print(f"Episode {episode}: Steps taken = {step_count}, Epsilon = {EPSILON:.3f}")

print("Training complete!")

# Test the trained agent
state = (0, 0)  # Start from the top-left
done = False

print("\nAgent Test Run:")
while not done:
    action = get_action(state)
    state, _ = get_next_position(state, action)
    print(state)
    if state == GOAL:
        done = True
