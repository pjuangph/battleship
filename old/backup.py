import tensorflow as tf
import numpy as np

# 1.2 Define the nn variable network.
BOARD_SIZE = 10
SHIP_SIZE = 3

hidden_units = BOARD_SIZE
output_units = BOARD_SIZE

# Define model
class BattleshipNN(tf.keras.Model):
    def __init__(self):
        """Inputs = board size, output = board size. 
        """
        super(BattleshipNN, self).__init__()
        self.W1 = tf.Variable(tf.random.truncated_normal([BOARD_SIZE, hidden_units], stddev=0.1))   # Weights
        self.b1 = tf.Variable(tf.zeros([1, hidden_units]))  # Biases            
        self.W2 = tf.Variable(tf.random.truncated_normal([hidden_units, output_units], stddev=0.1)) # Weights
        self.b2 = tf.Variable(tf.zeros([1, output_units]))  # Biases

    def call(self, input_positions):
        h1 = tf.tanh(tf.matmul(input_positions, self.W1) + self.b1)
        logits = tf.matmul(h1, self.W2) + self.b2
        probabilities = tf.nn.softmax(logits)
        return logits, probabilities

# Instantiate model
model = BattleshipNN()

# Define loss and optimizer
loss_fn = lambda logits, labels: tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Training step function
def train_step(input_positions, labels, learning_rate):
    with tf.GradientTape() as tape:
        logits, _ = model(input_positions)
        loss = loss_fn(logits, labels)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# 1.4 Game play definition.
def play_game(training=False):
    """ Play game of battleship using network."""
    ship_left = np.random.randint(BOARD_SIZE - SHIP_SIZE + 1)
    ship_positions = set(range(ship_left, ship_left + SHIP_SIZE))

    board_position_log = []
    action_log = []
    hit_log = []
    current_board = np.array([[-1] * BOARD_SIZE], dtype=np.float32)

    while (sum(hit_log) < SHIP_SIZE) and (len(action_log) < BOARD_SIZE):
        board_position_log.append(current_board.copy())
        logits, probs = model(current_board)

        probs = probs.numpy()[0]  # Convert to NumPy for probability filtering
        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        probs = [p / sum(probs) for p in probs]

        probs = np.clip(probs, 0, 1)  # Ensure values are within [0,1]
        probs /= np.sum(probs)  # Re-normalize to exactly sum to 1
        bomb_index = np.random.choice(BOARD_SIZE, p=probs) if training else np.argmax(probs)
        hit_log.append(1 * (bomb_index in ship_positions))
        current_board[0, bomb_index] = 1 * (bomb_index in ship_positions)
        action_log.append(bomb_index)

    return board_position_log, action_log, hit_log

# Example:
play_game(training=False)

# 1.5 Reward function definition
def rewards_calculator(hit_log, gamma=0.5):
    """ Discounted sum of future hits over trajectory"""            
    hit_log_weighted = [(item - float(SHIP_SIZE - sum(hit_log[:index])) / (BOARD_SIZE - index)) * 
                        (gamma ** index) for index, item in enumerate(hit_log)]
    return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]

# Example
reward =  rewards_calculator([0,0,1,1,1])

# 1.6 Training loop: Play and learn
game_lengths = []
TRAINING = True   # Boolean specifies training mode
ALPHA = 0.06      # step size

for game in range(10000):
    board_position_log, action_log, hit_log = play_game(training=TRAINING)
    game_lengths.append(len(action_log))
    rewards_log = rewards_calculator(hit_log)
    
    for reward, current_board, action in zip(rewards_log, board_position_log, action_log):
        if TRAINING:
            labels = tf.constant([action], dtype=tf.int64)
            train_step(tf.convert_to_tensor(current_board, dtype=tf.float32), labels, ALPHA * reward)
