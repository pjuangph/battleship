'''
    Board values: -1 for no guesses, 0 bomb, 1 for hit
'''

import torch.nn as nn
import torch.optim as optim
import numpy as np
from ship_placements import place_ships
from transformer import Transformer 

# 1.2 Define the nn variable network.
board_height = 10
board_width = 10
SHIP_SIZES = [2,3,3,4,5]


# Instantiate model
model = Transformer(src_vocab_size=board_height*board_width,
                    tgt_vocab_size=board_height*board_width, 
                    d_model=512, num_heads=4, num_layers=6, 
                    d_ff=2048, 
                    max_seq_length=board_height*board_width, dropout=0.1)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define loss and optimizer
def play_game(training=False,board_height=10,board_width=10):
    """ Play game of battleship using network."""
    if training:
        model.train()
    else:
        model.eval()
    ship_positions = place_ships(board_height,board_width,SHIP_SIZES)

    board_position_log = []
    action_log = []
    hit_log = []
    board_size = board_height * board_width
    current_board = -1*np.ones(shape=(1,board_size), dtype=np.float32)
    
        optimizer.zero_grad()
        output = model(src_data, tgt_data[:, :-1])
        loss = loss_fn(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

model.eval()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Training step function
def train_step(input_positions, labels, learning_rate):
    with tf.GradientTape() as tape:
        logits, _ = model(input_positions)
        loss = loss_fn(logits, labels)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

board_size = board_height * board_width

def play_game(training=False,board_height=10,board_width=10):
    """ Play game of battleship using network."""
    ship_positions = place_ships(board_height,board_width,SHIP_SIZES)
    board_position_log = []
    action_log = []
    hit_log = []
    board_size = board_height * board_width
    current_board = -1*np.ones(shape=(board_size,), dtype=np.float32)

    while (sum(hit_log) < sum(SHIP_SIZES)) and (len(action_log) < board_size):
        board_position_log.append(current_board.copy())
        logits, probs = model(current_board)

        probs = probs.numpy()[0]  # Convert to NumPy for probability filtering
        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        probs = [p / sum(probs) for p in probs]

        probs = np.clip(probs, 0, 1)  # Ensure values are within [0,1]
        probs /= np.sum(probs)  # Re-normalize to exactly sum to 1
        bomb_index = np.random.choice(board_size, p=probs) if training else np.argmax(probs)
        hit_log.append(1 * (bomb_index in ship_positions))
        current_board[0, bomb_index] = 1 * (bomb_index in ship_positions)
        action_log.append(bomb_index)

    return board_position_log, action_log, hit_log

# Example:
play_game(training=False)

# 1.5 Reward function definition
def rewards_calculator(hit_log, board_height:int=10,board_width:int=10,gamma:float=0.5):
    """ Discounted sum of future hits over trajectory"""
    total_ship_size = sum(SHIP_SIZES)
    board_size = board_height * board_width
    hit_log_weighted = [(item - float(total_ship_size - sum(hit_log[:index])) / (board_size - index)) * 
                        (gamma ** index) for index, item in enumerate(hit_log)]
    return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]

# Example
rewards_calculator([0,0,1,1,1])

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
