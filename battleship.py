'''
    Board values: -1 for no guesses, 0 bomb, 1 for hit
'''

from typing import List
import torch.nn as nn
import torch.optim as optim
import torch
from torch import Tensor
import numpy as np
from ship_placements import place_ships
from transformer import Transformer 

class BattleshipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, current_board: Tensor, repeated_guesses: Tensor, action_index: Tensor) -> Tensor:
        """Compute the loss of the battleship game."""
        
        # Ensure action_index is a tensor
        if not isinstance(action_index, Tensor):
            action_index = torch.tensor(action_index, dtype=torch.float32, device=current_board.device, requires_grad=True)

        # Ensure repeated_guesses is a tensor
        if not isinstance(repeated_guesses, Tensor):
            repeated_guesses = torch.tensor(repeated_guesses, dtype=torch.float32, device=current_board.device, requires_grad=True)

        # Convert board to float tensor for differentiability
        current_board = current_board.float()

        # Compute correct and wrong guesses using differentiable operations
        correct_guesses = torch.sum((current_board == 2).float())
        wrong_guesses = torch.sum((current_board == 1).float())

        # Compute loss with floating point division
        loss = (wrong_guesses + repeated_guesses) / (action_index + correct_guesses + 1e-6)  # Avoid division by zero

        return loss

# Define loss and optimizer
def play_game(training:bool=False,board_height:int=10,board_width:int=10):
    """ Play game of battleship using network."""
    if training:
        model.train()
    else:
        model.eval()
    board_position_log = []
    ship_positions = place_ships(board_height,board_width,SHIP_SIZES)
    ship_position_indices = np.where(ship_positions == 1)[1]

    board_size = board_height * board_width
    action_log = torch.zeros((board_size), dtype=torch.int32)
    hit_log = torch.zeros((board_size), dtype=torch.int32)

    current_board = torch.from_numpy(0*np.ones(shape=(1,board_size), dtype=np.int64)).type(torch.int32)  # 0 no bomb, 1 bomb, 2 hit

    loss_fn = BattleshipLoss()
    action_index = 0
    
    while (torch.sum(hit_log) < sum(SHIP_SIZES)) and (action_index < board_size):
        optimizer.zero_grad()
        output = model.encode(current_board)

        # Get the max value and its index
        bomb_index = torch.argmax(output)  # Flattened index
        bomb_index = bomb_index % board_width*board_height
        action_log[bomb_index] += 1 # Increment the number of times the index has been guessed

        current_board[0,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)  # 0 no bomb, 1 bomb, 2 hit

        # Check if the index has already been guessed
        repeated_guesses = torch.sum(action_log == action_log[bomb_index])-1
        repeated_guesses = torch.tensor(repeated_guesses, dtype=torch.int32, device=current_board.device, requires_grad=False)

        action_index += 1
        loss = loss_fn(current_board,repeated_guesses,action_index)
        loss.backward()
        optimizer.step()

        board_position_log.append(current_board.numpy().copy())
        
        if current_board[0,bomb_index] == 2:
            hit_log[action_index] = 1* (bomb_index in ship_position_indices)

        
        hit_to_guess_ratio = torch.sum(hit_log == 1)/action_index
        wrong_guess_ratio = torch.sum(current_board == 1)/action_index
        correct_guess_ratio = torch.sum(current_board == 2)/action_index
        print(f"Action_index: {action_index} Hit to guess ratio: {hit_to_guess_ratio}, Wrong guess ratio: {wrong_guess_ratio}, Correct guess ratio: {correct_guess_ratio}")
    return hit_to_guess_ratio, wrong_guess_ratio, correct_guess_ratio

if __name__ =="__main__":
    n_games = 100
    board_height = 10
    board_width = 10
    SHIP_SIZES = [2,3,3,4,5]

    # Instantiate model
    model = Transformer(src_vocab_size=board_height,
                        tgt_vocab_size=1, 
                        d_model=board_width*board_height, num_heads=5, num_layers=6, 
                        d_ff=2048, 
                        max_seq_length=board_height*board_width, dropout=0.1)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    hit_to_guess_tracker = []; wrong_guess_tracker = []; correct_guess_tracker = []
    # Train the model
    for i in range(n_games):
        hit_to_guess_ratio, wrong_guess_ratio, correct_ratio = play_game(training=True,board_height=board_height,board_width=board_width)
        print(f"Game {i}: Hit to guess ratio: {hit_to_guess_ratio}, Wrong guess ratio: {wrong_guess_ratio}, Correct guess ratio: {correct_ratio}")
        hit_to_guess_tracker.append(hit_to_guess_ratio)
        wrong_guess_tracker.append(wrong_guess_ratio)
        correct_guess_tracker.append(correct_ratio)
    
