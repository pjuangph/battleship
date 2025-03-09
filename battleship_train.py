'''
    Board values: -1 for no guesses, 0 bomb, 1 for hit
'''

from typing import List, Tuple
import torch.nn as nn
import torch.optim as optim
import torch
from torch import Tensor
import numpy as np
from ship_placements import place_ships
from transformer import Transformer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BattleshipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, current_board: Tensor, repeated_guesses: Tensor, action_index: Tensor,ship_sizes:Tensor) -> Tensor:
        """Compute the loss of the battleship game."""
        
        # Ensure action_index is a tensor
        if not isinstance(action_index, Tensor):
            action_index = torch.tensor(action_index, dtype=torch.float32, device=device, requires_grad=True)

        # Ensure repeated_guesses is a tensor
        if not isinstance(repeated_guesses, Tensor):
            repeated_guesses = torch.tensor(repeated_guesses, dtype=torch.float32, device=device, requires_grad=True)

        # Convert board to float tensor for differentiability
        current_board = current_board.float()

        # Compute correct and wrong guesses using differentiable operations
        correct_guesses = torch.sum((current_board == 2).float(),dim=1)
        cumsum_correct_guesses = torch.cumsum((current_board==2),dim=0)
        action_index_each_game = torch.argmax((cumsum_correct_guesses == torch.sum(ship_sizes)).float(), dim=-1)+action_index
        wrong_guesses = torch.sum((current_board == 1).float(),dim=1)
        
        # Compute loss with floating point division
        loss = torch.mean((wrong_guesses + repeated_guesses) / (action_index_each_game + correct_guesses + 1e-6))  # Avoid division by zero

        return loss,action_index_each_game

# Define loss and optimizer
def play_game(model:nn.Module,optimizer, training:bool=False,board_height:int=10,board_width:int=10,n_games:int=64,ship_sizes:List[int]=[2,3,3,4,5]) -> Tuple[Tensor,Tensor,Tensor]:
    """ Play game of battleship using network."""
    if training:
        model.train()
    else:
        model.eval()
    ship_position_indices = np.zeros(shape=(n_games,sum(ship_sizes)), dtype=np.int64)
    for i in range(n_games):    
        ship_positions = place_ships(board_height,board_width,ship_sizes)
        ship_position_indices[i,:] = np.where(ship_positions == 1)[1]
    board_size = board_height * board_width

    board_position_log = []
    action_log = torch.zeros((n_games,board_size), dtype=torch.int32)
    hit_log = torch.zeros((n_games,board_size), dtype=torch.int32)

    current_board = torch.from_numpy(0*np.ones(shape=(n_games,board_size), dtype=np.int64)).type(torch.int32)  # 0 no bomb, 1 bomb, 2 hit
    ship_sizes = torch.tensor(ship_sizes, dtype=torch.float32, device=device, requires_grad=True)
    loss_fn = BattleshipLoss()
    action_index = 0
    
    while (torch.min(torch.sum(hit_log,dim=1)) < sum(ship_sizes)) and (action_index < board_size):
        optimizer.zero_grad()
        output = model.encode(current_board)
        bomb_index = torch.argmax(output,dim=1)  # Get the bomb index for all the games
        
        action_index += 1
        
        for i in range(n_games):
            action_log[i,bomb_index[i]] += 1 # Increment the number of times the index has been guessed
            # Set the board values for each game
            current_board[i,bomb_index[i]] = 2 * (bomb_index[i] in ship_position_indices[i,:]) + 1 * (bomb_index[i] not in ship_position_indices[i,:])  # 0 no bomb, 1 bomb, 2 hit

            # Check if the index has already been guessed
            repeated_guesses = torch.sum(action_log[i,:] == action_log[i,bomb_index[i]])-1
            repeated_guesses = torch.tensor(repeated_guesses, dtype=torch.float32, device=device, requires_grad=True)

            hit_log[i,action_index-1] = (current_board[i,bomb_index[i]] == 2) * 1* (bomb_index[i] in ship_position_indices[i,:])

        loss,action_index_for_each_game = loss_fn(current_board,repeated_guesses,action_index,ship_sizes)
        loss.backward()
        optimizer.step()

        board_position_log.append(current_board.numpy().copy())

    return hit_log, action_log, current_board, action_index_for_each_game

if __name__ =="__main__":
    n_games_per_epoch = 1
    epochs = 10000
    board_height = 10
    board_width = 10
    SHIP_SIZES = [2,3,3,4,5]

    src_vocab_size = board_height*board_width
    tgt_vocab_size = 1
    d_model = board_width*board_height
    num_heads = 5
    num_layers = 2
    d_ff = 2048
    max_seq_length = board_height*board_width
    dropout = 0.05
    # Instantiate model
    model = Transformer(src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size, 
                        d_model=d_model, num_heads=num_heads, num_layers=num_layers, 
                        d_ff=d_ff, 
                        max_seq_length=max_seq_length, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    hit_to_guess_tracker = []; wrong_guess_tracker = []; correct_guess_tracker = []
    # Train the model
    for epoch in range(epochs):
        hit_log, action_log,current_board, action_index_for_each_game = play_game(model,optimizer,training=True,board_height=board_height,board_width=board_width,n_games=n_games_per_epoch,ship_sizes=SHIP_SIZES)
        
        repeated_guess_ratio = torch.mean(torch.sum(action_log > 1,dim=1)/action_index_for_each_game)
        hit_to_guess_ratio = torch.mean(torch.sum(hit_log == 1,dim=1)/action_index_for_each_game)
        wrong_guess_ratio = torch.mean(torch.sum(current_board == 1,dim=1)/action_index_for_each_game)
        correct_guess_ratio = torch.mean(torch.sum(current_board == 2,dim=1)/action_index_for_each_game)
        print(f"Epoch: {epoch:d} Hit to guess ratio: {hit_to_guess_ratio:0.3e}, Wrong guess ratio: {wrong_guess_ratio:0.3e}, Correct guess ratio: {correct_guess_ratio:0.3e}")

        # Print Games Statistics, total guesses to find all ships, hit to guess ratio, wrong guess ratio, correct guess ratio
        print(f"Epoch: {epoch:d} Average guesses to find all ships: {torch.mean(action_index_for_each_game):0.2f}")
        print(f"Epoch: {epoch:d} Min guess to find all ships: {torch.min(action_index_for_each_game):0.2f}")
        print(f"Epoch: {epoch:d} Max guess to find all ships: {torch.max(action_index_for_each_game):0.2f}")
        print(f"Epoch: {epoch:d} Repeat guess ratio: {repeated_guess_ratio:0.2f}")

        hit_to_guess_tracker.append(hit_to_guess_ratio)
        wrong_guess_tracker.append(wrong_guess_ratio)
        correct_guess_tracker.append(correct_guess_ratio)
    
    # Save the model
    data = dict()
    data['model'] = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
    }
    
    data['hit_to_guess_tracker'] = hit_to_guess_tracker
    data['wrong_guess_tracker'] = wrong_guess_tracker
    data['correct_guess_tracker'] = correct_guess_tracker
    torch.save(data, "battleship_data.pth")
    
    
