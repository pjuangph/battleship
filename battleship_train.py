'''
    Board values: -1 for no guesses, 0 bomb, 1 for hit
'''

from typing import List, Tuple
import torch.nn as nn
import torch.optim as optim
import torch, os
from torch import Tensor
import numpy as np
import pickle 
from ship_placements import place_ships
from transformer import Transformer 
import os.path as osp 
from tqdm import trange
import random 

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

class BattleshipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, current_board: Tensor, action_index: Tensor,ship_sizes:Tensor) -> Tensor:
        """Compute the loss of the battleship game."""
        
        # Ensure action_index is a tensor
        if not isinstance(action_index, Tensor):
            action_index = torch.tensor(action_index, dtype=torch.float16, device=device, requires_grad=True)

        # Convert board to float tensor for differentiability
        current_board = current_board.float()
        current_board.requires_grad = True

        # Compute correct and wrong guesses using differentiable operations
        correct_guesses = torch.sum((current_board == 2).float(),dim=1)
        cumsum_correct_guesses = torch.cumsum((current_board==2),dim=0)
        action_index_each_game = torch.argmax((cumsum_correct_guesses == torch.sum(ship_sizes)).float(), dim=-1)+action_index
        no_guesses = torch.sum((current_board == 0).float(),dim=1)
        misses = torch.sum((current_board == 1).float(),dim=1)
        
        misses.requires_grad = True
        correct_guesses.requires_grad = True
        # Compute loss with floating point division
        loss = torch.mean((misses) / (correct_guesses**2 + 1e-6))  # Avoid division by zero

        return loss,action_index_each_game

# Define loss and optimizer
def play_game(model:nn.Module,optimizer, target:Tensor,training:bool=False,board_height:int=10,board_width:int=10,ship_sizes:List[int]=[]) -> Tuple[Tensor,Tensor,Tensor]:
    """ Play game of battleship using network."""
    if training:
        model.train()
    else:
        model.eval()
   
    board_size = board_height * board_width
    
    action_log = torch.zeros((board_size), dtype=torch.int32,device=device)
    hit_log = torch.zeros((board_size), dtype=torch.int32,device=device)

    current_board = torch.from_numpy(np.zeros(shape=(1,board_size), dtype=np.int64)).type(torch.int32).to(device)  # 0 no bomb, 1 bomb, 2 hit
    ship_sizes = torch.tensor(ship_sizes, dtype=torch.float16, device=device, requires_grad=True)
    # loss_fn = BattleshipLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    action_index = 0
    while (torch.min(torch.sum(hit_log,dim=-1)) < sum(ship_sizes)) and (action_index < board_size):
        optimizer.zero_grad()
        src = current_board.clone()
        output = model(target,target)

        val_loss = criterion(output.contiguous().view(-1, board_size), current_board.view(-1).contiguous().long())  # Convert to long

        # loss,action_index_for_each_game = loss_fn(current_board,action_index,ship_sizes)
        # loss.backward()
        val_loss.backward()
        optimizer.step()
        print(f"Action Index: {action_index:d}, Loss: {val_loss.item():0.3e}")
    return hit_log, action_log, current_board, action_index

def train():
    epochs = 10
    batches = 50
    batch_size = 32 

    board_height = 10
    board_width = 10
    SHIP_SIZES = [2,3,3,4,5]

    src_vocab_size = board_height*board_width
    tgt_vocab_size = board_height*board_width
    d_model = 256
    num_heads = 4
    num_layers = 10
    d_ff = 2048
    max_seq_length = board_height*board_width
    dropout = 0.1
    # Instantiate model
    model = Transformer(src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size, 
                        d_model=d_model, num_heads=num_heads, num_layers=num_layers, 
                        d_ff=d_ff, 
                        max_seq_length=max_seq_length, dropout=dropout).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    if (not osp.exists("data/training_data.pickle")):
        print("Generating Games to play")
        src_board = np.zeros((batches,batch_size,board_height*board_width))
        tgt_board = np.zeros((batches,batch_size,board_height*board_width))
        for batch_index in trange(batches):
            for batch in range(batch_size):
                ship_positions = place_ships(board_height,board_width,SHIP_SIZES)
                ship_position_indices = np.where(ship_positions == 1)[1]
                for bomb_index in range(board_width*board_height):
                    tgt_board[batch_index,batch,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)  # 0 no bomb, 1 bomb, 2 hit
                src_board[batch_index,batch,:] = tgt_board[batch_index,batch,:] * np.random.choice([0, 1], size=board_width*board_height, p=[0.7, 0.3])

        os.makedirs('data',exist_ok=True)
        pickle.dump({'src':src_board,'tgt':tgt_board},open('data/training_data.pickle','wb'))
    else:
        data = pickle.load(open('data/training_data.pickle','rb'))
        src_board = data['src']
        tgt_board = data['tgt']

    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Training the model")
    pbar = trange(epochs)
    for epoch in pbar:
        for batch_indx in range(batches):
            src = src_board[batch_indx,:,:]
            tgt = tgt_board[batch_indx,:,:]
            optimizer.zero_grad()
            src = torch.from_numpy(src).type(torch.LongTensor).to(device)  # 0 no bomb, 1 bomb, 2 hit
            tgt = torch.from_numpy(tgt).type(torch.LongTensor).to(device)  
            output = model(src,tgt,0.15)
            val_loss = criterion(output.contiguous().view(-1, board_height*board_width), tgt.view(-1).contiguous().long())  # Convert to long
            val_loss.backward()
            optimizer.step()
        pbar.set_description(f"Action Index: {epoch:d}, Loss: {val_loss.item():0.2e}")

    print("Training with 50% dropout")
    pbar = trange(epochs)
    for epoch in pbar:
        for batch_indx in range(batches):
            src = src_board[batch_indx,:,:]
            tgt = tgt_board[batch_indx,:,:]
            optimizer.zero_grad()
            src = torch.from_numpy(src).type(torch.LongTensor).to(device)  # 0 no bomb, 1 bomb, 2 hit
            tgt = torch.from_numpy(tgt).type(torch.LongTensor).to(device)  
            output = model(src,tgt,0.50)
            val_loss = criterion(output.contiguous().view(-1, board_height*board_width), tgt.view(-1).contiguous().long())  # Convert to long
            val_loss.backward()
            optimizer.step()
        pbar.set_description(f"Action Index: {epoch:d}, Loss: {val_loss.item():0.2e}")
    
    # Train the encoder to guess the target based on partial information
    # Save the model
    data = dict()
    data['model'] = {
        'state_dict': model.state_dict(),
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
        'max_seq_length': max_seq_length,
        'dropout': dropout
    }
    data['optimizer'] = optimizer.state_dict()
    torch.save(data, "data/trained_model.pth")

if __name__ =="__main__":
    train()
