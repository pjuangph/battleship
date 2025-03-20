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
from tqdm import trange,tqdm
import numpy.typing as npt  
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

def generate_game_data(nboards:int,board_height:int,board_width:int,ship_sizes:List[int],src_blank:float) -> Tuple[npt.NDArray,npt.NDArray]:
    """Generates dummy game data for training 

    Args:
        nboards (int): number boards to generate
        board_height (int): board height in units
        board_width (int): board width in units
        ship_sizes (List[int]): Array of ship sizes e.g. [2,3,3,4,5]
        src_blank (float): percent of source board to blank out 

    Returns:
        Tuple[npt.NDArray,npt.NDArray]: source, target
    """
    src_board = np.zeros((nboards,board_height*board_width))
    tgt_board = np.zeros((nboards,board_height*board_width))
    for indx in trange(nboards):
        ship_positions = place_ships(board_height,board_width,ship_sizes)
        ship_position_indices = np.where(ship_positions == 1)[1]
        for bomb_index in range(board_width*board_height):
            tgt_board[indx,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)  # 0 no bomb, 1 bomb, 2 hit
            src_board[indx,:] = tgt_board[indx,:] * np.random.choice([0, 1], size=board_width*board_height, p=[src_blank, 1-src_blank])
    return src_board,tgt_board
    
def train():
    epochs = 10
    ngames = 10000  # Number of games to generate

    board_height = 10
    board_width = 10
    SHIP_SIZES = [2,3,3,4,5]

    src_vocab_size = board_height*board_width
    tgt_vocab_size = 3 # 0, 1, 2
    d_model = 256
    num_heads = 4
    num_layers = 12
    d_ff = 1024
    max_seq_length = board_height*board_width
    dropout = 0.2
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
        src1,tgt1 = generate_game_data(ngames,board_height,board_width,SHIP_SIZES,0.3)
        src2,tgt2 = generate_game_data(ngames,board_height,board_width,SHIP_SIZES,0.5)
        src3,tgt3 = generate_game_data(ngames,board_height,board_width,SHIP_SIZES,0.7)

        os.makedirs('data',exist_ok=True)
        data = {'mask1':{
                            'src':src1,
                            'tgt':tgt1,
                        },
                'mask2':{
                            'src':src2,
                            'tgt':tgt2,
                        },
                'mask3':{
                            'src':src3,
                            'tgt':tgt3,
                        }
                }
        pickle.dump(data,open('data/training_data.pickle','wb'))
    else:
        data = pickle.load(open('data/training_data.pickle','rb'))


    def train_loop(src:npt.NDArray,tgt:npt.NDArray):
        # Train the model
        src_train, src_test, tgt_train, tgt_test = train_test_split(src, tgt, test_size=0.3,shuffle=True)
        src_train_tensor = torch.tensor(src_train, dtype=torch.long)
        tgt_train_tensor = torch.tensor(tgt_train, dtype=torch.long)
        src_test_tensor = torch.tensor(src_test, dtype=torch.long)
        tgt_test_tensor = torch.tensor(tgt_test, dtype=torch.long)
        
        train_dataset = TensorDataset(src_train_tensor, tgt_train_tensor)       # Create a dataset
        test_dataset = TensorDataset(src_test_tensor, tgt_test_tensor)       # Create a dataset

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_loader)
            for batch in pbar:
                src_batch, tgt_batch = batch
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                optimizer.zero_grad()
                
                # Masking some of source 1 to 0
                percent_of_map_to_mask = 0.1 + (0.8 - 0.1) * torch.rand(1).to(device)
                is_one = src_batch == 1
                # Generate a random mask for these positions
                random_mask = torch.rand_like(tgt_batch.float()) > percent_of_map_to_mask  # True = keep, False = mask
                # Apply the mask: keep original values if not '2' or if not selected for masking
                src_batch_mask = torch.where(is_one & ~random_mask, torch.tensor(0), src_batch)

                # Masking some of source 2 to 0 
                percent_of_map_to_mask = 0.1 + (0.8 - 0.1) * torch.rand(1).to(device)
                is_two = src_batch == 2
                # Generate a random mask for these positions
                random_mask = torch.rand_like(tgt_batch.float()) > percent_of_map_to_mask  # True = keep, False = mask
                # Apply the mask: keep original values if not '2' or if not selected for masking
                src_batch_mask = torch.where(is_two & ~random_mask, torch.tensor(0), src_batch_mask)
                
                # Masking some of the target 1 to 0 
                percent_of_map_to_mask = 0.1 + (0.8 - 0.1) * torch.rand(1).to(device)
                is_one = tgt_batch == 1
                # Generate a random mask for these positions
                random_mask = torch.rand_like(tgt_batch.float()) > percent_of_map_to_mask  # True = keep, False = mask
                # Apply the mask: keep original values if not '2' or if not selected for masking
                tgt_batch_mask = torch.where(is_one & ~random_mask, torch.tensor(0), tgt_batch)

                output = model(src_batch_mask,tgt_batch_mask)
                output_tokens = output.argmax(dim=-1)

                matches = (output_tokens == tgt_batch)
                # print(torch.sum(matches))
                loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch.view(-1).contiguous().long())  # Convert to long
                loss.backward()
                optimizer.step()

            pbar = tqdm(test_loader)
            total_val_loss = 0; num_batches = 0
            model.eval()
            for batch in pbar:
                src_batch, tgt_batch = batch
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                
                # Masking some of source 1 to 0
                percent_of_map_to_mask = 0.1 + (0.8 - 0.1) * torch.rand(1).to(device)
                is_one = src_batch == 1
                # Generate a random mask for these positions
                random_mask = torch.rand_like(tgt_batch.float()) > percent_of_map_to_mask  # True = keep, False = mask
                # Apply the mask: keep original values if not '2' or if not selected for masking
                src_batch_mask = torch.where(is_one & ~random_mask, torch.tensor(0), src_batch)

                # Masking some of source 2 to 0 
                percent_of_map_to_mask = 0.1 + (0.8 - 0.1) * torch.rand(1).to(device)
                is_two = src_batch == 2
                # Generate a random mask for these positions
                random_mask = torch.rand_like(tgt_batch.float()) > percent_of_map_to_mask  # True = keep, False = mask
                # Apply the mask: keep original values if not '2' or if not selected for masking
                src_batch_mask = torch.where(is_two & ~random_mask, torch.tensor(0), src_batch_mask)
                
                # Masking some of the target 1 to 0 
                percent_of_map_to_mask = 0.1 + (0.8 - 0.1) * torch.rand(1).to(device)
                is_one = tgt_batch == 1
                # Generate a random mask for these positions
                random_mask = torch.rand_like(tgt_batch.float()) > percent_of_map_to_mask  # True = keep, False = mask
                # Apply the mask: keep original values if not '2' or if not selected for masking
                tgt_batch_mask = torch.where(is_one & ~random_mask, torch.tensor(0), tgt_batch)

                output = model(src_batch_mask,tgt_batch_mask)
                val_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch.view(-1).contiguous().long())  # Convert to long
                total_val_loss += val_loss.item()
                num_batches += 1
                pbar.set_description(f"Epoch: {epoch:d} Train Loss: {loss.item():0.2e} Val Loss: {val_loss.item():0.2e}")
            average_val_loss = total_val_loss / num_batches  # Compute average validation loss
            pbar.set_description(f"Epoch: {epoch:d} Train Loss: {loss.item():0.2e} Val Loss: {average_val_loss:0.2e}")
    
    model.to(device)
    src1 = data['mask1']['src']
    tgt1 = data['mask1']['tgt']    
    src2 = data['mask2']['src']
    tgt2 = data['mask2']['tgt']
    src3 = data['mask3']['src']
    tgt3 = data['mask3']['tgt']
    

    print("Training with 25% source and target mask")
    train_loop(src1,tgt1)
    train_loop(src2,tgt2)
    train_loop(src3,tgt3)
    
    # print("Training with 20% source and target mask")
    # train_loop(src1,tgt1,0.4)
    # train_loop(src2,tgt2,0.4)
    # train_loop(src3,tgt3,0.4)
    

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
    torch.save(data, "data/trained_model.bak.pth")

if __name__ =="__main__":
    train()
