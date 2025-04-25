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
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.cuda.empty_cache()
SHIP_SIZES = [2,3,3,4,5]
board_height = 10
board_width = 10


def generate_game_data(nboards:int,board_height:int,board_width:int,ship_sizes:List[int]) -> Tuple[npt.NDArray,npt.NDArray]:
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
    percent_of_src_to_generate = 0.10
    number_of_guesses = int(board_height * board_width*(1-percent_of_src_to_generate))
    src_board = np.zeros((nboards*number_of_guesses,board_height*board_width))
    tgt_board = np.zeros((nboards*number_of_guesses,board_height*board_width))
    for indx in trange(nboards):
        ship_positions = place_ships(board_height,board_width,ship_sizes)
        ship_position_indices = np.where(ship_positions == 1)[1]
        bomb_locations = np.arange(board_height*board_width)
        for guess in range(number_of_guesses):
            tgt_board[indx*number_of_guesses+guess,:] = 2*(ship_positions  == 1) + 1*(ship_positions == 0)
        
        for p in range(board_height*board_width-number_of_guesses): # Lets guess 15 % of the board before we begin training 
            bomb_index = np.random.choice(bomb_locations)            
            src_board[indx*number_of_guesses,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)
            bomb_locations = np.delete(bomb_locations, np.where(bomb_locations == bomb_index))
            
        for guess in range(1,number_of_guesses):
            src_board[indx*number_of_guesses+guess,:] = src_board[indx*number_of_guesses+guess-1,:]
            bomb_index = np.random.choice(bomb_locations)            
            src_board[indx*number_of_guesses+guess,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)
            bomb_locations = np.delete(bomb_locations, np.where(bomb_locations == bomb_index))   
          
    return src_board,tgt_board

def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)  # Upper triangular mask
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def apply_random_mask(tgt: torch.Tensor, mask_token: int = 0, mask_prob: float = 0.15):
    masked_tgt = tgt.clone()
    labels = tgt.clone()

    # Create random mask
    mask = torch.rand(tgt.shape, device=tgt.device) < mask_prob

    # Replace input with mask token
    masked_tgt[mask] = mask_token

    # Optionally, ignore loss on unmasked tokens using ignore_index
    loss_mask = mask  # use this to mask the loss later

    return masked_tgt, labels, loss_mask

def generate_and_save_games(ngames:int=2000,board_height:int=10,board_width:int=10,ship_sizes:List[int]=SHIP_SIZES):
    """Generate Games 

    Args:
        ngames (int, optional): number of games to generate. Defaults to 2000.
        board_height (int, optional): board height. Defaults to 10.
        board_width (int, optional): board width. Defaults to 10.
        ship_sizes (List[int], optional): ship sizes to use. Defaults to SHIP_SIZES.
    """
    print("Generating Games to play")
    src,tgt = generate_game_data(ngames,board_height,board_width,ship_sizes)

    os.makedirs('data',exist_ok=True)
    data = {'src':src,'tgt':tgt}
    pickle.dump(data,open('data/training_data.pickle','wb'))


def load_model():
    path = 'data'
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith('.pth') and os.path.isfile(os.path.join(path, f))
    ]
    filename = max(files, key=os.path.getmtime)

    data = torch.load(filename,map_location=device)

    model = Transformer(src_vocab_size=data['model']['src_vocab_size'],
                        tgt_vocab_size=data['model']['tgt_vocab_size'],
                        d_model=data['model']['d_model'],
                        num_heads=data['model']['num_heads'],
                        num_layers=data['model']['num_layers'],
                        d_ff=data['model']['d_ff'],
                        max_seq_length=data['model']['max_seq_length'],
                        dropout=data['model']['dropout']).to(device)

    model.load_state_dict(data['model']['state_dict'])
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    optimizer.load_state_dict(data['optimizer'])
    try:
        epochs = data['model']['epochs']
    except:
        epochs = 0
    print(f"Loaded model with {epochs} epochs")
    return model,optimizer,epochs,data

def train(resume_training:bool=False,save_every_n_epoch:int=10):
    """This function will train the model using the data generated by generate_game_data.

    Args:
        resume_training (bool, optional): Resume training. Defaults to False.
        save_every_n_epoch (int, optional): Save every n epochs. Defaults to 10.
    """
    epochs = 100

    src_vocab_size = 3
    tgt_vocab_size = 3 # 0, 1, 2
    d_model = 512
    num_heads = 4
    num_layers = 4
    d_ff = 2048
    max_seq_length = board_height*board_width
    dropout = 0.1
    # Instantiate model
    model = Transformer(src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size, 
                        d_model=d_model, num_heads=num_heads, num_layers=num_layers, 
                        d_ff=d_ff, 
                        max_seq_length=max_seq_length, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    if (not osp.exists("data/training_data.pickle")):
        generate_and_save_games(ngames=20000,board_height=board_height,board_width=board_width,ship_sizes=SHIP_SIZES)
    
    data = pickle.load(open('data/training_data.pickle','rb'))

    if resume_training:
        model,optimizer,current_epochs,_ = load_model()
    else:
        current_epochs = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)


    def train_loop(src:npt.NDArray,tgt:npt.NDArray):
        # Train the model
        src_train, src_test, tgt_train, tgt_test = train_test_split(src, tgt, test_size=0.3,shuffle=True)
        src_train_tensor = torch.tensor(src_train, dtype=torch.long)
        tgt_train_tensor = torch.tensor(tgt_train, dtype=torch.long)
        src_test_tensor = torch.tensor(src_test, dtype=torch.long)
        tgt_test_tensor = torch.tensor(tgt_test, dtype=torch.long)
        
        train_dataset = TensorDataset(src_train_tensor, tgt_train_tensor)       # Create a dataset
        test_dataset = TensorDataset(src_test_tensor, tgt_test_tensor)       # Create a dataset

        batch_size = 128
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Calculate class weights (adjust manually if needed)
        all_targets = tgt_train_tensor.view(-1)
        class_counts = torch.bincount(all_targets, minlength=3).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
        criterion_train = nn.CrossEntropyLoss(weight=class_weights,ignore_index=0)
        
        all_targets = tgt_test_tensor.view(-1)
        class_counts = torch.bincount(all_targets, minlength=3).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()
        class_weights = class_weights.to(device)
        criterion_val = nn.CrossEntropyLoss(weight=class_weights,ignore_index=0)

        for epoch in range(epochs):
            model.train()
            pbar = tqdm(train_loader)
            for batch in pbar:      
                optimizer.zero_grad()
          
                src_batch, tgt_batch = batch
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)

                guessed_mask = (tgt_batch != 0)
                random_mask = (torch.rand_like(tgt_batch.float()) > 0.9).to(device)
                final_mask = guessed_mask & random_mask
                tgt_batch_masked = tgt_batch.clone()
                tgt_batch_masked[~final_mask] = 0
                with autocast(device_type='cuda', dtype=torch.float16):
                    output = model(src_batch, tgt_batch_masked) 
                    loss = criterion_train(output.view(-1, tgt_vocab_size), tgt_batch.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                output_tokens = output.argmax(dim=-1)
                hits = torch.sum(output_tokens == 2).detach().cpu() 
                matches = torch.sum(output_tokens == tgt_batch).detach().cpu()
                # print(torch.sum(matches))
                pbar.set_description(f"Epoch: {epoch+current_epochs:d} Train Loss: {loss.item():0.2e} Hits match {hits/batch_size:0.2f} Matches {matches/batch_size:0.2f}")
                

            pbar = tqdm(test_loader)
            total_val_loss = 0; num_batches = 0
            model.eval()
            for batch in pbar:
                src_batch, tgt_batch = batch
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
      
                output = model(src_batch, tgt_batch)
                val_loss = criterion_val(output.view(-1, tgt_vocab_size), tgt_batch.view(-1))

                pred_classes = output.argmax(dim=-1)
                # print(src_batch[0,:])
                # print(pred_classes[0,:])

                total_val_loss += val_loss.item()
                num_batches += 1
                pbar.set_description(f"Epoch: {epoch+current_epochs:d} Train Loss: {loss.item():0.2e} Val Loss: {val_loss.item():0.2e}")
            average_val_loss = total_val_loss / num_batches  # Compute average validation loss
            pbar.set_description(f"Epoch: {epoch+current_epochs:d} Train Loss: {loss.item():0.2e} Val Loss: {average_val_loss:0.2e}")
            scheduler.step()
            if (epoch % save_every_n_epoch == 0) or (epoch == epochs-1):
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
                    'dropout': dropout,
                    'epochs':epoch+current_epochs
                }
                data['optimizer'] = optimizer.state_dict()
                torch.save(data, f"data/trained_model-{epoch+current_epochs}.pth")
                if not resume_training:
                    torch.save(data, "data/trained_model.bak.pth")
                print(f"Saved model at epoch {epoch+current_epochs}")

    model.to(device)
    src = data['src']
    tgt = data['tgt']    
    
    print("Train Loop")
    train_loop(src,tgt)
    
if __name__ =="__main__":
    # generate_and_save_games(10000,board_height,board_width,SHIP_SIZES)
    train(resume_training=True)
