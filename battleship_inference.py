from typing import List
import torch
from transformer import Transformer
import numpy as np 
import numpy.typing as npt
from ship_placements import place_ships, print_board
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    data = torch.load('data/trained_model.pth')

    model = Transformer(src_vocab_size=data['model']['src_vocab_size'],
                        tgt_vocab_size=data['model']['tgt_vocab_size'],
                        d_model=data['model']['d_model'],
                        num_heads=data['model']['num_heads'],
                        num_layers=data['model']['num_layers'],
                        d_ff=data['model']['d_ff'],
                        max_seq_length=data['model']['max_seq_length'],
                        dropout=data['model']['dropout']).to(device)

    model.load_state_dict(data['model']['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    optimizer.load_state_dict(data['optimizer'])

    model.eval()
    return model,optimizer
 
# tgt_start = torch.tensor([[start_token]])  # Start decoding with "<SOS>"
# output = transformer(src, tgt_start)  # First decoder step

# for _ in range(max_len):  # Generate tokens until "<EOS>" appears
#     next_token = output[:, -1]  # Get last generated token
#     tgt_start = torch.cat([tgt_start, next_token], dim=1)  # Append new token
#     output = transformer(src, tgt_start)  # Decode again
#     if next_token == eos_token:
#         break
    
def run_inference(model:torch.nn.Module,current_board:torch.Tensor)->str:
    board_width  = current_board.shape[0]
    board_length = current_board.shape[1]
    
    board_size = current_board.shape[0]*current_board.shape[1]
    current_board = torch.reshape(current_board, (1, board_size)).to(device)
    current_board = torch.tensor(current_board, dtype=torch.long).to(device)
    target_board = torch.zeros((current_board.shape),dtype=torch.long).to(device)

    bomb_index = -1
    with torch.no_grad():  # Disable gradient computation for speedup
        output = model(current_board,current_board*0)
        predicted_token_ids = torch.argmax(output, dim=-1).cpu()  # Shape: (batch_size, seq_length)

    predicted_token_ids = predicted_token_ids.numpy()
    bomb_indices = np.where(predicted_token_ids==2)[1]
    # Convert prediction back to matrix 
    human_readable_bomb_locations = []
    bomb_locations = []
    for bomb_index in bomb_indices:
        row = bomb_index // board_width
        col = int(bomb_index - row * board_width)
        # convert col to letter
        row_str = chr(col+65)
        human_readable_bomb_locations.append(f"{row_str}-{row+1}")
        bomb_locations.append(bomb_index)
    return human_readable_bomb_locations,bomb_locations,predicted_token_ids.reshape((board_length,board_width))

def ai_helper():
    """AI helps you win
    """
    # Game configuration
    ship_sizes = [2,3,3,4,5]
    board_height = 10
    board_width = 10

    board = np.zeros(shape=(board_width,board_height), dtype=np.float32)
    hits = []
    past_predictions = []
    guesses = 0     
    
    while guesses < board_height*board_width and sum(hits) < sum(ship_sizes):
        guess, past_predictions,row,col,_ = run_inference(board, past_predictions)
        print(f"AI Guess: {guess}")
        hit = input("Hit (y) or (n): ")
        hit = 1 if hit == 'y' else 0
        # Update board
        board[row,col] = 2 if hit else 1
        hits.append(hit)
        guesses += 1

def auto_game(n_games:int=1,train:bool=False):
    ship_sizes = [2,3,3,4,5]
    board_height = 10
    board_width = 10
    model,optimizer = load_model()
    model = model.to(device)
    pbar = trange(n_games)
    for game in pbar:
        hits = 0
        bomb_index = -1
        past_predictions = [-1]
        guesses = 0     
        ship_positions = place_ships(board_height,board_width,ship_sizes)
        ship_position_indices = np.where(ship_positions == 1)[1]
        hit_or_miss = ""
        if n_games==1:
            print("Ship positions:")
            print_board(ship_positions.reshape(board_height,board_width))
            
        if train:
            criterion = torch.nn.CrossEntropyLoss()
            tgt_vocab_size = 3 
            
        current_board = torch.from_numpy(np.zeros(shape=(board_height,board_width), dtype=np.int64)).type(torch.long)  # 0 no bomb, 1 bomb, 2 hit
        while guesses < board_height*board_width and hits < sum(ship_sizes):
            human_readable_bomb_locations, bomb_locations,predicted_board = run_inference(model,current_board)
            tries = 0
            while (bomb_index in past_predictions) & (tries <100):
                if len(bomb_locations) == 0:
                    bomb_index = np.random.randint(0,board_height*board_width-1)
                    human_readable_bomb_index = bomb_index
                else:
                    bomb_index = np.random.choice(bomb_locations)
                    while bomb_index in past_predictions:
                        bomb_index = np.random.randint(0,board_height*board_width-1)
                    try:                        
                        human_readable_bomb_index = human_readable_bomb_locations[np.where(bomb_locations==bomb_index)[0][0]]
                    except:
                        human_readable_bomb_index = ""
                        pass
                tries+=1
            past_predictions.append(bomb_index)
            current_board = current_board.reshape((1,board_height*board_width)).to(device)

            prev_board = current_board.detach().clone().to(device)
            current_board[0,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)  # 0 no bomb, 1 bomb, 2 hit        
            if train & (guesses>50):
                model.train()
                optimizer.zero_grad()
                output = model(prev_board,current_board)
                loss = criterion(output.contiguous().view(-1, tgt_vocab_size), current_board.view(-1).contiguous().long())  # Convert to long
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Game: {game:d} Guess: {guesses:d} Train Loss: {loss.item():0.2e}")
                                     
            if (current_board[0,bomb_index] == 2) & (tries < 20):
                hit_or_miss = "hit"
                hits += 1
            else:
                hit_or_miss = "miss"
            current_board = current_board.reshape((board_height,board_width))
            if n_games==1:
                print(f"\nGuessed {guesses} {human_readable_bomb_index} {hit_or_miss}")
                print_board(current_board, predicted_board)
                
            guesses += 1
        if n_games==1:
            print(f"total hits {hits}")
            print_board(current_board.reshape(board_height,board_width)) 
                       
    # if train:
    #     data = torch.load('data/trained_model.pth')
    #     data['model']['state_dict'] = model.state_dict()
    #     torch.save(data, "data/trained_model.pth")
        
if __name__=="__main__":
    # game_helper()
    # auto_game(n_games=100, train=True)
    auto_game()