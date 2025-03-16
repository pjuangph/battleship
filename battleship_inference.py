from typing import List
import torch
from transformer import Transformer
import numpy as np 
import numpy.typing as npt
from ship_placements import place_ships, print_board

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
    return model 
 
def run_inference(model:torch.nn.Module,current_board:npt.NDArray)->str:
    board_width  = current_board.shape[0]
    board_length = current_board.shape[1]
    
    board_size = current_board.shape[0]*current_board.shape[1]
    current_board = np.reshape(current_board, (1, board_size))
    current_board = torch.tensor(current_board, dtype=torch.long).to(device)
    target_board = torch.zeros((current_board.shape),dtype=torch.long).to(device)

    bomb_index = -1
    with torch.no_grad():  # Disable gradient computation for speedup
        output = model(current_board,target_board)
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
        row_str = chr(row+65)
        human_readable_bomb_locations.append(f"{row_str}-{col}")
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

def auto_game():
    ship_sizes = [2,3,3,4,5]
    board_height = 10
    board_width = 10
    model = load_model()
    hits = []
    past_predictions = []
    guesses = 0     
    ship_positions = place_ships(board_height,board_width,ship_sizes)
    ship_position_indices = np.where(ship_positions == 1)[1]
    hit_or_miss = ""
    
    print("Ship positions:")
    print_board(ship_positions.reshape(board_height,board_width))

    current_board = torch.from_numpy(np.zeros(shape=(board_height,board_width), dtype=np.int64)).type(torch.long)  # 0 no bomb, 1 bomb, 2 hit
    while guesses < board_height*board_width and sum(hits) < sum(ship_sizes):
        human_readable_bomb_locations, bomb_locations,predicted_board = run_inference(model,current_board)
        
        print(f"\nGuess {guesses} ")
        print_board(current_board, predicted_board)
        bomb_index = np.random.choice(bomb_locations)
        
        current_board = current_board.reshape((1,board_height*board_width))
        index = np.where(bomb_locations == bomb_index)[0][0]
        human_readable_bomb_index = human_readable_bomb_locations[index]
        
        current_board[0,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)  # 0 no bomb, 1 bomb, 2 hit
        
        if current_board[0,bomb_index] == 2:
            hit_or_miss = "hit"
        else:
            hit_or_miss = "miss"
        print(f"Guessed {human_readable_bomb_index} {hit_or_miss}")
        guesses += 1
    print_board(current_board.reshape(board_height,board_width))

if __name__=="__main__":
    # game_helper()
    auto_game()