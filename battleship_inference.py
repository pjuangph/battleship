from typing import List
import torch
from transformer import Transformer
import numpy as np 
import numpy.typing as npt
from ship_placements import place_ships, print_board

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load('battleship_data.pth')

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
optimizer.load_state_dict(data['model']['optimizer'])

model.eval()


def run_inference(current_board:npt.NDArray,past_predictions:List[int],board_height:int=10,board_width:int=10)->str:
    board_size = current_board.shape[0]*current_board.shape[1]
    current_board = np.reshape(current_board, (1, board_size)) 
    current_board = torch.tensor(current_board, dtype=torch.int64).to(device)
    bomb_index = -1
    with torch.no_grad():  # Disable gradient computation for speedup
        while bomb_index not in past_predictions and len(past_predictions)-1<board_size:
            prediction = model.encode(current_board)
            bomb_index = torch.argmax(prediction,dim=1)  # Get the bomb index for all the games    
            if bomb_index not in past_predictions:
                past_predictions.append(int(bomb_index))
                break
        
    # Convert prediction back to matrix 
    row = bomb_index // board_width
    col = int(bomb_index - row * board_width)
    # convert col to letter
    row_str = chr(row+65)
    return f"{row_str}-{col}", past_predictions, row, col, bomb_index

def game_helper():
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

    hits = []
    past_predictions = []
    guesses = 0     
    ship_positions = place_ships(board_height,board_width,ship_sizes)
    ship_position_indices = np.where(ship_positions == 1)[1]

    print("Ship positions:")
    print_board(ship_positions.reshape(board_height,board_width))

    current_board = torch.from_numpy(np.zeros(shape=(1,board_height*board_width), dtype=np.int64)).type(torch.int32)  # 0 no bomb, 1 bomb, 2 hit
    while guesses < board_height*board_width and sum(hits) < sum(ship_sizes):
        guess, past_predictions,row,col, bomb_index= run_inference(current_board, past_predictions)
        current_board[0,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)  # 0 no bomb, 1 bomb, 2 hit
        guesses += 1
    print_board(current_board.reshape(board_height,board_width))

if __name__=="__main__":
    # game_helper()
    auto_game()