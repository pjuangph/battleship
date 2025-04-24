from typing import List, Tuple
import torch
from transformer import Transformer
import numpy as np 
import numpy.typing as npt
from ship_placements import place_ships, print_board
from tqdm import trange
from battleship_train import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def bomb_index_to_human_readable(bomb_index:int,board_width:int)->str:
    """Convert bomb index to human-readable format"""
    row = bomb_index // board_width
    col = int(bomb_index - row * board_width)
    # convert col to letter
    row_str = chr(col+65)
    return f"{row_str}-{row+1}"

def human_readable_to_bomb_index(human_readable:str, board_width:int) -> int:
    """Convert human-readable format to bomb index"""
    
    # Split the input string into row and column parts
    col_str, row_str = human_readable.split('-')
    
    # Convert the row letter (e.g. 'J') to a column index
    col = ord(col_str.upper()) - 65
    
    # Convert the row number (e.g. '2') to a zero-based index
    row = int(row_str) - 1
    
    # Calculate the bomb index from row and column
    bomb_index = row * board_width + col
    return bomb_index

def run_inference(model:torch.nn.Module,current_board:torch.Tensor)->Tuple[np.ndarray,np.ndarray]:
    """Calls the model and returns the bomb indices
    This function is used to run inference on the model and get the bomb indices.
    It takes the current board as input and returns the bomb indices.

    Args:
        model (torch.nn.Module): _Transformer model
        current_board (torch.Tensor): current board 

    Returns:
        Tuple containing:
            - bomb_indices (np.ndarray): Indices of the bombs
            - predicted_board (np.ndarray): Predicted board

    """
    board_width  = current_board.shape[0]
    board_length = current_board.shape[1]
    
    board_size = current_board.shape[0]*current_board.shape[1]
    current_board = torch.tensor(current_board, dtype=torch.long).to(device)
    current_board = torch.reshape(current_board, (1, board_size)).to(device)

    with torch.no_grad():  # Disable gradient computation for speedup
        memory = model.encoder(current_board)
        output = current_board.clone()        
        output = model.decoder(memory, output)
        output = torch.argmax(output, dim=-1)  # Shape: (batch_size, seq_length)
    predicted_token_ids = output.cpu().numpy()
    
    bomb_indices = np.where(predicted_token_ids==2)[1]
    # Convert prediction back to matrix 
    return bomb_indices,predicted_token_ids.reshape((board_length,board_width))

def ai_helper():
    """AI helps you win
    """
    ship_sizes = [2,3,3,4,5]
    board_height = 10
    board_width = 10
    model,_,_,_ = load_model()
    percent_of_board_to_guess = 0.15

    hits = 0 
    guesses = 0
    past_predictions = list()
    current_board = np.zeros(shape=(1,board_height*board_width), dtype=np.int64)  # 0 no bomb, 1 bomb, 2 hit
    bomb_guesses = np.arange(board_height*board_width)

    while guesses < board_height*board_width and hits < sum(ship_sizes):
        if guesses < board_width*board_height*percent_of_board_to_guess:
            bomb_index = np.random.choice(bomb_guesses)
        else:
            bomb_locations,predicted_board = run_inference(model,torch.tensor(current_board))  
            print_board(current_board.reshape(board_height,board_width),predicted_board.reshape(board_height,board_width))
            for p in past_predictions:              
                bomb_locations = np.delete(bomb_locations, np.where(bomb_locations == p)) 
            if len(bomb_locations) == 0:
                bomb_index = np.random.randint(0,board_height*board_width-1) 
                print("No more bomb locations, making a random guess")
            else:
                bomb_index = np.random.choice(bomb_locations)
        
        human_readable_bomb_index = bomb_index_to_human_readable(bomb_index,board_width)
        bomb_guesses = np.delete(bomb_guesses, np.where(bomb_guesses == bomb_index))
        past_predictions.append(bomb_index)

        # Loop until the user enters a valid input
        while True:
            try:
                print(f"AI Guess: {human_readable_bomb_index}")
                guess = input("Enter guess or hit [Enter] for AI Guess: ").lower()
                if guess == "":
                    guess = human_readable_bomb_index
                else:
                    guess = guess.upper()
                
                bomb_index = human_readable_to_bomb_index(guess,board_width)
                hit = input("Hit (y) or (n): ").lower()  # Convert to lowercase for consistency

                if hit == 'y' or hit == 'n':
                    hit = 1 if hit == 'y' else 0
                    # Update board
                    if hit == 1:
                        current_board[0, bomb_index] = 2
                        hits += 1
                    else:
                        current_board[0, bomb_index] = 1
                    break  # Exit the loop once a valid input is entered
                else:
                    print("Invalid input. Please enter 'y' for hit or 'n' for miss.")
            except ValueError:
                print("Invalid input. Please enter a valid guess (e.g., A1, B2, etc.).")
                continue
            
        current_board[0,bomb_index] = 2 if hit else 1
        guesses += 1

def auto_game(n_games:int=1,train:bool=False):
    ship_sizes = [2,3,3,4,5]
    board_height = 10
    board_width = 10
    model,optimizer,_,data = load_model()
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = model.to(device)
    pbar = trange(n_games)
    for game in pbar:
        hits = 0
        bomb_index = -1
        past_predictions = [-1]
        guesses = 0     
        ship_positions = place_ships(board_height,board_width,ship_sizes)
        ship_position_indices = np.where(ship_positions == 1)[1]
        bomb_guesses = np.arange(board_height*board_width)

        hit_or_miss = ""
        if n_games==1:
            print("Ship positions:")
            print_board(ship_positions.reshape(board_height,board_width))
            
        if train:
            ships = sum(ship_sizes)
            no_ships = board_height*board_width - ships
            class_counts = torch.tensor([0, no_ships, ships])
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum()
            class_weights = class_weights.to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=0)            
            tgt_vocab_size = 3 
        # 0 no bomb, 1 bomb, 2 hit
        # Store all the games played 
        games = np.zeros(shape=(board_height*board_width, board_height, board_width), dtype=np.int64)
        game_index = 0
        percent_of_board_to_guess = 0.15
        current_board = torch.from_numpy(np.zeros(shape=(board_height,board_width), dtype=np.int64)).type(torch.long)  # 0 no bomb, 1 bomb, 2 hit

        while guesses < board_height*board_width and hits < sum(ship_sizes):
            # Make the guess
            if guesses < board_width*board_height*percent_of_board_to_guess:
                bomb_index = np.random.choice(bomb_guesses)
            else:
                bomb_locations,predicted_board = run_inference(model,current_board)  
                for p in past_predictions:              
                    bomb_locations = np.delete(bomb_locations, np.where(bomb_locations == p)) 
                if len(bomb_locations) == 0:
                    bomb_index = np.random.randint(0,board_height*board_width-1) 
                    print("No more bomb locations, making a random guess")
                else:
                    bomb_index = np.random.choice(bomb_locations)
            
            human_readable_bomb_index = bomb_index_to_human_readable(bomb_index,board_width)
            # Test the guess
            current_board = current_board.reshape((1,board_height*board_width)).to(device)
            current_board[0,bomb_index] = 2 * (bomb_index in ship_position_indices) + 1 * (bomb_index not in ship_position_indices)  # 0 no bomb, 1 bomb, 2 hit                    
                                     
            if (current_board[0,bomb_index] == 2):
                hit_or_miss = "hit"
                hits += 1
            else:
                hit_or_miss = "miss"
            current_board = current_board.reshape((board_height,board_width))
            
            if n_games==1:
                print(f"\nGuessed {guesses} human_readable_bomb_index {human_readable_bomb_index} bomb_index {bomb_index} {hit_or_miss}")
                if guesses < board_width*board_height*percent_of_board_to_guess:
                    print_board(current_board)
                else:
                    print_board(current_board, predicted_board)
                    
            bomb_guesses = np.delete(bomb_guesses, np.where(bomb_guesses == bomb_index))
            past_predictions.append(bomb_index)
            if guesses > board_width*board_height*percent_of_board_to_guess:
                games[game_index,:,:] = current_board.detach().cpu().numpy()
                game_index+=1
            guesses += 1

        if n_games==1:
            print(f"total hits {hits}")
            print_board(current_board.reshape(board_height,board_width)) 
                       
        if train: # Train the board on the game it just played 
            src = torch.tensor(games[:game_index-1,:,:]).reshape(game_index-1,board_height*board_width).to(device)
            tgt = src.clone()
            tgt[:,:] = torch.where(current_board == 0, torch.tensor(1), current_board).reshape(1,board_height*board_width).to(device)

            print(f'Training on {game_index-1} games')        
            model.train()
            optimizer.zero_grad()
            output = model(src,tgt)
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt.view(-1).contiguous().long())  # Convert to long
            loss.backward()
            optimizer.step()
            if (game%50 == 0):
                scheduler.step()
    if train: # Save the model state as auto_game
        print(f'Train Loss: {loss.item():0.2e}')
        data['model']['state_dict'] = model.state_dict()
        data['optimizer'] = optimizer.state_dict()
        torch.save(data, "data/trained_model_auto_game.pth")
        
if __name__=="__main__":
    ai_helper()
    # auto_game(n_games=1000, train=True)
    # auto_game(n_games=1,train=False)
    