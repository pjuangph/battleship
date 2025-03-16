from typing import List
import numpy as np 
import numpy.typing as npt

def place_ships(board_height:int=10,board_width:int=10,ship_sizes:List[int]=[2,3,3,4,5]) -> np.ndarray:
    """ Return random ship positions."""
    board = np.zeros(shape=(board_width,board_height), dtype=np.float32)
    board_size = board_width * board_height
    def can_place_ship(x, y, length, direction):
        """Check if a ship can be placed at (x, y) in a given direction without overlapping."""
        if direction == "H":  # Horizontal
            if y + length > board_height:
                return False
            return all(board[x, y+i] == 0 for i in range(length))
        else:  # Vertical
            if x + length > board_width:
                return False
            return all(board[x+i, y] == 0 for i in range(length))

    def place_ship(x, y, length, direction):
        """Place a ship at (x, y) in a given direction."""
        for i in range(length):
            if direction == "H":
                board[x, y+i] = 1  # Mark ship presence
            else:
                board[x+i, y] = 1

    for ship_size in ship_sizes:
        placed = False
        while not placed:
            x, y = np.random.randint(0, board_width-1),np.random.randint(0, board_height-1)
            direction = np.random.choice(["H", "V"])  # Horizontal or Vertical
            
            if can_place_ship(x, y, ship_size, direction):
                place_ship(x, y, ship_size, direction)
                placed = True    
    return np.reshape(board, (1, board_size)) 

def print_board(board: npt.NDArray, predicted_board: npt.NDArray = None):
    """Print the board with proper alignment."""
    
    cols = board.shape[1]  # Number of columns
    col_width = 1  # Space for each number
    separator = " | "  # Column separator

    # Header row
    column_numbers = "    " + separator.join(f"{i:{col_width}}" for i in range(cols))
    top_border = "  " + "-" * (cols * (col_width + 3) - 1)

    if predicted_board is not None:
        column_numbers += "          " + separator.join(f"{i:{col_width}}" for i in range(predicted_board.shape[1]))
        top_border += "      " + "-" * (predicted_board.shape[1] * (col_width + 3) - 1)
    print("Current Board \t\t\t\t\t Predicted Board")
    print(column_numbers)
    print(top_border)

    # Print board row by row
    for i, row in enumerate(board):
        row_str = f"{i:{col_width}} | " + separator.join(f"{int(cell):{col_width}}" for cell in row) + " |"
        if predicted_board is not None:
            row_p = predicted_board[i, :]
            row_str += "    " + f"{i:{col_width}} | " + separator.join(f"{int(cell):{col_width}}" for cell in row_p) + " |"
        print(row_str)

