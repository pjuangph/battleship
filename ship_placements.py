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
    """Print the transposed board with columns as rows and rows as columns."""
    board_T = board.T
    predicted_T = predicted_board.T if predicted_board is not None else None

    rows = board_T.shape[0]  # Was columns, now rows after transpose
    col_width = 1
    separator = " | "

    # Header (letters become numbers now)
    header_row = "     " + separator.join(f"{i+1:{col_width}}" for i in range(board_T.shape[1]))
    top_border = "  " + "-" * (board_T.shape[1] * (col_width + 3) - 1)

    if predicted_T is not None:
        header_row += "          " + separator.join(f"{i+1:{col_width}}" for i in range(predicted_T.shape[1]))
        top_border += "      " + "-" * (predicted_T.shape[1] * (col_width + 3) - 1)

    print("Current Board \t\t\t\t\t Predicted Board")
    print(header_row)
    print(top_border)

    # Each "row" is now a column label (A to J)
    for i, row in enumerate(board_T):
        row_label = chr(65 + i)  # A, B, C, ..., J
        row_str = f"{row_label:>2} | " + separator.join(f"{int(cell):{col_width}}" for cell in row) + " |"

        if predicted_T is not None:
            row_p = predicted_T[i, :]
            row_str += "    " + f"{row_label:>2} | " + separator.join(f"{int(cell):{col_width}}" for cell in row_p) + " |"

        print(row_str)

