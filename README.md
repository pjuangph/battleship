# Using Transformers to play Battleship
Install requirements `pip install -r requirements.txt` 

To train the transformer and generate the boards modify and run `python battleship_train.py` This will create the data folder and store `source` and `test` data.
Modify the `epochs` in `battleship_train.py` to change the number of epochs.

# Running inference
There are 2 codes to run inference there are 2 functions to choose from. I reccomend using auto_game first to get a feel for how the transformer is performing.

Inference Example
```python
auto_game(n_games=1,train=False)  # This one runs a single game and prints out the board for each guess
# auto_game(n_games=1000, train=True)
# ai_helper()
```
