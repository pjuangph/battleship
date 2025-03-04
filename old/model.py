import tensorflow as tf

# Define model
class BattleshipNN(tf.keras.Model):
    def __init__(self,board_height:int=10,board_width:int=10):
        """_summary_

        Args:
            board_height (int, optional): board height. Defaults to 10.
            board_width (int, optional): board width. Defaults to 10.
        """
        super(BattleshipNN, self).__init__()
        board_size = board_height * board_width
        hidden_units = board_size
        output_units = board_size
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=(board_size,hidden_units), stddev=0.1))   # Weights
        self.b1 = tf.Variable(tf.zeros([1, hidden_units]))  # Biases            
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=(hidden_units, board_size), stddev=0.1)) # Weights
        self.b2 = tf.Variable(tf.zeros([1, output_units]))  # Biases

    def call(self, input_positions):
        h1 = tf.tanh(tf.matmul(input_positions, self.W1) + self.b1)
        logits = tf.matmul(h1, self.W2) + self.b2
        probabilities = tf.nn.softmax(logits)
        return logits, probabilities
