#import chess
import random
from time import sleep

class RandomAI():
    def __init__(self):
        pass

    def choose_move(self, board):
        moves = list(board.legal_moves)
        if not moves:  # Check if there are no legal moves
            return None
        move = random.choice(moves)
        sleep(0.5)   # Reduced sleep time for GUI responsiveness
        print("Random AI recommending move " + str(move))
        return move