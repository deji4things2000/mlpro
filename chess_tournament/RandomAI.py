#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran 

import random
from time import sleep

class RandomAI():
    def __init__(self):
        pass

    def choose_move(self, board, time_remaining=None):
        # Accept time_remaining parameter for compatibility, but ignore it
        moves = list(board.legal_moves)
        if not moves:
            return None
        move = random.choice(moves)
        sleep(0.1)  # Reduced sleep time for tournament play
        print("Random AI recommending move " + str(move))
        return move