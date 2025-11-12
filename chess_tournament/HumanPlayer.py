#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran 

import chess

class HumanPlayer():
    def __init__(self):
        print("Moves can be entered using four characters. For example, d2d4 moves the piece "
              "at d2 to d4.")
        pass

    def choose_move(self, board, time_remaining=None):
        # Accept time_remaining parameter for compatibility
        moves = list(board.legal_moves)

        uci_move = None

        while uci_move not in moves:
            print("Please enter your move: ")
            human_move = input()

            try:
                uci_move = chess.Move.from_uci(human_move)
            except:
                uci_move = None

            if uci_move not in moves:
                print("  That is not a legal move!")

        print(uci_move in moves)
        return uci_move