import chess


class ChessGame:
    def __init__(self, player1, player2):
        self.board = chess.Board()
        self.players = [player1, player2]

    def make_move(self):
        # Check if game is over before making a move
        if self.is_game_over():
            return None
            
        player = self.players[1 - int(self.board.turn)]
        move = player.choose_move(self.board)

        # Double-check that the move is legal (safety check)
        if move in self.board.legal_moves:
            self.board.push(move)
            return move
        else:
            # Fallback: choose first legal move if the AI returns an illegal move
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = legal_moves[0]
                self.board.push(move)
                return move
            return None

    def is_game_over(self):
        return self.board.is_game_over()

    def __str__(self):
        column_labels = "\n----------------\na b c d e f g h\n"
        board_str =  str(self.board) + column_labels

        # did you know python had a ternary conditional operator?
        move_str = "White to move" if self.board.turn else "Black to move"

        return board_str + "\n" + move_str + "\n"