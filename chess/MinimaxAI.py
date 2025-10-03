import chess
from math import inf


class MinimaxAI():
    def __init__(self, depth):
        self.depth = depth
        self.nodes_visited = 0

    def choose_move(self, board):
        self.nodes_visited = 0
        best_move = None
        best_value = -inf
        
        for move in board.legal_moves:
            board.push(move)
            value = self.minimax_min(board, self.depth - 1)
            board.pop()
            
            if value > best_value:
                best_value = value
                best_move = move
        
        print(f"Minimax AI recommending move {best_move} (evaluation: {best_value:.2f}, nodes visited: {self.nodes_visited})")
        return best_move

    def minimax_max(self, board, depth):
        self.nodes_visited += 1
        
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        max_eval = -inf
        for move in board.legal_moves:
            board.push(move)
            eval = self.minimax_min(board, depth - 1)
            board.pop()
            max_eval = max(max_eval, eval)
        
        return max_eval

    def minimax_min(self, board, depth):
        self.nodes_visited += 1
        
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        min_eval = inf
        for move in board.legal_moves:
            board.push(move)
            eval = self.minimax_max(board, depth - 1)
            board.pop()
            min_eval = min(min_eval, eval)
        
        return min_eval

    def evaluate_board(self, board):
        if board.is_checkmate():
            if board.turn:  # Black's turn means white delivered checkmate
                return 1000
            else:  # White's turn means black delivered checkmate
                return -1000
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0
        
        # Simple material evaluation
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score