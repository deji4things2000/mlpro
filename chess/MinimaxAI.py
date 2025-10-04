import chess
from math import inf

class MinimaxAI():
    def __init__(self, depth, iterative_deepening=False):
        self.depth = depth
        self.iterative_deepening = iterative_deepening
        self.nodes_visited = 0
        self.best_move = None
        
    def choose_move(self, board):
        # Check if game is over or no legal moves
        if board.is_game_over() or not list(board.legal_moves):
            return None
            
        self.nodes_visited = 0
        
        if self.iterative_deepening:
            return self.iterative_deepening_search(board)
        else:
            return self.minimax_search(board)
    
    def iterative_deepening_search(self, board):
        """Implement iterative deepening to find best move within time constraints"""
        self.best_move = None
        
        for current_depth in range(1, self.depth + 1):
            print(f"Searching depth {current_depth}...")
            value, move = self.minimax(board, current_depth, board.turn == chess.WHITE)
            if move:
                self.best_move = move
                print(f"Depth {current_depth}: Best move = {move}, Value = {value}")
        
        print(f"Minimax AI recommending move {self.best_move}")
        print(f"Nodes visited: {self.nodes_visited}")
        return self.best_move
    
    def minimax_search(self, board):
        """Standard fixed-depth minimax search"""
        value, move = self.minimax(board, self.depth, board.turn == chess.WHITE)
        print(f"Minimax AI recommending move {move}")
        print(f"Nodes visited: {self.nodes_visited}")
        return move
    
    def minimax(self, board, depth, maximizing_player):
        """Minimax algorithm with alpha-beta pruning"""
        self.nodes_visited += 1
        
        # Cutoff test
        if self.cutoff_test(board, depth):
            return self.evaluate_board(board), None
        
        moves = list(board.legal_moves)
        
        if maximizing_player:
            max_value = -inf
            best_move = moves[0] if moves else None
            
            for move in moves:
                board.push(move)
                value, _ = self.minimax(board, depth - 1, False)
                board.pop()
                
                if value > max_value:
                    max_value = value
                    best_move = move
            
            return max_value, best_move
        else:
            min_value = inf
            best_move = moves[0] if moves else None
            
            for move in moves:
                board.push(move)
                value, _ = self.minimax(board, depth - 1, True)
                board.pop()
                
                if value < min_value:
                    min_value = value
                    best_move = move
            
            return min_value, best_move
    
    def cutoff_test(self, board, depth):
        """Test if we should stop searching"""
        return depth == 0 or board.is_game_over()
    
    def evaluate_board(self, board):
        """Evaluation function using material counting"""
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -1000  # Black wins
            else:
                return 1000   # White wins
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0  # Draw
        
        # Material value evaluation
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # Kings don't have material value in this simple evaluation
        }
        
        score = 0
        
        # Sum material for white, subtract for black
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        return score