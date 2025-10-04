import chess
from math import inf

class AlphaBetaAI():
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
            return self.alphabeta_search(board)
    
    def iterative_deepening_search(self, board):
        """Implement iterative deepening with alpha-beta pruning"""
        self.best_move = None
        
        for current_depth in range(1, self.depth + 1):
            print(f"Searching depth {current_depth}...")
            value, move = self.alphabeta(board, current_depth, -inf, inf, board.turn == chess.WHITE)
            if move:
                self.best_move = move
                print(f"Depth {current_depth}: Best move = {move}, Value = {value}")
        
        print(f"AlphaBeta AI recommending move {self.best_move}")
        print(f"Nodes visited: {self.nodes_visited}")
        return self.best_move
    
    def alphabeta_search(self, board):
        """Standard fixed-depth alpha-beta search"""
        value, move = self.alphabeta(board, self.depth, -inf, inf, board.turn == chess.WHITE)
        print(f"AlphaBeta AI recommending move {move}")
        print(f"Nodes visited: {self.nodes_visited}")
        return move
    
    def alphabeta(self, board, depth, alpha, beta, maximizing_player):
        """Alpha-beta pruning algorithm"""
        self.nodes_visited += 1
        
        # Cutoff test
        if self.cutoff_test(board, depth):
            return self.evaluate_board(board), None
        
        moves = list(board.legal_moves)
        if not moves:
            return self.evaluate_board(board), None
        
        # Move ordering - sort captures first for better pruning
        moves = self.order_moves(board, moves)
        
        if maximizing_player:
            max_value = -inf
            best_move = moves[0]
            
            for move in moves:
                board.push(move)
                value, _ = self.alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if value > max_value:
                    max_value = value
                    best_move = move
                
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_value, best_move
        else:
            min_value = inf
            best_move = moves[0]
            
            for move in moves:
                board.push(move)
                value, _ = self.alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if value < min_value:
                    min_value = value
                    best_move = move
                
                beta = min(beta, min_value)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_value, best_move
    
    def order_moves(self, board, moves):
        """Simple move ordering - prioritize captures and checks"""
        scored_moves = []
        
        for move in moves:
            score = 0
            
            # Prioritize captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    # Higher score for capturing more valuable pieces
                    piece_values = {
                        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9
                    }
                    score += 10 + piece_values.get(captured_piece.piece_type, 0)
            
            # Prioritize checks
            board.push(move)
            if board.is_check():
                score += 5
            board.pop()
            
            scored_moves.append((score, move))
        
        # Sort by score in descending order
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for score, move in scored_moves]
    
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