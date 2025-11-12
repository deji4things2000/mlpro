#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran 

import chess
from math import inf
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

class AlphaBetaAI():
    def __init__(self, depth, iterative_deepening=False, time_limit=300):
        self.depth = depth
        self.iterative_deepening = iterative_deepening
        self.nodes_visited = 0
        self.best_move = None
        self.time_limit = time_limit
        self.start_time = None
        self.max_cores = mp.cpu_count()
        
    def choose_move(self, board, time_remaining=None):
        # Handle both with and without time_remaining parameter
        if board.is_game_over() or not list(board.legal_moves):
            return None
            
        self.nodes_visited = 0
        self.start_time = time.time()
        
        # Use provided time or default to time_limit
        actual_time_remaining = time_remaining if time_remaining is not None else self.time_limit
        
        if self.iterative_deepening:
            return self.iterative_deepening_search(board, actual_time_remaining)
        else:
            return self.alphabeta_search(board, actual_time_remaining)
    
    def iterative_deepening_search(self, board, time_remaining):
        self.best_move = None
        current_depth = 1
        move_time_allocation = self.calculate_move_time(time_remaining)
        
        while current_depth <= self.depth:
            elapsed = time.time() - self.start_time
            if elapsed > move_time_allocation * 0.9:
                break
                
            print(f"AlphaBeta searching depth {current_depth} with {self.max_cores} cores...")
            value, move = self.alphabeta(board, current_depth, -inf, inf, board.turn == chess.WHITE, time_remaining - elapsed)
            
            if move:
                self.best_move = move
                print(f"Depth {current_depth}: Best move = {move}, Value = {value}")
            
            current_depth += 1
        
        if self.best_move:
            print(f"AlphaBeta AI recommending move {self.best_move}")
            print(f"Nodes visited: {self.nodes_visited}")
        return self.best_move
    
    def alphabeta_search(self, board, time_remaining):
        move_time_allocation = self.calculate_move_time(time_remaining)
        adaptive_depth = self.calculate_adaptive_depth(time_remaining, move_time_allocation)
        
        value, move = self.alphabeta(board, adaptive_depth, -inf, inf, board.turn == chess.WHITE, move_time_allocation)
        if move:
            print(f"AlphaBeta AI recommending move {move}")
            print(f"Nodes visited: {self.nodes_visited}")
        return move
    
    def alphabeta(self, board, depth, alpha, beta, maximizing_player, time_remaining):
        self.nodes_visited += 1
        
        # Check time every 100 nodes
        if self.nodes_visited % 100 == 0:
            elapsed = time.time() - self.start_time
            if elapsed > time_remaining:
                return self.evaluate_board(board), None
        
        if self.cutoff_test(board, depth):
            return self.evaluate_board(board), None
        
        moves = list(board.legal_moves)
        if not moves:
            return self.evaluate_board(board), None
        
        moves = self.order_moves(board, moves)
        
        if maximizing_player:
            max_value = -inf
            best_move = moves[0]
            
            for move in moves:
                board.push(move)
                value, _ = self.alphabeta(board, depth - 1, alpha, beta, False, time_remaining)
                board.pop()
                
                if value > max_value:
                    max_value = value
                    best_move = move
                
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break
            
            return max_value, best_move
        else:
            min_value = inf
            best_move = moves[0]
            
            for move in moves:
                board.push(move)
                value, _ = self.alphabeta(board, depth - 1, alpha, beta, True, time_remaining)
                board.pop()
                
                if value < min_value:
                    min_value = value
                    best_move = move
                
                beta = min(beta, min_value)
                if beta <= alpha:
                    break
            
            return min_value, best_move
    
    def order_moves(self, board, moves):
        scored_moves = []
        
        for move in moves:
            score = 0
            
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                moving_piece = board.piece_at(move.from_square)
                
                if captured_piece and moving_piece:
                    piece_values = {
                        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
                    }
                    victim_value = piece_values.get(captured_piece.piece_type, 0)
                    aggressor_value = piece_values.get(moving_piece.piece_type, 0)
                    score += 100 + victim_value - aggressor_value * 0.1
            
            board.push(move)
            if board.is_check():
                score += 50
            board.pop()
            
            center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
            if move.to_square in center_squares and board.fullmove_number < 10:
                score += 20
            
            if board.fullmove_number < 8:
                moving_piece = board.piece_at(move.from_square)
                if (moving_piece and moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP] and
                    chess.square_file(move.from_square) in [0, 7]):
                    score += 15
            
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for score, move in scored_moves]
    
    def calculate_move_time(self, time_remaining):
        safe_time = max(1, time_remaining - 60)
        base_time = min(30, safe_time * 0.1)
        return base_time
    
    def calculate_adaptive_depth(self, time_remaining, move_time):
        if time_remaining > 240:
            return self.depth + 1
        elif time_remaining > 120:
            return self.depth
        elif time_remaining > 30:
            return max(2, self.depth - 1)
        else:
            return 2
    
    def cutoff_test(self, board, depth):
        return depth == 0 or board.is_game_over()
    
    def evaluate_board(self, board):
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0
        
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
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