#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran 

import chess
from math import inf
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

class MinimaxAI():
    def __init__(self, depth, iterative_deepening=False, time_limit=300):
        self.depth = depth
        self.iterative_deepening = iterative_deepening
        self.nodes_visited = 0
        self.best_move = None
        self.time_limit = time_limit
        self.start_time = None
        self.max_cores = mp.cpu_count()
        
    def choose_move(self, board, time_remaining=None):
        if board.is_game_over() or not list(board.legal_moves):
            return None
            
        self.nodes_visited = 0
        self.start_time = time.time()
        
        actual_time_remaining = time_remaining if time_remaining is not None else self.time_limit
        
        if self.iterative_deepening:
            return self.iterative_deepening_search(board, actual_time_remaining)
        else:
            return self.minimax_search(board, actual_time_remaining)
    
    def iterative_deepening_search(self, board, time_remaining):
        self.best_move = None
        current_depth = 1
        move_time_allocation = self.calculate_move_time(time_remaining)
        
        while current_depth <= self.depth:
            elapsed = time.time() - self.start_time
            if elapsed > move_time_allocation * 0.9:
                break
                
            print(f"Searching depth {current_depth}...")
            value, move = self.minimax(board, current_depth, board.turn == chess.WHITE, time_remaining - elapsed)
            
            if move:
                self.best_move = move
                print(f"Depth {current_depth}: Best move = {move}, Value = {value}")
            
            current_depth += 1
        
        print(f"Minimax AI recommending move {self.best_move}")
        print(f"Nodes visited: {self.nodes_visited}")
        return self.best_move
    
    def minimax_search(self, board, time_remaining):
        move_time_allocation = self.calculate_move_time(time_remaining)
        adaptive_depth = self.calculate_adaptive_depth(time_remaining, move_time_allocation)
        
        value, move = self.minimax(board, adaptive_depth, board.turn == chess.WHITE, move_time_allocation)
        print(f"Minimax AI recommending move {move}")
        print(f"Nodes visited: {self.nodes_visited}")
        return move
    
    def minimax(self, board, depth, maximizing_player, time_remaining):
        self.nodes_visited += 1
        
        # Check time every 100 nodes
        if self.nodes_visited % 100 == 0:
            elapsed = time.time() - self.start_time
            if elapsed > time_remaining:
                return self.evaluate_board(board), None
        
        if self.cutoff_test(board, depth):
            return self.evaluate_board(board), None
        
        moves = list(board.legal_moves)
        
        if maximizing_player:
            max_value = -inf
            best_move = moves[0] if moves else None
            
            for move in moves:
                board.push(move)
                value, _ = self.minimax(board, depth - 1, False, time_remaining)
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
                value, _ = self.minimax(board, depth - 1, True, time_remaining)
                board.pop()
                
                if value < min_value:
                    min_value = value
                    best_move = move
            
            return min_value, best_move
    
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
            if board.turn == chess.WHITE:
                return -1000
            else:
                return 1000
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return 0
        
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