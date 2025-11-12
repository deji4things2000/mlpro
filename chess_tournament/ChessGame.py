#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran 


import chess
import time

class TournamentChessGame:
    def __init__(self, player1, player2, time_limit=300):
        self.board = chess.Board()
        self.players = [player1, player2]
        self.time_limits = [time_limit, time_limit]  # 5 minutes per player
        self.time_used = [0.0, 0.0]
        self.move_times = []
        
    def make_move(self):
        if self.is_game_over():
            return None
            
        player_index = 1 - int(self.board.turn)
        player = self.players[player_index]
        
        # Calculate time remaining for this player
        time_remaining = self.time_limits[player_index] - self.time_used[player_index]
        
        # Start timing
        move_start = time.time()
        
        # Get move from player with time information
        if hasattr(player, 'choose_move'):
            move = player.choose_move(self.board, time_remaining)
        else:
            move = player.choose_move(self.board)
        
        # End timing
        move_end = time.time()
        move_time = move_end - move_start
        
        # Update time used
        self.time_used[player_index] += move_time
        self.move_times.append(move_time)
        
        print(f"Move time: {move_time:.2f}s, Time remaining: {time_remaining - move_time:.2f}s")
        
        # Validate move
        if move in self.board.legal_moves:
            self.board.push(move)
            return move
        else:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = legal_moves[0]
                self.board.push(move)
                return move
            return None

    def is_game_over(self):
        return self.board.is_game_over() or self.is_timeout()

    def is_timeout(self):
        """Check if either player has run out of time"""
        for i in range(2):
            if self.time_used[i] >= self.time_limits[i]:
                return True
        return False

    def get_time_remaining(self, color):
        """Get remaining time for a color"""
        index = 0 if color == chess.WHITE else 1
        return max(0, self.time_limits[index] - self.time_used[index])

    def __str__(self):
        column_labels = "\n----------------\na b c d e f g h\n"
        board_str = str(self.board) + column_labels
        move_str = "White to move" if self.board.turn else "Black to move"
        
        # Add time information
        time_str = (f"\nTime - White: {self.get_time_remaining(chess.WHITE):.1f}s, "
                   f"Black: {self.get_time_remaining(chess.BLACK):.1f}s")
        
        return board_str + "\n" + move_str + time_str + "\n"