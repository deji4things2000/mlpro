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
        self.time_limits = [time_limit, time_limit]
        self.time_used = [0.0, 0.0]
        self.move_times = []
        self.player_names = [player1.__class__.__name__, player2.__class__.__name__]
        
    def make_move(self):
        if self.is_game_over():
            return None
            
        player_index = 1 - int(self.board.turn)
        player = self.players[player_index]
        
        time_remaining = self.time_limits[player_index] - self.time_used[player_index]
        
        move_start = time.time()
        
        try:
            move = player.choose_move(self.board, time_remaining)
        except TypeError as e:
            if "unexpected keyword argument" in str(e) or "takes 2 positional" in str(e):
                print(f"Note: {player.__class__.__name__} doesn't support time_remaining parameter")
                move = player.choose_move(self.board)
            else:
                raise e
        
        move_end = time.time()
        move_time = move_end - move_start
        
        self.time_used[player_index] += move_time
        self.move_times.append(move_time)
        
        print(f"Move time: {move_time:.2f}s, Time remaining: {time_remaining - move_time:.2f}s")
        
        if move and move in self.board.legal_moves:
            self.board.push(move)
            return move
        else:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = legal_moves[0]
                print(f"Fallback to legal move: {move}")
                self.board.push(move)
                return move
            return None

    def is_game_over(self):
        return self.board.is_game_over() or self.is_timeout()

    def is_timeout(self):
        for i in range(2):
            if self.time_used[i] >= self.time_limits[i]:
                print(f"Player {i} ({self.player_names[i]}) ran out of time!")
                return True
        return False

    def get_game_result_with_time_tiebreak(self):
        """
        Returns game result with time-based tiebreak for draws
        Format: "1-0", "0-1", or "1/2-1/2" for actual draws with equal time
        """
        
        # Check for timeout first
        for i in range(2):
            if self.time_used[i] >= self.time_limits[i]:
                # The other player wins by timeout
                return "1-0" if i == 1 else "0-1"
        
        # Check standard game over conditions
        if not self.board.is_game_over():
            return None
        
        result = self.board.result()
        
        # If it's not a draw, return the normal result
        if result != "1/2-1/2":
            return result
        
        # For draws, apply time-based tiebreak
        white_time_remaining = self.get_time_remaining(chess.WHITE)
        black_time_remaining = self.get_time_remaining(chess.BLACK)
        
        print(f"Draw detected! Time remaining - White: {white_time_remaining:.1f}s, Black: {black_time_remaining:.1f}s")
        
        if white_time_remaining > black_time_remaining:
            print("White wins on time tiebreak!")
            return "1-0"
        elif black_time_remaining > white_time_remaining:
            print("Black wins on time tiebreak!")
            return "0-1"
        else:
            print("Perfect time equality - game remains a draw")
            return "1/2-1/2"

    def get_winner_description(self):
        """Get human-readable description of game result with time tiebreak"""
        result = self.get_game_result_with_time_tiebreak()
        
        if result == "1-0":
            white_time = self.get_time_remaining(chess.WHITE)
            black_time = self.get_time_remaining(chess.BLACK)
            if self.board.is_checkmate():
                return f"White wins by checkmate! (Time: {white_time:.1f}s vs {black_time:.1f}s)"
            else:
                return f"White wins on time tiebreak! ({white_time:.1f}s vs {black_time:.1f}s)"
        elif result == "0-1":
            white_time = self.get_time_remaining(chess.WHITE)
            black_time = self.get_time_remaining(chess.BLACK)
            if self.board.is_checkmate():
                return f"Black wins by checkmate! (Time: {black_time:.1f}s vs {white_time:.1f}s)"
            else:
                return f"Black wins on time tiebreak! ({black_time:.1f}s vs {white_time:.1f}s)"
        else:
            return "Game drawn by stalemate or insufficient material"

    def get_time_remaining(self, color):
        index = 0 if color == chess.WHITE else 1
        return max(0, self.time_limits[index] - self.time_used[index])

    def __str__(self):
        column_labels = "\n----------------\na b c d e f g h\n"
        board_str = str(self.board) + column_labels
        move_str = "White to move" if self.board.turn else "Black to move"
        
        time_str = (f"\nTime - White: {self.get_time_remaining(chess.WHITE):.1f}s, "
                   f"Black: {self.get_time_remaining(chess.BLACK):.1f}s")
        
        return board_str + "\n" + move_str + time_str + "\n"