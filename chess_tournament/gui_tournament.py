#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran 


from PyQt5 import QtGui, QtSvg
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
import sys
import chess, chess.svg
from RandomAI import RandomAI
from MinimaxAI import MinimaxAI
from AlphaBetaAI import AlphaBetaAI
from ChessGame import TournamentChessGame
from HumanPlayer import HumanPlayer
import multiprocessing

class TournamentChessGui:
    def __init__(self, player1, player2, time_limit=300):
        self.player1 = player1
        self.player2 = player2
        self.game = TournamentChessGame(player1, player2, time_limit)
        self.game_over = False
        self.time_limit = time_limit

        self.app = QApplication(sys.argv)
        self.window = QWidget()
        self.window.setWindowTitle("Tournament Chess AI")
        self.window.setGeometry(100, 100, 600, 700)
        
        # Create main layout
        layout = QVBoxLayout()
        
        # Time display
        time_layout = QHBoxLayout()
        self.white_time_label = QLabel(f"White: {time_limit}s")
        self.black_time_label = QLabel(f"Black: {time_limit}s")
        self.white_progress = QProgressBar()
        self.black_progress = QProgressBar()
        
        self.white_progress.setMaximum(time_limit)
        self.black_progress.setMaximum(time_limit)
        self.white_progress.setValue(time_limit)
        self.black_progress.setValue(time_limit)
        
        time_layout.addWidget(self.white_time_label)
        time_layout.addWidget(self.white_progress)
        time_layout.addWidget(self.black_time_label)
        time_layout.addWidget(self.black_progress)
        
        # Chess board
        self.svgWidget = QtSvg.QSvgWidget()
        self.svgWidget.setFixedSize(500, 500)
        
        # Info display
        self.info_label = QLabel("Game starting...")
        self.info_label.setAlignment(Qt.AlignCenter)
        
        layout.addLayout(time_layout)
        layout.addWidget(self.svgWidget)
        layout.addWidget(self.info_label)
        
        self.window.setLayout(layout)
        self.window.show()
        
        # Display CPU info
        cores = multiprocessing.cpu_count()
        print(f"Detected {cores} CPU cores - AI will use all available cores")

    def update_time_display(self):
        """Update the time display widgets"""
        white_remaining = self.game.get_time_remaining(chess.WHITE)
        black_remaining = self.game.get_time_remaining(chess.BLACK)
        
        self.white_time_label.setText(f"White: {white_remaining:.1f}s")
        self.black_time_label.setText(f"Black: {black_remaining:.1f}s")
        
        self.white_progress.setValue(int(white_remaining))
        self.black_progress.setValue(int(black_remaining))
        
        # Change color when time is low
        if white_remaining < 30:
            self.white_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        if black_remaining < 30:
            self.black_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")

    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.make_move)
        self.timer.start(50)  # 50ms for responsive timing

        self.display_board()

    def display_board(self):
        svgboard = chess.svg.board(self.game.board)
        svgbytes = QByteArray()
        svgbytes.append(svgboard)
        self.svgWidget.load(svgbytes)
        self.update_time_display()
    
    def get_game_result_description(self, result):
        """Enhanced result description with timeout and time tiebreak detection"""
        if self.game.is_timeout():
            if self.game.time_used[0] >= self.time_limit:
                black_time = self.game.get_time_remaining(chess.BLACK)
                return f"Black wins on time! ({black_time:.1f}s remaining)"
            else:
                white_time = self.game.get_time_remaining(chess.WHITE)
                return f"White wins on time! ({white_time:.1f}s remaining)"
        
        # Use the new time-based result system
        actual_result = self.game.get_game_result_with_time_tiebreak()
        
        if actual_result == "1-0":
            white_time = self.game.get_time_remaining(chess.WHITE)
            black_time = self.game.get_time_remaining(chess.BLACK)
            if self.game.board.is_checkmate():
                return f"White wins by checkmate! (Time: {white_time:.1f}s vs {black_time:.1f}s)"
            else:
                return f"White wins on time tiebreak! ({white_time:.1f}s remaining vs {black_time:.1f}s)"
        elif actual_result == "0-1":
            white_time = self.game.get_time_remaining(chess.WHITE)
            black_time = self.game.get_time_remaining(chess.BLACK)
            if self.game.board.is_checkmate():
                return f"Black wins by checkmate! (Time: {black_time:.1f}s vs {white_time:.1f}s)"
            else:
                return f"Black wins on time tiebreak! ({black_time:.1f}s remaining vs {white_time:.1f}s)"
        else:
            if self.game.board.is_stalemate():
                return "Game drawn by stalemate (equal time)"
            elif self.game.board.is_insufficient_material():
                return "Game drawn by insufficient material (equal time)"
            elif self.game.board.is_seventyfive_moves():
                return "Game drawn by 75-move rule (equal time)"
            else:
                return "Game drawn (equal time remaining)"

    def make_move(self):
        if self.game_over:
            return
            
        if self.game.is_game_over():
            self.game_over = True
            result_description = self.game.get_winner_description()
            
            # Add detailed time usage info
            time_info = (f"\n\nTime Usage:\n"
                        f"White: {self.game.time_used[0]:.1f}s used, "
                        f"{self.game.get_time_remaining(chess.WHITE):.1f}s remaining\n"
                        f"Black: {self.game.time_used[1]:.1f}s used, "
                        f"{self.game.get_time_remaining(chess.BLACK):.1f}s remaining")
            
            QMessageBox.information(self.window, "Game Over", result_description + time_info)
            self.timer.stop()
            return

        self.info_label.setText(f"Thinking... (Cores: {multiprocessing.cpu_count()})")
        self.game.make_move()
        self.display_board()
        
        move_count = self.game.board.fullmove_number
        self.info_label.setText(f"Move {move_count} - Cores: {multiprocessing.cpu_count()}")

        if self.game.is_game_over():
            self.game_over = True
            result_description = self.game.get_winner_description()
            
            time_info = (f"\n\nTime Usage:\n"
                        f"White: {self.game.time_used[0]:.1f}s used, "
                        f"{self.game.get_time_remaining(chess.WHITE):.1f}s remaining\n"
                        f"Black: {self.game.time_used[1]:.1f}s used, "
                        f"{self.game.get_time_remaining(chess.BLACK):.1f}s remaining")
            
            QMessageBox.information(self.window, "Game Over", result_description + time_info)
            self.timer.stop()


if __name__ == "__main__":
    print("=== Tournament Chess AI ===")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    
    print("\nSelect AI for White:")
    print("1. RandomAI")
    print("2. MinimaxAI (depth 3)")
    print("3. AlphaBetaAI (parallel)")
    
    white_choice = input("Choice (1-3): ").strip()
    
    print("\nSelect AI for Black:")
    print("1. RandomAI") 
    print("2. MinimaxAI (depth 3)")
    print("3. AlphaBetaAI (parallel)")
    
    black_choice = input("Choice (1-3): ").strip()
    
    # Time limit selection
    time_choice = input("\nTime limit in minutes (default 5): ").strip()
    time_limit = 300  # 5 minutes default
    if time_choice:
        time_limit = int(time_choice) * 60
    
    # White player
    if white_choice == "1":
        player1 = RandomAI()
    elif white_choice == "2":
        player1 = MinimaxAI(depth=3)
    else:
        player1 = AlphaBetaAI(depth=4, time_limit=time_limit)
    
    # Black player  
    if black_choice == "1":
        player2 = RandomAI()
    elif black_choice == "2":
        player2 = MinimaxAI(depth=3)
    else:
        player2 = AlphaBetaAI(depth=4, time_limit=time_limit)

    gui = TournamentChessGui(player1, player2, time_limit)
    gui.start()
    sys.exit(gui.app.exec_())