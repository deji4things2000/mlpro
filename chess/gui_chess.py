# brew install pyqt
from PyQt5 import QtGui, QtSvg
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
import sys
import chess, chess.svg
from RandomAI import RandomAI
from MinimaxAI import MinimaxAI
from AlphaBetaAI import AlphaBetaAI
from ChessGame import ChessGame
from HumanPlayer import HumanPlayer

import random


class ChessGui:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.game = ChessGame(player1, player2)
        self.game_over = False

        self.app = QApplication(sys.argv)
        self.svgWidget = QtSvg.QSvgWidget()
        self.svgWidget.setGeometry(50, 50, 400, 400)
        self.svgWidget.show()

    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.make_move)
        self.timer.start(100)  # Increased delay to 100ms for better performance

        self.display_board()

    def display_board(self):
        svgboard = chess.svg.board(self.game.board)
        svgbytes = QByteArray()
        svgbytes.append(svgboard)
        self.svgWidget.load(svgbytes)

    def get_game_result_description(self, result):
        """Convert chess result code to human-readable description"""
        if result == "1-0":
            return "White wins by checkmate!"
        elif result == "0-1":
            return "Black wins by checkmate!"
        elif result == "1/2-1/2":
            if self.game.board.is_stalemate():
                return "Game drawn by stalemate!"
            elif self.game.board.is_insufficient_material():
                return "Game drawn by insufficient material!"
            elif self.game.board.is_seventyfive_moves():
                return "Game drawn by 75-move rule!"
            else:
                return "Game drawn!"
        else:
            return f"Game finished: {result}"

    def make_move(self):
        if self.game_over:
            return
            
        if self.game.is_game_over():
            self.game_over = True
            result = self.game.board.result()
            result_description = self.get_game_result_description(result)
            QMessageBox.information(self.svgWidget, "Game Over", result_description)
            self.timer.stop()
            return

        print(f"making move, white turn: {self.game.board.turn}")
        self.game.make_move()
        self.display_board()

        # Check again after move
        if self.game.is_game_over():
            self.game_over = True
            result = self.game.board.result()
            result_description = self.get_game_result_description(result)
            QMessageBox.information(self.svgWidget, "Game Over", result_description)
            self.timer.stop()

if __name__ == "__main__":
    random.seed(1)

    # Test different AI combinations in GUI
    print("Select AI for White:")
    print("1. RandomAI")
    print("2. MinimaxAI (depth 2)")
    print("3. AlphaBetaAI (depth 3)")
    
    white_choice = input("Choice (1-3): ").strip()
    
    print("Select AI for Black:")
    print("1. RandomAI") 
    print("2. MinimaxAI (depth 2)")
    print("3. AlphaBetaAI (depth 3)")
    
    black_choice = input("Choice (1-3): ").strip()
    
    # White player
    if white_choice == "1":
        player1 = RandomAI()
    elif white_choice == "2":
        player1 = MinimaxAI(depth=2)
    else:
        player1 = AlphaBetaAI(depth=3)
    
    # Black player  
    if black_choice == "1":
        player2 = RandomAI()
    elif black_choice == "2":
        player2 = MinimaxAI(depth=2)
    else:
        player2 = AlphaBetaAI(depth=3)

    gui = ChessGui(player1, player2)
    gui.start()
    sys.exit(gui.app.exec_())