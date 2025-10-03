# brew install pyqt
from PyQt5 import QtGui, QtSvg
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
import sys
import chess, chess.svg
from RandomAI import RandomAI
from MinimaxAI import MinimaxAI
from ChessGame import ChessGame
from HumanPlayer import HumanPlayer

import random


class ChessGui:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

        self.game = ChessGame(player1, player2)

        self.app = QApplication(sys.argv)
        self.svgWidget = QtSvg.QSvgWidget()
        self.svgWidget.setGeometry(50, 50, 400, 400)
        self.svgWidget.show()


    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.make_move)
        self.timer.start(10)

        self.display_board()

    def display_board(self):
        svgboard = chess.svg.board(self.game.board)

        svgbytes = QByteArray()
        svgbytes.append(svgboard)
        self.svgWidget.load(svgbytes)


    def make_move(self):

        if self.game.is_game_over():
            self.timer.stop()
            self.show_game_result()
            return


        print("making move, white turn " + str(self.game.board.turn))

        success = self.game.make_move()

        if not success:
            self.timer.stop()
            self.show_game_result()
            return
        
        self.display_board()
    
    def show_game_result(self):
        result = self.game.board.result()
        if self.game.board.is_checkmate():
            winner = "Black" if self.game.board.turn else "White"
            message = f"Checkmate! {winner} wins!"
        elif self.game.board.is_stalemate():
            message = "Stalemate!"
        elif self.game.board.is_insufficient_material():
            message = "Draw due to insufficient material!"
        else:
            message = f"Game over! Result: {result}"
            
        msg = QMessageBox()
        msg.setWindowTitle("Game Over")
        msg.setText(message)
        msg.exec_()




if __name__ == "__main__":

    random.seed(1)

    #player_ronda = RandomAI()

    # to do: gui does not work well with HumanPlayer, due to input() use on stdin conflict
    #   with event loop.

    player1 = RandomAI()
    player2 = RandomAI()

    game = ChessGame(player1, player2)
    gui = ChessGui(player1, player2)

    gui.start()

    sys.exit(gui.app.exec_())
