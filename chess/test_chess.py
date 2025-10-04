# pip3 install python-chess

import chess
from RandomAI import RandomAI
from HumanPlayer import HumanPlayer
from MinimaxAI import MinimaxAI
from AlphaBetaAI import AlphaBetaAI
from ChessGame import ChessGame

import sys

def get_game_result_description(result):
    """Convert chess result code to human-readable description"""
    if result == "1-0":
        return "White wins by checkmate"
    elif result == "0-1":
        return "Black wins by checkmate"
    elif result == "1/2-1/2":
        # Determine the type of draw
        return "Game drawn"
    else:
        return f"Game finished: {result}"

# Test different AI configurations
def test_minimax():
    print("Testing Minimax AI vs Random AI")
    player1 = MinimaxAI(depth=3)
    player2 = RandomAI()
    game = ChessGame(player1, player2)
    
    while not game.is_game_over():
        print(game)
        game.make_move()
    
    print("Game over!")
    result = game.board.result()
    print(get_game_result_description(result))

def test_alphabeta():
    print("Testing AlphaBeta AI vs Random AI")
    player1 = AlphaBetaAI(depth=3)
    player2 = RandomAI()
    game = ChessGame(player1, player2)
    
    while not game.is_game_over():
        print(game)
        game.make_move()
    
    print("Game over!")
    result = game.board.result()
    print(get_game_result_description(result))

def test_iterative_deepening():
    print("Testing Iterative Deepening AlphaBeta AI")
    player1 = AlphaBetaAI(depth=4, iterative_deepening=True)
    player2 = RandomAI()
    game = ChessGame(player1, player2)
    
    while not game.is_game_over():
        print(game)
        game.make_move()
    
    print("Game over!")
    result = game.board.result()
    print(get_game_result_description(result))

def human_vs_ai():
    print("Human vs AlphaBeta AI")
    player1 = HumanPlayer()
    player2 = AlphaBetaAI(depth=3)
    game = ChessGame(player1, player2)
    
    while not game.is_game_over():
        print(game)
        game.make_move()
    
    print("Game over!")
    result = game.board.result()
    print(get_game_result_description(result))

def compare_minimax_alphabeta():
    """Compare that Minimax and AlphaBeta return the same moves"""
    print("Comparing Minimax vs AlphaBeta (should be identical moves)")
    
    # Test with a few positions
    test_positions = [
        chess.Board(),
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"),
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1")
    ]
    
    for i, board in enumerate(test_positions):
        print(f"\nTest position {i + 1}:")
        print(board)
        
        minimax_ai = MinimaxAI(depth=3)
        alphabeta_ai = AlphaBetaAI(depth=3)
        
        minimax_move = minimax_ai.choose_move(board.copy())
        alphabeta_move = alphabeta_ai.choose_move(board.copy())
        
        print(f"Minimax nodes: {minimax_ai.nodes_visited}, AlphaBeta nodes: {alphabeta_ai.nodes_visited}")
        print(f"Minimax move: {minimax_move}, AlphaBeta move: {alphabeta_move}")
        print(f"Moves match: {minimax_move == alphabeta_move}")
        print(f"AlphaBeta efficiency: {minimax_ai.nodes_visited / alphabeta_ai.nodes_visited:.2f}x fewer nodes")

if __name__ == "__main__":
    print("Chess AI Test Suite")
    print("1. Minimax vs Random")
    print("2. AlphaBeta vs Random") 
    print("3. Iterative Deepening AlphaBeta vs Random")
    print("4. Human vs AlphaBeta")
    print("5. Compare Minimax vs AlphaBeta")
    
    choice = input("Select test (1-5): ").strip()
    
    if choice == "1":
        test_minimax()
    elif choice == "2":
        test_alphabeta()
    elif choice == "3":
        test_iterative_deepening()
    elif choice == "4":
        human_vs_ai()
    elif choice == "5":
        compare_minimax_alphabeta()
    else:
        # Default: human vs random
        player1 = HumanPlayer()
        player2 = RandomAI()
        game = ChessGame(player1, player2)
        
        while not game.is_game_over():
            print(game)
            game.make_move()
        
        print("Game over!")
        result = game.board.result()
        print(get_game_result_description(result))