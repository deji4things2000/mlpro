#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran


import chess
from ChessGame import TournamentChessGame
from HumanPlayer import HumanPlayer
from RandomAI import RandomAI
from MinimaxAI import MinimaxAI
from AlphaBetaAI import AlphaBetaAI
import time

def main():
    print("Welcome to Tournament Chess!")
    print("This version includes time controls and parallel processing.")
    print(f"Available CPU cores will be used for optimal performance.\n")
    
    # Player selection
    print("Select White player:")
    print("1. Human")
    print("2. RandomAI")
    print("3. MinimaxAI")
    print("4. AlphaBetaAI (Parallel)")
    
    white_choice = input("Your choice (1-4): ").strip()
    
    print("\nSelect Black player:")
    print("1. Human")
    print("2. RandomAI") 
    print("3. MinimaxAI")
    print("4. AlphaBetaAI (Parallel)")
    
    black_choice = input("Your choice (1-4): ").strip()
    
    # Time limit selection
    time_choice = input("\nTime limit in minutes (default 5): ").strip()
    time_limit = 300  # 5 minutes default
    if time_choice:
        time_limit = int(time_choice) * 60
    
    # Create players
    if white_choice == "1":
        player1 = HumanPlayer()
    elif white_choice == "2":
        player1 = RandomAI()
    elif white_choice == "3":
        depth = input("Minimax depth (default 3): ").strip()
        depth = int(depth) if depth else 3
        player1 = MinimaxAI(depth=depth, iterative_deepening=True, time_limit=time_limit)
    else:
        depth = input("AlphaBeta depth (default 3): ").strip()
        depth = int(depth) if depth else 3
        player1 = AlphaBetaAI(depth=depth, iterative_deepening=True, time_limit=time_limit)
    
    if black_choice == "1":
        player2 = HumanPlayer()
    elif black_choice == "2":
        player2 = RandomAI()
    elif black_choice == "3":
        depth = input("Minimax depth (default 3): ").strip()
        depth = int(depth) if depth else 3
        player2 = MinimaxAI(depth=depth, iterative_deepening=True, time_limit=time_limit)
    else:
        depth = input("AlphaBeta depth (default 3): ").strip()
        depth = int(depth) if depth else 3
        player2 = AlphaBetaAI(depth=depth, iterative_deepening=True, time_limit=time_limit)
    
    # Create and run game
    game = TournamentChessGame(player1, player2, time_limit)
    
    print(f"\nStarting game with {time_limit//60} minute time control.")
    print("Draws will be decided by time remaining.\n")
    
    move_count = 0
    game_start = time.time()
    
    while not game.is_game_over():
        print(game)
        print("-" * 50)
        
        move = game.make_move()
        move_count += 1
        
        if move:
            print(f"Move {move_count}: {move}")
        else:
            print("No legal move made!")
            break
        
        # Show thinking time occasionally
        if move_count % 5 == 0:
            elapsed = time.time() - game_start
            print(f"\nProgress: {move_count} moves in {elapsed:.1f}s")
            print(f"White remaining: {game.get_time_remaining(chess.WHITE):.1f}s")
            print(f"Black remaining: {game.get_time_remaining(chess.BLACK):.1f}s\n")
    
    # Game over
    print("\n" + "="*60)
    print("GAME OVER")
    print("="*60)
    
    result = game.get_game_result_with_time_tiebreak()
    description = game.get_winner_description()
    
    print(description)
    print(f"\nFinal position:")
    print(game)
    
    # Detailed time statistics
    print("\nTime Statistics:")
    print(f"White: {game.time_used[0]:.1f}s used, {game.get_time_remaining(chess.WHITE):.1f}s remaining")
    print(f"Black: {game.time_used[1]:.1f}s used, {game.get_time_remaining(chess.BLACK):.1f}s remaining")
    print(f"Total game time: {time.time() - game_start:.1f}s")
    print(f"Total moves: {move_count}")
    
    if game.move_times:
        avg_move_time = sum(game.move_times) / len(game.move_times)
        print(f"Average move time: {avg_move_time:.2f}s")

def test_quick_tournament():
    """Run a quick tournament between AIs for testing"""
    print("\n" + "="*60)
    print("QUICK TOURNAMENT TEST")
    print("="*60)
    
    players = [
        ("RandomAI", RandomAI()),
        ("Minimax Depth 2", MinimaxAI(depth=2, time_limit=60)),
        ("AlphaBeta Depth 2", AlphaBetaAI(depth=2, time_limit=60)),
        ("AlphaBeta Depth 3", AlphaBetaAI(depth=3, time_limit=60)),
    ]
    
    results = {}
    
    for i, (name1, player1) in enumerate(players):
        for j, (name2, player2) in enumerate(players):
            if i >= j:  # Avoid self-play and duplicates
                continue
                
            print(f"\n*** {name1} vs {name2} ***")
            game = TournamentChessGame(player1, player2, time_limit=30)  # 30 seconds for testing
            
            move_count = 0
            max_moves = 20  # Limit game length
            
            while not game.is_game_over() and move_count < max_moves:
                move = game.make_move()
                if move:
                    move_count += 1
                else:
                    break
            
            result = game.get_game_result_with_time_tiebreak()
            print(f"Result: {result} in {move_count} moves")
            
            # Track results
            match_key = f"{name1}_vs_{name2}"
            results[match_key] = {
                'result': result,
                'moves': move_count,
                'white_time': game.get_time_remaining(chess.WHITE),
                'black_time': game.get_time_remaining(chess.BLACK),
            }
    
    print("\n" + "="*60)
    print("TOURNAMENT RESULTS")
    print("="*60)
    
    for match, info in results.items():
        print(f"{match}: {info['result']} ({info['moves']} moves)")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Interactive Game")
    print("2. Quick Tournament Test")
    
    mode = input("Your choice (1-2): ").strip()
    
    if mode == "2":
        test_quick_tournament()
    else:
        main()