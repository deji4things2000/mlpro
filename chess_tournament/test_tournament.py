#Class:** COSC 276 - Artificial Intelligence  
#Term:** Fall 2025  
#Assignment:** Chess Tournament 
#Student:** Adedeji Sunday Adediran 


import chess
from ChessGame import TournamentChessGame
from AlphaBetaAI import AlphaBetaAI
from MinimaxAI import MinimaxAI
from RandomAI import RandomAI
import multiprocessing
import time
import statistics

def run_tournament_match(player1, player2, time_limit=300, games=2):
    """Run multiple games between two players"""
    results = {
        'player1_wins': 0,
        'player2_wins': 0, 
        'draws': 0,
        'avg_move_time': [],
        'total_nodes': 0
    }
    
    for game_num in range(games):
        print(f"\n=== Game {game_num + 1}/{games} ===")
        
        # Alternate colors
        if game_num % 2 == 0:
            white, black = player1, player2
        else:
            white, black = player2, player1
            
        game = TournamentChessGame(white, black, time_limit)
        
        while not game.is_game_over():
            move = game.make_move()
            if move:
                print(f"Move {game.board.fullmove_number}: {move}")
        
        # Record result
        result = game.board.result()
        if result == "1-0":
            winner = "White"
        elif result == "0-1":
            winner = "Black" 
        else:
            winner = "Draw"
            
        print(f"Result: {result} ({winner})")
        
        # Update statistics
        if winner == "White":
            if game_num % 2 == 0:
                results['player1_wins'] += 1
            else:
                results['player2_wins'] += 1
        elif winner == "Black":
            if game_num % 2 == 0:
                results['player2_wins'] += 1
            else:
                results['player1_wins'] += 1
        else:
            results['draws'] += 1
            
        if game.move_times:
            results['avg_move_time'].extend(game.move_times)
    
    return results

def main():
    print("=== Chess AI Tournament ===")
    cores = multiprocessing.cpu_count()
    print(f"Running on {cores} CPU cores")
    
    # Create tournament participants
    participants = [
        ("RandomAI", RandomAI()),
        ("MinimaxAI", MinimaxAI(depth=3)),
        ("AlphaBetaAI", AlphaBetaAI(depth=3)),
        ("TournamentAI", AlphaBetaAI(depth=4))  # Your best AI
    ]
    
    time_limit = 300  # 5 minutes
    
    print(f"\nTime limit: {time_limit//60} minutes per player")
    print("Starting tournament...\n")
    
    # Round-robin tournament
    for i, (name1, player1) in enumerate(participants):
        for j, (name2, player2) in enumerate(participants):
            if i >= j:  # Avoid duplicate matches
                continue
                
            print(f"\n*** {name1} vs {name2} ***")
            results = run_tournament_match(player1, player2, time_limit, games=2)
            
            print(f"Results: {name1} {results['player1_wins']}-{results['player2_wins']}-{results['draws']} {name2}")
            if results['avg_move_time']:
                avg_time = statistics.mean(results['avg_move_time'])
                print(f"Average move time: {avg_time:.2f}s")

if __name__ == "__main__":
    main()