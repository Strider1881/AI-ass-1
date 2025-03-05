#!/usr/bin/env python3
import argparse
import time
import random

# Import the tic tac toe function from Exercise 1.
# Ensure that the file is named 'tic_tac_toe.py' and that it defines play_game.
from tic_tac_toe import play_game

# Import the WumpusNavigator class from Exercise 2.
# Ensure that the file is named 'task2.py' and that it defines WumpusNavigator.
from wumpus_world_system import WumpusNavigator

def main():
    parser = argparse.ArgumentParser(description="Integrated Simulation: Tic Tac Toe & Wumpus World")
    parser.add_argument("--dimension", type=int, default=5, help="Dimension of the Wumpus World (N >= 4)")
    parser.add_argument("--ttt_size", type=int, default=3, help="Board size for tic tac toe (typically 3)")
    parser.add_argument("--max_cycles", type=int, default=50, help="Maximum iterations of the integrated simulation")
    args = parser.parse_args()

    # Create the Wumpus World navigator instance.
    navigator = WumpusNavigator(args.dimension)

    cycle = 0
    print("=== Integrated Simulation Started ===")
    # Loop until expedition is complete or maximum cycles reached.
    while not navigator.expedition_complete and cycle < args.max_cycles:
        print(f"\n--- Cycle {cycle} ---")
        # Play a tic tac toe game to decide the move strategy.
        # This function should run an LLM vs LLM tic tac toe game and return:
        # 1 if LLM1 wins, 2 if LLM2 wins, or 0 if draw.
        tic_result = play_game(board_size=args.ttt_size, mode="LLMvsLLM")
        if tic_result == 1:
            # LLM1 wins: use the best (reasoned) move.
            use_reasoning = True
            print("Tic Tac Toe Outcome: LLM1 wins. Using reasoned move in Wumpus World.")
        elif tic_result == 2:
            # LLM2 wins: use random move.
            use_reasoning = False
            print("Tic Tac Toe Outcome: LLM2 wins. Using random move in Wumpus World.")
        else:
            # In case of draw, default to best move.
            use_reasoning = True
            print("Tic Tac Toe Outcome: Draw. Defaulting to reasoned move.")

        # Execute one move in the Wumpus World based on the selected strategy.
        result = navigator.execute_move(use_reasoning=use_reasoning)
        print(result['message'])
        # Optionally, display the current environment.
        navigator.display_environment()

        # Pause briefly to observe the cycle (optional)
        time.sleep(1)
        cycle += 1

    # Final expedition status.
    status = navigator.get_expedition_status()
    print("\n=== Expedition Complete ===")
    print(f"Total moves: {status['move_count']}")
    print(f"Treasure found: {status['treasure_found']}")
    print("Check the 'wumpus_navigation' directory for danger map visualizations.")

    explored = len(status['explored_cells'])
    total_cells = args.dimension * args.dimension
    print(f"Explored cells: {explored} out of {total_cells}")


if __name__ == "__main__":
    main()
