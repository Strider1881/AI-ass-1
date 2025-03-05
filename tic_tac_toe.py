#!/usr/bin/env python3
import random, json
import matplotlib.pyplot as plt

def print_board(board):
    for row in board:
        print(" | ".join(row))
    print()

def check_win(board, symbol):
    n = len(board)
    # Check rows
    for row in board:
        if all(cell == symbol for cell in row):
            return True
    # Check columns
    for col in range(n):
        if all(board[row][col] == symbol for row in range(n)):
            return True
    # Check diagonal
    if all(board[i][i] == symbol for i in range(n)):
        return True
    # Check anti-diagonal
    if all(board[i][n-1-i] == symbol for i in range(n)):
        return True
    return False

def available_moves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == " ":
                moves.append((i, j))
    return moves

def llm_agent1(board, last_move, symbol):
    """
    LLM Agent 1: Uses a simple heuristic – prefers center, then corners,
    then a random available move.
    """
    n = len(board)
    center = (n // 2, n // 2)
    if center in available_moves(board):
        return center
    # Prefer corners
    corners = [(0,0), (0, n-1), (n-1, 0), (n-1, n-1)]
    available_corners = [move for move in corners if move in available_moves(board)]
    if available_corners:
        return random.choice(available_corners)
    # Otherwise, choose random move
    return random.choice(available_moves(board))

def llm_agent2(board, last_move, symbol):
    """
    LLM Agent 2: Chooses a random available move.
    """
    return random.choice(available_moves(board))

def human_move(board):
    while True:
        try:
            move = input("Enter your move as row,col (e.g., 0,1): ")
            i, j = map(int, move.strip().split(","))
            if (i, j) in available_moves(board):
                return (i, j)
            else:
                print("Invalid move. Try again.")
        except Exception:
            print("Invalid input. Please enter in row,col format.")

def play_game(board_size=3, mode='LLMvsLLM'):
    board = [[" " for _ in range(board_size)] for _ in range(board_size)]
    current_agent = 1
    symbols = {1: "X", 2: "O"}
    last_move = None
    while available_moves(board):
        if mode == 'LLMvsLLM':
            if current_agent == 1:
                move = llm_agent1(board, last_move, symbols[current_agent])
            else:
                move = llm_agent2(board, last_move, symbols[current_agent])
        elif mode == 'LLMvsHuman':
            if current_agent == 1:
                move = llm_agent1(board, last_move, symbols[current_agent])
            else:
                move = human_move(board)
        board[move[0]][move[1]] = symbols[current_agent]
        last_move = move
        # Uncomment the next line to display the board after each move:
        # print_board(board)
        if check_win(board, symbols[current_agent]):
            return current_agent
        current_agent = 2 if current_agent == 1 else 1
    return 0  # Draw

def simulate_games(num_games=500, board_size=3):
    outcomes = {"LLM1": 0, "LLM2": 0, "Draw": 0}
    results = []
    for _ in range(num_games):
        winner = play_game(board_size, mode='LLMvsLLM')
        if winner == 1:
            outcomes["LLM1"] += 1
            results.append("LLM1")
        elif winner == 2:
            outcomes["LLM2"] += 1
            results.append("LLM2")
        else:
            outcomes["Draw"] += 1
            results.append("Draw")
    with open("Exercise1.json", "w") as f:
        json.dump(results, f)
    # Plot outcomes as a bar chart
    plt.figure()
    plt.bar(["LLM1", "LLM2", "Draw"], [outcomes["LLM1"], outcomes["LLM2"], outcomes["Draw"]])
    plt.title(f"Outcomes over {num_games} games")
    plt.savefig("Exercise1.png")
    plt.close()
    return outcomes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tic Tac Toe between two LLM agents.")
    parser.add_argument("--mode", type=str, default="LLMvsLLM", choices=["LLMvsLLM", "LLMvsHuman"],
                        help="Game mode: LLMvsLLM or LLMvsHuman")
    parser.add_argument("--size", type=int, default=3, help="Board size (NxN)")
    parser.add_argument("--simulate", action="store_true", help="Simulate 500 games")
    args = parser.parse_args()
    if args.simulate:
        outcomes = simulate_games(num_games=500, board_size=args.size)
        print("Simulation outcomes:", outcomes)
    else:
        winner = play_game(board_size=args.size, mode=args.mode)
        if winner == 0:
            print("Game Draw!")
        else:
            print(f"Winner is {'LLM1' if winner == 1 else 'LLM2'}")
