# AI Assignment: CS F407 (Artificial Intelligence) Project Assignment - I

This repository contains the solution for the CS F407 AI project assignment at Birla Institute of Technology and Science, Pilani Hyderabad Campus (2nd semester 2025). The project comprises three main components:

1. **Hands-On Exercise 1: Tic-Tac-Toe Game-playing System**
   - Implements a Tic-Tac-Toe game where two LLM agents (e.g., ChatGPT, Claude, or similar) compete on a flexible board.
   - Supports playing interactively or simulating 500 games (Bernoulli trials) with outcome visualization (binomial distribution plot).
   - **Source code:** `tic_tac_toe.py`

2. **Hands-On Exercise 2: Wumpus World System**
   - Develops a Wumpus World of flexible size (NxN, where N ≥ 4) using a Bayesian Network (via pgmpy) to model hazard uncertainties.
   - Implements two movement strategies for the agent:
     - **Reasoned Move:** Uses Bayesian inference to choose the safest adjacent cell.
     - **Random Move:** Chooses a random adjacent cell.
   - Visualizes the pit/danger probabilities across the grid.
   - **Source code:** `wumpus_world_system.py` 

3. **Hands-On Exercise 3: Integrated System**
   - Merges the Tic-Tac-Toe and Wumpus World systems.
   - The outcome of the Tic-Tac-Toe game determines the agent's move strategy in the Wumpus World:
     - If LLM1 wins, the agent uses the reasoned (best) move.
     - If LLM2 wins (or in case of a draw), the agent uses a random move.
   - **Source code:** `main.py`

## Repository Structure


## Requirements

- Python 3.9 or higher
- Conda (recommended)
- Dependencies:
  - numpy
  - matplotlib
  - pgmpy
  - ipython
  - jupyter

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/CSF407_2025_2021B4A32924H.git
   ```
   
2. **Change Directory**
   ```bash
   cd CSF407_2025_2021B4A32924H
   ```

3. **Create and Activate the Conda Environment**
   ```bash
   conda env create -f config.yml
   conda activate CSF407_2025_2021B4A32924H
   ```


## Running the Program

   1. **Hands-On Exercise 1: Tic-Tac-Toe Game**
   - In simulation mode, the code saves the outcomes of the 500 games to a file such as Exercise1.json or Exercise1.txt  
   - Interactive Mode:  
     ```bash
     python src/tic_tac_toe.py --size 3 --mode LLMvsLLM
     ```
   - Simulation Mode (500 games):  
     ```bash
     python src/tic_tac_toe.py --simulate --size 3
     ```

   

2. **Hands-On Exercise 2: Wumpus World**
   - The code saves danger map images to the wumpus_navigation directory. These images are named as danger_assessment_XXX.png (where XXX is the move count)

   - Reasoned Strategy:  
     ```bash
     python src/wumpus_world_system.py --dimension 5 --strategy reasoned
     ```
   - Random Strategy:  
     ```bash
     python src/wumpus_world_system.py --dimension 5 --strategy random
     ```
   - Hybrid Strategy:  
     ```bash
     python src/wumpus_world_system.py --dimension 5 --strategy hybrid
     ```

   

3. **Hands-On Exercise 3: Integrated System**  
   ```bash
   python src/main.py --dimension 5 --ttt_size 3 --max_cycles 50



