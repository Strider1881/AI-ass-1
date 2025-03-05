import numpy as np
import random
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from collections import Counter
import queue
import os

class WumpusNavigator:
    def __init__(self, dimension):
        self.dimension = dimension
        self.terrain = {}  # Dictionary storing cell information instead of array
        for i in range(dimension):
            for j in range(dimension):
                self.terrain[(i, j)] = {'category': 'Empty', 'sensory_input': []}

        self.explorer_location = (0, 0)
        self.explored_cells = set()
        self.exploration_frequency = Counter()  # Track cell visit frequency
        self.probability_model = None
        self.reasoner = None
        self.move_count = 0
        self.awareness = {}  # Explorer's understanding about the environment
        self.border_cells = set()  # Unexplored cells adjacent to explored ones
        self.position_history = []  # Track previous positions to identify cycles
        self.treasure_location = None
        self.confirmed_traps = set()  # Verified pit locations
        self.confirmed_monsters = set()  # Verified wumpus locations
        self.fatalities = Counter()  # Track locations where explorer perished

        # Create directory for solution visualization
        os.makedirs('wumpus_navigation', exist_ok=True)

        # Setup random environment and ensure treasure is accessible
        self.generate_random_environment()
        self.construct_probability_model()

        # Initialize starting state
        self.explored_cells.add(self.explorer_location)
        self.exploration_frequency[self.explorer_location] += 1
        self.position_history.append(self.explorer_location)

        # Begin perception cycle
        self.perceive_surroundings()
        self.update_awareness()
        self.refresh_border_cells()

        # Expedition status
        self.treasure_found = False
        self.expedition_complete = False
        self.move_limit = self.dimension * self.dimension * 3

        # Compute and visualize initial danger assessment
        self.danger_levels = self.assess_danger()
        self.create_danger_map(self.danger_levels)

    def generate_random_environment(self):
        """Generate a random Wumpus World with traps, monster, and treasure"""
        # Environment parameters
        trap_chance = 0.2  # 20% probability of trap in each cell

        # Reset the terrain
        for i in range(self.dimension):
            for j in range(self.dimension):
                self.terrain[(i, j)] = {'category': 'Empty', 'sensory_input': []}

        # Keep starting point safe
        protected_cells = {(0, 0)}

        # Distribute traps randomly
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (i, j) in protected_cells:
                    continue  # Skip protected areas

                if random.random() < trap_chance:
                    self.terrain[(i, j)]['category'] = 'Pit'

        # Position the monster (avoiding starting point)
        possible_monster_spots = [(i, j) for i in range(self.dimension) for j in range(self.dimension)
                                if (i, j) != (0, 0) and self.terrain[(i, j)]['category'] == 'Empty']
        if possible_monster_spots:
            monster_pos = random.choice(possible_monster_spots)
            self.terrain[monster_pos]['category'] = 'Wumpus'

        # Find accessible location for treasure
        while True:
            # Pick random location for treasure (not at start)
            possible_treasure_spots = [(i, j) for i in range(self.dimension) for j in range(self.dimension)
                                    if (i, j) != (0, 0) and self.terrain[(i, j)]['category'] == 'Empty']

            if not possible_treasure_spots:
                # No valid spots - regenerate environment
                print("Recreating environment - no valid treasure locations")
                return self.generate_random_environment()

            treasure_pos = random.choice(possible_treasure_spots)
            self.terrain[treasure_pos]['category'] = 'Gold'
            self.treasure_location = treasure_pos

            # Verify treasure accessibility
            if self.check_accessibility((0, 0), treasure_pos):
                break
            else:
                # Treasure unreachable - try again
                print("Treasure inaccessible, rebuilding environment...")
                self.terrain[treasure_pos]['category'] = 'Empty'

        # Add sensory cues
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.terrain[(i, j)]['category'] == 'Pit':
                    self.add_sensory_cue(i, j, 'Breeze')
                elif self.terrain[(i, j)]['category'] == 'Wumpus':
                    self.add_sensory_cue(i, j, 'Stench')

    def check_accessibility(self, origin, destination):
        """Determine if there's a safe path from origin to destination"""
        # Implement breadth-first search for path finding
        explored = set()
        search_queue = queue.Queue()
        search_queue.put(origin)
        explored.add(origin)

        while not search_queue.empty():
            current = search_queue.get()

            if current == destination:
                return True

            for adjacent in self.get_adjacent_cells(*current):
                if (adjacent not in explored and
                    self.terrain[adjacent]['category'] != 'Pit' and
                    self.terrain[adjacent]['category'] != 'Wumpus'):
                    explored.add(adjacent)
                    search_queue.put(adjacent)

        return False

    def add_sensory_cue(self, i, j, cue):
        for ni, nj in self.get_adjacent_cells(i, j):
            if self.terrain[(ni, nj)]['category'] == 'Empty' or self.terrain[(ni, nj)]['category'] == 'Gold':
                if cue not in self.terrain[(ni, nj)]['sensory_input']:
                    self.terrain[(ni, nj)]['sensory_input'].append(cue)

    def get_adjacent_cells(self, i, j):
        adjacent = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.dimension and 0 <= nj < self.dimension:
                adjacent.append((ni, nj))
        return adjacent

    def construct_probability_model(self):
        model = BayesianNetwork()

        # Create nodes for hazards
        for i in range(self.dimension):
            for j in range(self.dimension):
                model.add_node(f"Pit_{i}_{j}")
                model.add_node(f"Wumpus_{i}_{j}")

        # Create nodes for sensory inputs
        for i in range(self.dimension):
            for j in range(self.dimension):
                model.add_node(f"Breeze_{i}_{j}")
                model.add_node(f"Stench_{i}_{j}")

        # Define causal relationships
        for i in range(self.dimension):
            for j in range(self.dimension):
                # Pits cause breezes
                pit_node = f"Pit_{i}_{j}"
                for ni, nj in self.get_adjacent_cells(i, j):
                    breeze_node = f"Breeze_{ni}_{nj}"
                    if not model.has_edge(pit_node, breeze_node):
                        model.add_edge(pit_node, breeze_node)

                # Wumpus causes stenches
                wumpus_node = f"Wumpus_{i}_{j}"
                for ni, nj in self.get_adjacent_cells(i, j):
                    stench_node = f"Stench_{ni}_{nj}"
                    if not model.has_edge(wumpus_node, stench_node):
                        model.add_edge(wumpus_node, stench_node)

        # Define conditional probability distributions
        probability_tables = []

        # Prior probabilities
        for i in range(self.dimension):
            for j in range(self.dimension):
                # Starting position is guaranteed safe
                if (i, j) == (0, 0):
                    probability_tables.append(TabularCPD(f"Pit_{i}_{j}", 2, [[1], [0]]))  # No pit
                    probability_tables.append(TabularCPD(f"Wumpus_{i}_{j}", 2, [[1], [0]]))  # No wumpus
                else:
                    probability_tables.append(TabularCPD(f"Pit_{i}_{j}", 2, [[0.8], [0.2]]))  # 20% pit chance
                    probability_tables.append(TabularCPD(f"Wumpus_{i}_{j}", 2, [[0.95], [0.05]]))  # 5% wumpus chance

        # Breeze probability distributions
        for i in range(self.dimension):
            for j in range(self.dimension):
                breeze_node = f"Breeze_{i}_{j}"
                adjacent = self.get_adjacent_cells(i, j)

                if not adjacent:
                    probability_tables.append(TabularCPD(breeze_node, 2, [[1], [0]]))
                    continue

                # Breeze appears if ANY adjacent cell has a pit
                evidence = [f"Pit_{ni}_{nj}" for ni, nj in adjacent]
                evidence_cardinality = [2] * len(evidence)

                # Create conditional probability: P(Breeze=True | Any Pit=True) = 1.0
                combinations = 2 ** len(evidence)
                distribution = []

                for k in range(combinations):
                    binary = format(k, f'0{len(evidence)}b')
                    config = [int(bit) for bit in binary]

                    # If any pit exists nearby, there's a breeze
                    has_pit = any(v == 1 for v in config)
                    distribution.append([0 if has_pit else 1, 1 if has_pit else 0])

                probability_tables.append(TabularCPD(breeze_node, 2, np.array(distribution).T,
                                          evidence=evidence, evidence_card=evidence_cardinality))

        # Stench probability distributions
        for i in range(self.dimension):
            for j in range(self.dimension):
                stench_node = f"Stench_{i}_{j}"
                adjacent = self.get_adjacent_cells(i, j)

                if not adjacent:
                    probability_tables.append(TabularCPD(stench_node, 2, [[1], [0]]))
                    continue

                # Stench appears if ANY adjacent cell has a wumpus
                evidence = [f"Wumpus_{ni}_{nj}" for ni, nj in adjacent]
                evidence_cardinality = [2] * len(evidence)

                # Create conditional probability: P(Stench=True | Any Wumpus=True) = 1.0
                combinations = 2 ** len(evidence)
                distribution = []

                for k in range(combinations):
                    binary = format(k, f'0{len(evidence)}b')
                    config = [int(bit) for bit in binary]

                    # If any wumpus exists nearby, there's a stench
                    has_wumpus = any(v == 1 for v in config)
                    distribution.append([0 if has_wumpus else 1, 1 if has_wumpus else 0])

                probability_tables.append(TabularCPD(stench_node, 2, np.array(distribution).T,
                                          evidence=evidence, evidence_card=evidence_cardinality))

        for probability_table in probability_tables:
            model.add_cpds(probability_table)

        # Validate model
        try:
            assert model.check_model()
            self.probability_model = model
            self.reasoner = VariableElimination(model)
        except Exception as e:
            print(f"Probability model error: {e}")
            # Use fallback reasoning if model fails
            self.probability_model = None

    def update_awareness(self):
        """Update explorer's understanding based on current perceptions"""
        i, j = self.explorer_location

        # Record sensory inputs at current location
        sensory_input = self.terrain[(i, j)]['sensory_input']
        self.awareness[(i, j)] = {
            'breeze': 'Breeze' in sensory_input,
            'stench': 'Stench' in sensory_input,
            'safe': True  # Current location is safe
        }

        # Update border cells (potential moves)
        for ni, nj in self.get_adjacent_cells(i, j):
            if ((ni, nj) not in self.explored_cells and
                (ni, nj) not in self.confirmed_traps and
                (ni, nj) not in self.confirmed_monsters):
                self.border_cells.add((ni, nj))

            if (ni, nj) not in self.awareness:
                self.awareness[(ni, nj)] = {'breeze': False, 'stench': False, 'safe': None}

        # Apply logical deductions to mark cells as safe or dangerous
        self.deduce_safety()

    def deduce_safety(self):
        """Apply logical rules to identify safe or dangerous cells"""
        # If current cell has no sensory inputs, all adjacent cells are safe
        i, j = self.explorer_location
        sensory_input = self.terrain[(i, j)]['sensory_input']

        if not sensory_input:  # No sensory inputs detected
            for ni, nj in self.get_adjacent_cells(i, j):
                if (ni, nj) not in self.awareness:
                    self.awareness[(ni, nj)] = {'breeze': False, 'stench': False, 'safe': True}
                else:
                    self.awareness[(ni, nj)]['safe'] = True

    def refresh_border_cells(self):
        """Recalculate the set of unexplored cells adjacent to explored areas"""
        self.border_cells = set()
        for i, j in self.explored_cells:
            for ni, nj in self.get_adjacent_cells(i, j):
                if ((ni, nj) not in self.explored_cells and
                    (ni, nj) not in self.confirmed_traps and
                    (ni, nj) not in self.confirmed_monsters):
                    self.border_cells.add((ni, nj))

    def assess_danger(self):
        """Calculate danger assessment map using probability reasoning"""
        danger_map = np.zeros((self.dimension, self.dimension))

        # Mark explored cells as safe
        for i, j in self.explored_cells:
            danger_map[i, j] = 0

        # Mark confirmed dangerous cells
        for i, j in self.confirmed_traps:
            danger_map[i, j] = 1.0
        for i, j in self.confirmed_monsters:
            danger_map[i, j] = 1.0

        # Adjust danger based on past fatalities
        for pos, count in self.fatalities.items():
            if count > 0:
                i, j = pos
                danger_map[i, j] = min(1.0, 0.7 + count * 0.1)  # Increased danger with death count

        if self.probability_model and self.reasoner:
            # Apply Bayesian inference
            observations = {}

            # Add evidence from explored cells
            for (i, j), info in self.awareness.items():
                if (i, j) in self.explored_cells:
                    observations[f"Breeze_{i}_{j}"] = 1 if info['breeze'] else 0
                    observations[f"Stench_{i}_{j}"] = 1 if info['stench'] else 0

            # Add evidence for confirmed hazards
            for i, j in self.confirmed_traps:
                observations[f"Pit_{i}_{j}"] = 1
            for i, j in self.confirmed_monsters:
                observations[f"Wumpus_{i}_{j}"] = 1

            # Analyze unexplored cells
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if ((i, j) not in self.explored_cells and
                        (i, j) not in self.confirmed_traps and
                        (i, j) not in self.confirmed_monsters):
                        try:
                            # Calculate pit probability
                            pit_query = self.reasoner.query([f"Pit_{i}_{j}"], evidence=observations)
                            pit_probability = pit_query.values[1]

                            # Calculate wumpus probability
                            wumpus_query = self.reasoner.query([f"Wumpus_{i}_{j}"], evidence=observations)
                            wumpus_probability = wumpus_query.values[1]

                            # Combined danger (either pit or wumpus)
                            danger_map[i, j] = 1 - (1 - pit_probability) * (1 - wumpus_probability)
                        except Exception as e:
                            # Fallback to heuristic if probabilistic inference fails
                            danger_map[i, j] = self.estimate_danger(i, j)
        else:
            # Use direct heuristic if no probability model available
            for i in range(self.dimension):
                for j in range(self.dimension):
                    if ((i, j) not in self.explored_cells and
                        (i, j) not in self.confirmed_traps and
                        (i, j) not in self.confirmed_monsters):
                        danger_map[i, j] = self.estimate_danger(i, j)

        return danger_map

    def estimate_danger(self, i, j):
        """Calculate estimated danger for a cell based on available evidence"""
        if (i, j) in self.explored_cells:
            return 0.0  # Explored cells are confirmed safe

        if (i, j) in self.confirmed_traps or (i, j) in self.confirmed_monsters:
            return 1.0  # Confirmed hazards

        if (i, j) in self.fatalities:
            return 0.9  # Previous fatality location

        adjacent = self.get_adjacent_cells(i, j)
        explored_adjacent = [cell for cell in adjacent if cell in self.explored_cells]

        if not explored_adjacent:
            return 0.5  # No information about neighboring cells

        breeze_nearby = [cell for cell in explored_adjacent if self.awareness.get(cell, {}).get('breeze', False)]
        stench_nearby = [cell for cell in explored_adjacent if self.awareness.get(cell, {}).get('stench', False)]

        if breeze_nearby or stench_nearby:
            return 0.8  # High danger - adjacent to sensory cues

        # Look for safe adjacent cells (no breeze, no stench)
        safe_adjacent = [cell for cell in explored_adjacent
                       if not self.awareness.get(cell, {}).get('breeze', False) and
                          not self.awareness.get(cell, {}).get('stench', False)]

        if safe_adjacent:
            return 0.1  # Low danger - adjacent to confirmed safe cells

        return 0.5  # Default moderate danger

    def create_danger_map(self, danger_levels):
        """Create visualization of danger levels"""
        plt.figure(figsize=(8, 8))

        # Process danger matrix for visualization
        adjusted_danger = np.copy(danger_levels)
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (i, j) not in self.explored_cells:
                    pass  # Keep original danger value
                elif (i, j) == self.explorer_location:
                    adjusted_danger[i, j] = -1  # Mark explorer position
                else:
                    adjusted_danger[i, j] = -0.5  # Mark explored cells

        # Custom color scheme: explorer=blue, explored=green, danger gradient=white->yellow->red
        color_scheme = plt.get_cmap('RdYlGn_r').copy()
        color_scheme.set_under('green')  # Explored cells
        color_scheme.set_over('blue')    # Explorer position

        plt.imshow(adjusted_danger, cmap=color_scheme, interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label='Danger Level')

        # Grid overlay
        plt.grid(True, color='black', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-.5, self.dimension, 1), [])
        plt.yticks(np.arange(-.5, self.dimension, 1), [])

        # Cell annotations
        for i in range(self.dimension):
            for j in range(self.dimension):
                if (i, j) in self.explored_cells:
                    if (i, j) == self.explorer_location:
                        plt.text(j, i, 'E', ha='center', va='center', color='white', fontweight='bold')
                    else:
                        breeze = 'B' if self.awareness.get((i, j), {}).get('breeze', False) else ''
                        stench = 'S' if self.awareness.get((i, j), {}).get('stench', False) else ''
                        visit_count = self.exploration_frequency[(i, j)]
                        label = f'{breeze}{stench}{"" if visit_count <= 1 else visit_count}'
                        plt.text(j, i, label, ha='center', va='center', color='black')
                elif (i, j) in self.confirmed_traps:
                    plt.text(j, i, 'P!', ha='center', va='center', color='white')
                elif (i, j) in self.confirmed_monsters:
                    plt.text(j, i, 'W!', ha='center', va='center', color='white')
                else:
                    plt.text(j, i, f'{danger_levels[i, j]:.1f}', ha='center', va='center',
                             color='black' if danger_levels[i, j] < 0.7 else 'white')

        plt.title(f'Wumpus Navigator - Danger Assessment (Move {self.move_count})')
        plt.savefig(f'wumpus_navigation/danger_assessment_{self.move_count:03d}.png')
        plt.close()

    def detect_movement_cycle(self, next_position):
        """Detect if moving to next_position would create a repetitive cycle"""
        if len(self.position_history) < 6:
            return False

        # Check for simple back-and-forth pattern (A-B-A-B)
        if (next_position == self.position_history[-2] and
            self.explorer_location == self.position_history[-1] and
            next_position == self.position_history[-4] and
            self.explorer_location == self.position_history[-3]):
            return True

        # Check for repeated sequences in movement history
        history_repr = str(self.position_history[-6:] + [next_position])
        # If this cell appears too frequently in recent history
        if history_repr.count(str(next_position)) > 2:
            return True

        return False

    def requires_exploration_boost(self):
        """Determine if exploration needs randomization to break movement patterns"""
        # Check if recent movements are confined to few cells
        if len(self.position_history) < 10:
            return False

        # Count diversity in recent movements
        recent = self.position_history[-10:]
        unique_positions = len(set(recent))

        # Limited position diversity
        if unique_positions <= 3:
            return True

        # Check for excessive revisiting of same locations
        current_pos = self.explorer_location
        if self.exploration_frequency[current_pos] > 20:
            adjacent = self.get_adjacent_cells(*current_pos)
            if any(self.exploration_frequency[pos] > 20 for pos in adjacent):
                return True

        return False

    def select_optimal_move(self):
        """Determine safest adjacent move while avoiding repetitive patterns"""
        danger_levels = self.assess_danger()
        self.create_danger_map(danger_levels)

        i, j = self.explorer_location
        adjacent_cells = self.get_adjacent_cells(i, j)

        # If stuck in repetitive movement, increase exploration randomness
        if self.requires_exploration_boost():
            print("Movement cycle detected - increasing exploration randomness")
            # Find adjacent cells not known to be dangerous
            viable_options = [(ni, nj) for ni, nj in adjacent_cells
                             if (ni, nj) not in self.confirmed_traps and
                                (ni, nj) not in self.confirmed_monsters]

            if viable_options:
                # Weight by exploration frequency (favor less visited)
                weights = [1.0 / (1 + self.exploration_frequency[pos]) for pos in viable_options]
                total = sum(weights)
                norm_weights = [w/total for w in weights]

                # Apply weighted random selection
                return random.choices(viable_options, weights=norm_weights, k=1)[0]

        # Standard move evaluation
        move_ratings = {}
        for ni, nj in adjacent_cells:
            # Skip confirmed hazardous cells
            if (ni, nj) in self.confirmed_traps or (ni, nj) in self.confirmed_monsters:
                move_ratings[(ni, nj)] = -1.0  # Extremely low rating
                continue

            # Base rating inversely proportional to danger
            base_rating = 1 - danger_levels[ni, nj]

            # Penalize frequently visited cells
            revisit_penalty = min(0.7, self.exploration_frequency[(ni, nj)] * 0.15)

            # Penalty for moves creating cycles
            cycle_penalty = 0.8 if self.detect_movement_cycle((ni, nj)) else 0

            # Bonus for unexplored cells
            exploration_bonus = 0.4 if (ni, nj) not in self.explored_cells else 0

            # Major penalty for previous fatality locations
            fatality_penalty = min(1.0, self.fatalities[(ni, nj)] * 0.8)

            # Compute final rating
            move_ratings[(ni, nj)] = base_rating - revisit_penalty - cycle_penalty - fatality_penalty + exploration_bonus

        # Prioritize exploring frontier cells
        frontier_adjacent = [pos for pos in adjacent_cells if pos in self.border_cells]
        if frontier_adjacent and any(move_ratings[pos] > 0.2 for pos in frontier_adjacent):
            best_frontier = max(frontier_adjacent, key=lambda pos: move_ratings[pos])
            return best_frontier

        # If all moves seem risky, increase randomness
        if all(score < 0.2 for score in move_ratings.values()):
            if random.random() < 0.4:  # Increased randomness chance
                candidates = [pos for pos in adjacent_cells if move_ratings[pos] > -0.5]
                if candidates:
                    return random.choice(candidates)

        # Choose highest rated move
        return max(move_ratings.items(), key=lambda x: x[1])[0]

    def perceive_surroundings(self):
        """Gather sensory information at current location"""
        i, j = self.explorer_location
        cell = self.terrain[(i, j)]

        sensory_input = cell['sensory_input']
        print(f"Sensed at {self.explorer_location}: {sensory_input}")

        return {
            'breeze': 'Breeze' in sensory_input,
            'stench': 'Stench' in sensory_input,
            'glitter': cell['category'] == 'Gold'
        }

    def display_environment(self):
        """Display the complete Wumpus World environment"""
        print("\n===== WUMPUS NAVIGATOR =====")
        for j in range(self.dimension-1, -1, -1):  # Top-to-bottom display
            for i in range(self.dimension):
                if (i, j) == self.explorer_location:
                    print("E", end=" ")
                else:
                    cell = self.terrain[(i, j)]
                    if cell['category'] == 'Empty' and not cell['sensory_input']:
                        print(".", end=" ")
                    elif cell['category'] == 'Pit':
                        print("P", end=" ")
                    elif cell['category'] == 'Wumpus':
                        print("W", end=" ")
                    elif cell['category'] == 'Gold':
                        print("G", end=" ")
                    elif 'Breeze' in cell['sensory_input'] and 'Stench' in cell['sensory_input']:
                        print("BS", end=" ")
                    elif 'Breeze' in cell['sensory_input']:
                        print("B", end=" ")
                    elif 'Stench' in cell['sensory_input']:
                        print("S", end=" ")
                    else:
                        print("?", end=" ")
            print()
        print("===========================")
        print("E = Explorer, P = Pit, W = Wumpus, G = Gold")
        print("B = Breeze, S = Stench, BS = Breeze+Stench")
        print(f"Treasure located at {self.treasure_location}\n")

    def record_fatality(self, position, cause):
        """Document explorer fatality and learn from it"""
        self.fatalities[position] += 1

        # Update confirmed hazards
        if cause == 'Pit':
            self.confirmed_traps.add(position)
            print(f"Learned: {position} contains a pit")
        elif cause == 'Wumpus':
            self.confirmed_monsters.add(position)
            print(f"Learned: {position} contains a wumpus")

        # Update awareness
        if position in self.awareness:
            self.awareness[position]['safe'] = False

        # Remove from border cells
        if position in self.border_cells:
            self.border_cells.remove(position)

    def choose_random_move(self):
        """Select a random adjacent move"""
        i, j = self.explorer_location
        adjacent = self.get_adjacent_cells(i, j)

        # Filter out confirmed hazards
        viable_options = [(ni, nj) for ni, nj in adjacent
                        if (ni, nj) not in self.confirmed_traps and
                           (ni, nj) not in self.confirmed_monsters]

        # Default to any move if no viable options
        if not viable_options:
            viable_options = adjacent

        # Prefer unexplored cells with 70% probability
        unexplored = [pos for pos in viable_options if pos not in self.explored_cells]
        if unexplored and random.random() < 0.7:
            return random.choice(unexplored)

        # Otherwise weight by inverse of visit frequency
        if viable_options:
            weights = [1.0 / (1 + self.exploration_frequency[pos]) for pos in viable_options]
            total = sum(weights)
            if total > 0:
                norm_weights = [w/total for w in weights]
                return random.choices(viable_options, weights=norm_weights, k=1)[0]

        # Fallback to any random adjacent cell
        return random.choice(adjacent) if adjacent else self.explorer_location

    def execute_move(self, use_reasoning=True):
        """Execute a single move in the environment

        Args:
            use_reasoning (bool): If True, use probabilistic reasoning.
                                  If False, use random exploration.

        Returns:
            dict: Status information about the move outcome
        """
        if self.expedition_complete:
            return {
                'status': 'expedition_complete',
                'treasure_found': self.treasure_found,
                'move_count': self.move_count,
                'position': self.explorer_location,
                'message': "Expedition has already concluded"
            }

        if self.move_count >= self.move_limit:
            self.expedition_complete = True
            return {
                'status': 'move_limit_reached',
                'treasure_found': False,
                'move_count': self.move_count,
                'position': self.explorer_location,
                'message': f"Maximum moves ({self.move_limit}) reached. Expedition terminated."
            }

        print(f"Move {self.move_count}: Explorer at {self.explorer_location}")
        i, j = self.explorer_location

        # Check for treasure
        if self.terrain[(i, j)]['category'] == 'Gold':
            self.treasure_found = True
            self.expedition_complete = True
            print("Explorer found the TREASURE at", self.explorer_location)
            return {
                'status': 'treasure_found',
                'treasure_found': True,
                'move_count': self.move_count,
                'position': self.explorer_location,
                'message': f"Treasure discovered at {self.explorer_location}!"
            }

        # Determine next move based on strategy
        if use_reasoning:
            next_position = self.select_optimal_move()
            strategy = "reasoned"
        else:
            next_position = self.choose_random_move()
            strategy = "random"

        print(f"Moving to {next_position} ({strategy} strategy)")

        # Check if revisiting a cell
        if next_position in self.explored_cells:
            visit_count = self.exploration_frequency[next_position]
            print(f"Revisiting {next_position} (visit #{visit_count+1})")

        # Update position history for pattern detection
        self.position_history.append(next_position)
        if len(self.position_history) > 10:
            self.position_history.pop(0)

        # Update position
        self.explorer_location = next_position

        # Check for hazards
        if self.terrain[next_position]['category'] in ['Pit', 'Wumpus']:
            hazard_type = self.terrain[next_position]['category']
            print(f"Explorer perished at {next_position}! " +
                 ("Fell into a pit." if hazard_type == 'Pit' else "Killed by Wumpus."))

            # Record and learn from this fatality
            self.record_fatality(next_position, hazard_type)

            # Return to starting position
            self.explorer_location = (0, 0)
            self.position_history = [(0, 0)]  # Reset position history
            self.exploration_frequency[(0, 0)] += 1
            self.explored_cells.add((0, 0))

            # Reevaluate surroundings after restart
            perceptions = self.perceive_surroundings()
            self.update_awareness()
            self.refresh_border_cells()

            self.move_count += 1

            # Update danger assessment after move
            self.danger_levels = self.assess_danger()
            self.create_danger_map(self.danger_levels)

            return {
                'status': 'fatality',
                'hazard_type': hazard_type,
                'move_count': self.move_count,
                'position': (0, 0),  # Reset to origin
                'fatal_position': next_position,  # Fatality location
                'strategy': strategy,
                'message': f"Fatality at {next_position} due to {hazard_type}. Restarting from (0,0)."
            }

        # Update exploration records
        self.explored_cells.add(next_position)
        self.exploration_frequency[next_position] += 1
        perceptions = self.perceive_surroundings()
        self.update_awareness()
        self.refresh_border_cells()

        self.move_count += 1

        # Check for exploration completion
        if len(self.explored_cells) == self.dimension * self.dimension:
            self.expedition_complete = True
            print("Environment fully explored. Terminating expedition.")

        # Update danger assessment
        self.danger_levels = self.assess_danger()
        self.create_danger_map(self.danger_levels)

        return {
            'status': 'successful_move',
            'move_count': self.move_count,
            'position': next_position,
            'perceptions': perceptions,
            'strategy': strategy,
            'message': f"Moved to {next_position} using {strategy} exploration strategy"
        }

    def get_current_danger_assessment(self):
        """Retrieve current danger assessment without recalculation"""
        return self.danger_levels

    def get_expedition_status(self):
        """Provide complete expedition status"""
        return {
            'move_count': self.move_count,
            'position': self.explorer_location,
            'explored_cells': list(self.explored_cells),
            'treasure_found': self.treasure_found,
            'expedition_complete': self.expedition_complete,
            'treasure_location': self.treasure_location,
            'confirmed_traps': list(self.confirmed_traps),
            'confirmed_monsters': list(self.confirmed_monsters)
        }

    def complete_expedition(self, strategy='reasoned'):
        """Run expedition until completion using specified strategy

        Args:
            strategy: 'reasoned' for probabilistic reasoning, 'random' for random exploration,
                     or 'hybrid' for alternating between strategies
        """
        if strategy not in ['reasoned', 'random', 'hybrid']:
            raise ValueError("Strategy must be 'reasoned', 'random', or 'hybrid'")

        while not self.expedition_complete and self.move_count < self.move_limit:
            if strategy == 'reasoned':
                use_reasoning = True
            elif strategy == 'random':
                use_reasoning = False
            else:  # hybrid
                use_reasoning = (self.move_count % 2 == 0)  # Alternate strategies

            result = self.execute_move(use_reasoning)
            if result['status'] == 'treasure_found':
                print(f"Treasure found after {self.move_count} moves!")
                break

        return {
            'status': 'expedition_complete',
            'treasure_found': self.treasure_found,
            'total_moves': self.move_count,
            'explored_cells': len(self.explored_cells),
            'total_cells': self.dimension * self.dimension
        }

    def navigate(self):
        """Legacy method - replaced by complete_expedition"""
        print("Using the complete_expedition method instead for greater flexibility.")
        return self.complete_expedition(strategy='reasoned')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run Wumpus Navigator")
    parser.add_argument("--dimension", type=int, default=4, help="Dimension of the environment (N >= 4)")
    parser.add_argument("--strategy", choices=["reasoned", "random", "hybrid"], default="reasoned",
                        help="Select strategy: 'reasoned' for best move, 'random' for random move, or 'hybrid'")
    args = parser.parse_args()

    expedition = WumpusNavigator(args.dimension)
    expedition.display_environment()
    result = expedition.complete_expedition(strategy=args.strategy)
    print(result)

