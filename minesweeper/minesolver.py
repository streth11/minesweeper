import numpy as np
from enum import Enum
from typing import List
from minegrids import MSGrid, MSGridState, MSGridElement, ContiguousGroup
from print_tools import PrintMode


class MineSolverState(Enum):
    START = 0
    FRONTIER_SIMPLE_SOLVE = 1
    FRONTIER_PATTERN_SOLVE = 2
    COMBINATION_SOLVE = 3
    RANDOM_GUESS = 4
    AWAITING_EVAL = 5
    COMPLETE = 6


class MineSolver:
    def __init__(self, grid: MSGrid, debug=False, seed=2468):
        self.grid = grid
        self.nMines = grid.nMines
        self.debug = debug
        if seed is None:
            self.rng = np.random.default_rng()
            #print(self.rng)
        else:
            self.rng = np.random.default_rng(seed=seed)
        self.stop_at_stalemate = False

        self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()
        self.prev_touched_cells = 0
        self.prev_mines_remaining = grid.nMines

        self.state = MineSolverState.START
        self.n_iterations = 0
        self.game_over = 0

        self.set_to_flag = set()
        self.set_to_reveal = set()
        self.cells_to_guess = []

    def solve(self, until_state=MineSolverState.COMPLETE, print_mode=PrintMode.NoPrint, stop_at_stalemate=False):
        if stop_at_stalemate:
            self.stop_at_stalemate = stop_at_stalemate
            until_state = min(MineSolverState.RANDOM_GUESS, until_state)

        while self.game_over == 0:
            self.game_over = self.solverIteration()
            if print_mode != PrintMode.NoPrint:
                self.grid.print(print_mode)
            if self.state == until_state:
                break
        return self.game_over, self.n_iterations

    def solverIteration(self):
        self.state = self.runSolverStateMachine()
        self.prev_touched_cells = self.iter_touched_cells
        self.prev_mines_remaining = self.iter_mines_remaining
        self.n_iterations += 1
        if self.grid.state == MSGridState.IN_PROGRESS:
            return 0
        elif self.grid.state == MSGridState.SOLVED:
            self.state = MineSolverState.COMPLETE
            return 1
        elif self.grid.state == MSGridState.FAILED:
            self.state = MineSolverState.COMPLETE
            return -1
        elif self.grid.state == MSGridState.STALEMATE:
            self.state = MineSolverState.COMPLETE
            return -1
        else:
            raise ValueError(f"Bad grid state: {self.grid.state}")

    def runSolverStateMachine(self) -> MineSolverState:
        # Handle START state
        if self.state == MineSolverState.START:
            if self.iter_touched_cells != 0:
                raise ValueError("Solver started with non-zero touched cells.")
            ret = self.startGuess()
            self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()
            if ret is None:
                raise ValueError("Start guess returned None.")
            return MineSolverState.FRONTIER_SIMPLE_SOLVE

        # Handle CLEANUP state
        if self.iter_mines_remaining == 0:
            self.runGridCleanup()
            return MineSolverState.AWAITING_EVAL

        # Map states to their corresponding methods
        state_methods = {
            MineSolverState.FRONTIER_SIMPLE_SOLVE: self.frontierSimpleSolve,
            MineSolverState.FRONTIER_PATTERN_SOLVE: self.frontierPatternSolve,
            MineSolverState.COMBINATION_SOLVE: self.combinationSolve,
            MineSolverState.RANDOM_GUESS: self.frontierRandomGuess,
        }

        # If in a frontier state, get frontier cells and call the appropriate method
        if self.state in state_methods:
            iter_frontier_cells = self.grid.getFrontierCells()
            ret = state_methods[self.state](iter_frontier_cells)
            if ret is None:
                raise ValueError(f"{self.state.name} returned None.")

            self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()

            # State transitions
            state_transitions = {
                MineSolverState.FRONTIER_SIMPLE_SOLVE: MineSolverState.FRONTIER_PATTERN_SOLVE,
                MineSolverState.FRONTIER_PATTERN_SOLVE: MineSolverState.COMBINATION_SOLVE,
                MineSolverState.COMBINATION_SOLVE: MineSolverState.RANDOM_GUESS,
                MineSolverState.RANDOM_GUESS: MineSolverState.AWAITING_EVAL,
            }

            if self.prev_touched_cells == self.iter_touched_cells:
                # Special case for RANDOM_GUESS: set grid state to STALEMATE
                if self.state == MineSolverState.RANDOM_GUESS:
                    self.grid.state = MSGridState.STALEMATE
                return state_transitions[self.state]
            return MineSolverState.FRONTIER_SIMPLE_SOLVE

        raise ValueError(f"Unknown state: {self.state}")


    def getCurrentGridState(self):
        return self.grid.numTouchedCells, self.grid.numMinesRemaining()

    def solverExecuteSets(self):
        for x, y in self.set_to_flag:
            self.grid.flagCell((x, y), debug=self.debug)
        for x, y in self.set_to_reveal:
            self.grid.revealCell((x, y), debug=self.debug)
        if self.debug:
            self.grid.print(PrintMode.RevealMines)

    def frontierSimpleSolve(self, frontier_cells: List[MSGridElement]):
        for cell in frontier_cells:
            unrevealed_neighbors = cell.untouchedNeighbors()
            num_flagged = cell.numFlaggedNeighbors()
            # If all mines are flagged, reveal the rest
            if cell.value == num_flagged:
                for neighbor in unrevealed_neighbors:
                    ret = self.grid.revealCell(neighbor.location, debug=self.debug)
                    if ret == 1 and self.debug:
                        self.grid.print(PrintMode.RevealMines)
                    elif ret == -1:
                        raise ValueError("Revealed a mine.")
            # If all unrevealed/unflagged neighbors must be mines, flag them
            elif cell.value - num_flagged == len(unrevealed_neighbors):
                for neighbor in unrevealed_neighbors:
                    self.grid.flagCell(neighbor.location, debug=self.debug)
                    if self.debug:
                        self.grid.print(PrintMode.RevealMines)
        return 1

    def frontierPatternSolve(self, frontier_cells: List[MSGridElement]):
        self.set_to_flag.clear()
        self.set_to_reveal.clear()
        for cell in frontier_cells:
            # 1-2-X pattern
            self.pattern12XSolve(cell)
            # 1-1-X pattern
            self.pattern11XSolve(cell)
        self.solverExecuteSets()
        return 1

    def pattern12XSolve(self, cell: MSGridElement):
        if cell.revealedReducedValue != 2:
            return
        if len(cell.untouchedNeighbors()) != 3:
            return

        # get target side
        unrevealed_cells = [
            1 if not n.isEdge and not n.touched else 0 for n in cell.cardinalSurround
        ]
        if sum(unrevealed_cells) != 1:
            return  # can only have 1 unrevealed cardinal side
        side_idx = unrevealed_cells.index(1) * 2

        # check its a row
        if cell.surround[side_idx + 1].touched or cell.surround[side_idx - 1].touched:
            return

        if cell.surround[side_idx + 2].revealedReducedValue == 1:
            self.set_to_flag.add(cell.surround[side_idx - 1].location)

        if cell.surround[side_idx - 2].revealedReducedValue == 1:
            self.set_to_flag.add(cell.surround[side_idx + 1].location)

    def pattern11XSolve(self, cell: MSGridElement):
        if cell.revealedReducedValue != 1:
            return
        if len(cell.untouchedNeighbors()) != 2:
            return

        # get target side
        unrevealed_cells = [
            1 if not n.isEdge and not n.touched else 0 for n in cell.cardinalSurround
        ]
        if sum(unrevealed_cells) != 1:
            return  # can only have 1 unrevealed cardinal side
        side_idx = unrevealed_cells.index(1) * 2

        # check there is exactly 1 unrevealed either side
        if not cell.surround[side_idx + 1].touched and not cell.surround[side_idx + 1].isEdge:
            direction = 1
        elif not cell.surround[side_idx - 1].touched and not cell.surround[side_idx - 1].isEdge:
            direction = -1
        else:
            return

        # the cell in that cardinal direction must have a value of 1
        if cell.surround[side_idx + 2 * direction].isEdge:
            return
        if cell.surround[side_idx + 2 * direction].revealedReducedValue != 1:
            return

        # check cell other side (should be open or edge)
        if not cell.surround[side_idx + -1 * direction].isEdge:
            if not cell.surround[side_idx + -1 * direction].touched:
                return

        cell_to_reveal = cell.surround[side_idx + 2 * direction].surround[
            side_idx + 1 * direction
        ]

        if cell_to_reveal.touched or cell_to_reveal.isEdge:
            return

        self.set_to_reveal.add(cell_to_reveal.location)

    def combinationSolve(self, frontier_cells: List[MSGridElement]):
        self.set_to_flag.clear()
        self.set_to_reveal.clear()
        groups = self.grid.establishContiguousCells(frontier_cells)
        # self.grid.print(PrintMode.RevealMines, show_groups=True)
        any_probability_calculated = False

        for g in groups:
            if g.max_prob_cell is None:
                if len(g) < 13:
                    # limit probability calculation for group size
                    # - if its this bad then just random guess
                    if len(self.set_to_flag) == 0 and len(self.set_to_reveal) == 0:
                        self.calculateGroupProbabilities(g,frontier_cells)
                        any_probability_calculated = True
            
        # no certainties in all groups, pick best option
        if len(self.set_to_flag) == 0 and len(self.set_to_reveal) == 0 and any_probability_calculated:
            
            groups_worst_case_nMines = sum([g.valid_comb_min_mines for g in groups])
            pred_ungrouped_mines = self.iter_mines_remaining - groups_worst_case_nMines
            if pred_ungrouped_mines < 0:
                raise ValueError("Mine estimation has failed")
            
            ungrouped_cells = self.grid.notInContiguousGroup()
            # check if this is not 0
            if len(ungrouped_cells) == 0 or pred_ungrouped_mines > 0:
                # from groups, calc least likely mine
                min_groups_prob = 1
                min_groups_prob_cell = None
                for g in groups:
                    if g.min_prob < min_groups_prob:
                        min_groups_prob = g.min_prob
                        min_groups_prob_cell = g.min_prob_cell
            
            if pred_ungrouped_mines == 0:
                # all cells outside groups should be free
                # return providing cells to make a random guess from
                if len(ungrouped_cells) > 0:
                    self.cells_to_guess = ungrouped_cells
                else:
                    self.set_to_reveal.add(min_groups_prob_cell.location)
            else:
                # calc probability an outside cell is a mine
                prob_ungrouped_mine = np.divide(len(ungrouped_cells),self.iter_mines_remaining)
                # if outside cell best guess random guess it, otherwise use group calc guess
                if prob_ungrouped_mine < min_groups_prob and len(ungrouped_cells) > 0:
                    # better probability to guess an ungrouped cell
                    self.cells_to_guess = ungrouped_cells
                else:
                    self.set_to_reveal.add(min_groups_prob_cell.location)
            
            # from groups, flag most likely mine
            #max_prob = 0
            #max_prob_cell = None
            #for g in groups:
                #if g.max_prob > max_prob:
                    #max_prob = g.max_prob
                    #max_prob_cell = g.max_prob_cell
            #self.set_to_flag.add(max_prob_cell.location)
                    
        self.solverExecuteSets()
        return 1
    
    def calculateGroupProbabilities(self, group:ContiguousGroup, frontier_cells: List[MSGridElement]):
        # work out probabilities for a mine on each cell
        n = len(group)
        n_combs = 2 ** n - 1
        combs = self.binary_mask_arr(n)
        valid_combs = np.empty((0, n), dtype=np.int64)
        for comb in combs:
            comb_nMines = np.sum(comb)
            if comb_nMines <= self.iter_mines_remaining:
                for idx,cell in enumerate(group):
                    mark = comb[idx]
                    cell.combination_mark = int(mark)
                is_valid = self.evaluateCombination(group, frontier_cells)
                if is_valid:
                    valid_combs = np.vstack((valid_combs,comb))
                    group.valid_comb_min_mines = np.min((group.valid_comb_min_mines,comb_nMines))
        n_valid_combs = len(valid_combs)
        if n_valid_combs == 0:
            return
        prob = np.divide(np.sum(valid_combs,axis=0),n_valid_combs)

        # assign certainties to be actioned
        group_has_certainties = False
        for idx,cell in enumerate(group):
            if prob[idx] == 0:
                self.set_to_reveal.add(cell.location)
                group_has_certainties = True
            if prob[idx] == 1:
                self.set_to_flag.add(cell.location)
                group_has_certainties = True
            # reset all group cells
            cell.combination_mark = None
                    
        # no certainties in group, store least likely mine cell
        if not group_has_certainties:
            #group_max_prob_idx = np.argmax(prob)
            group_min_prob_idx = np.argmin(prob)
            for idx,cell in enumerate(group):
                #if idx == group_max_prob_idx:
                    #g.max_prob = prob[group_max_prob_idx]
                    #g.max_prob_cell = cell
                if idx == group_min_prob_idx:
                    group.min_prob = prob[group_min_prob_idx]
                    group.min_prob_cell = cell
                    return

    def evaluateCombination(self, group:ContiguousGroup, frontier_cells: List[MSGridElement]) -> bool:
        for cell in frontier_cells:
            touching_group = False
            for n in cell.surround:
                if not n.touched and not n.isEdge:
                    if n in group:
                        touching_group = True
                        break
            
            if touching_group:
                numTouchingOtherGroup = sum(
                    1 for n in cell.surround if not n.isEdge and not n.touched and group.id != n.group_id
                )
                if numTouchingOtherGroup > 0:
                    x=1
                numMarkedCells = sum(
                    1 for n in cell.surround if not n.isEdge and n.combination_mark == 1
                )
                # current logic may ignore valid group combinations when there are frontier
                # cells touching another group
                conservativeValue = np.max((cell.revealedReducedValue - numTouchingOtherGroup, 1))
                conservativeNumMarked = numMarkedCells + numTouchingOtherGroup
                if conservativeNumMarked != conservativeValue:
                    return False
        return True

    def frontierRandomGuess(self, frontier_cells: List[MSGridElement]):
        if self.stop_at_stalemate:
            return 1
        
        local_neighbor_prob = [np.divide(cell.revealedReducedValue, cell.numUntouchedNeighbors()) for cell in frontier_cells]
        global_prob = np.divide(self.grid.numMinesRemaining(),self.grid.n-self.grid.numTouchedCells)

        # Find the minimum local probability
        min_local_prob = np.min(local_neighbor_prob)
        min_indices = [i for i, prob in enumerate(local_neighbor_prob) if prob == min_local_prob]

        if min_local_prob <= global_prob:
            # Randomly select one of the cells with the lowest local probability
            guess_candidate_list = frontier_cells[self.rng.choice(min_indices)].untouchedNeighbors()
        else:
            # Randomly select an untouched cell
            guess_candidate_list = self.grid.untouchedListFlattened()

        cell_to_reveal = guess_candidate_list[self.randomGuess(guess_candidate_list, return_only = True)]

        ret = self.grid.revealCell(cell_to_reveal.location, debug=self.debug)
        if ret == 1 and self.debug:
            self.grid.print(PrintMode.RevealMines)
        elif ret == -1:
            print("Random Guess Revealed a mine.")
        return ret

    def randomGuess(self, cells: List[MSGridElement], return_only = False):
        # select random cell
        rand_cell_idx = self.rng.integers(0,len(cells))
        if return_only:
            return rand_cell_idx
        # reveal
        ret = self.grid.revealCell(cells[rand_cell_idx].location, debug=self.debug)
        if ret == 1 and self.debug:
            self.grid.print(PrintMode.RevealMines)
        elif ret == -1:
            print("Random Guess Revealed a mine.")
        return ret
    
    def runGridCleanup(self):
        for cell in self.grid.untouchedListFlattened():
            self.grid.revealCell(cell.location)
        return 1

    def startGuess(self):
        ret = self.grid.revealCell((0, 0))
        while self.grid.state == MSGridState.FAILED:
            # If the first guess is a mine, try again
            self.grid.instantiateGrid()
            ret = self.grid.revealCell((0, 0), debug=self.debug)
        if ret != 1:
            raise ValueError("Start guess did not succeed.")
        return 1

    def binary_mask_arr(self,n) -> np.array:
        """
        Returns a 2D numpy array of shape (2**n, n), where each row is the binary representation
        of k (0 <= k < 2**n), least significant bit first.
        """
        return np.array([[(k >> i) & 1 for i in range(n)] for k in range(2 ** n)])


if __name__ == "__main__":
    seed = np.random.randint(99999)

    print(f"Seed = {seed}")

    grid = MSGrid(20, 9, nMines=28, seed=100)

    grid.instantiateGrid()
    solver = MineSolver(grid, debug=False, seed=2468)

    # grid.print(PrintMode.RevealAll)
    grid.print(PrintMode.RevealMines)
    # solver.solve(until_state=MineSolverState.COMBINATION_SOLVE)
    result,nIter = solver.solve()
    grid.print(PrintMode.Normal)

    if result == 1:
        print(f'Solve Successful in {nIter} iterations')
    elif result == -1:
        print(f'Solve Unsuccessful in {nIter} iterations')
