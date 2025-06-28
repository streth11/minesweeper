import numpy as np
from enum import Enum
from typing import List
from minegrids import MSGrid, MSGridState, MSGridElement, ContiguousGroup
from print_tools import PrintMode


class MineSolverState(Enum):
    START = 0
    FRONTIER_SIMPLE_SOLVE = 1
    FRONTIER_PATTERN_SOLVE = 2
    RANDOM_GUESS = 3
    CLEANUP = 4
    COMPLETE = 5


class MineSolver:
    def __init__(self, grid: MSGrid, debug=False):
        self.grid = grid
        self.nMines = grid.nMines
        self.debug = debug

        self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()
        self.prev_touched_cells = 0
        self.prev_mines_remaining = grid.nMines

        self.state = MineSolverState.START
        self.n_iterations = 0

        self.set_to_flag = set()
        self.set_to_reveal = set()

    def solve(self, until_state=MineSolverState.COMPLETE, print_mode=PrintMode.NoPrint):
        game_over = 0
        while game_over == 0:
            game_over = self.solverIteration()
            if print_mode != PrintMode.NoPrint:
                self.grid.print(print_mode)
            if self.state == until_state:
                break
        return game_over, self.n_iterations

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
        if self.state == MineSolverState.START:
            if self.iter_touched_cells != 0:
                raise ValueError("Solver started with non-zero touched cells.")
            ret = self.startGuess()

            self.iter_touched_cells, self.iter_mines_remaining = (
                self.getCurrentGridState()
            )
            if ret is None:
                raise ValueError("Start guess returned None.")
            return MineSolverState.FRONTIER_SIMPLE_SOLVE
        elif self.iter_mines_remaining == 0:
            self.runGridCleanup()
            return MineSolverState.CLEANUP
        elif (
            self.state == MineSolverState.FRONTIER_SIMPLE_SOLVE
            or self.state == MineSolverState.FRONTIER_PATTERN_SOLVE
            or self.state == MineSolverState.RANDOM_GUESS
        ):
            iter_frontier_cells = self.grid.getFrontierCells()

        if self.state == MineSolverState.FRONTIER_SIMPLE_SOLVE:
            # execute frontier simple solve
            ret = self.frontierSimpleSolve(iter_frontier_cells)
            if ret is None:
                raise ValueError("Frontier simple solve returned None.")

            # see what has changed
            self.iter_touched_cells, self.iter_mines_remaining = (
                self.getCurrentGridState()
            )
            # update state based on touched cells
            if self.prev_touched_cells == self.iter_touched_cells:
                return MineSolverState.FRONTIER_PATTERN_SOLVE
            return MineSolverState.FRONTIER_SIMPLE_SOLVE

        elif self.state == MineSolverState.FRONTIER_PATTERN_SOLVE:
            # execute frontier pattern solve
            ret = self.frontierPatternSolve(iter_frontier_cells)
            if ret is None:
                raise ValueError("Frontier pattern solve returned None.")

            # see what has changed
            self.iter_touched_cells, self.iter_mines_remaining = (
                self.getCurrentGridState()
            )
            # update state based on touched cells
            if self.prev_touched_cells == self.iter_touched_cells:
                return MineSolverState.RANDOM_GUESS
            return MineSolverState.FRONTIER_SIMPLE_SOLVE

        elif self.state == MineSolverState.RANDOM_GUESS:
            # execute random guess
            ret = self.randomGuess()
            if ret is None:
                raise ValueError("Random guess returned None.")

            self.iter_touched_cells, self.iter_mines_remaining = (
                self.getCurrentGridState()
            )

            # update state based on touched cells
            if self.prev_touched_cells == self.iter_touched_cells:
                self.grid.state = MSGridState.STALEMATE
                return MineSolverState.COMPLETE
            return MineSolverState.FRONTIER_SIMPLE_SOLVE

        else:
            raise ValueError(f"Unknown state: {self.state}")

    def getCurrentGridState(self):
        return self.grid.numTouchedCells, self.grid.numMinesRemaining()

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
        self.patternSolveExecuteSets()
        return 1

    def patternSolveExecuteSets(self):
        for x, y in self.set_to_flag:
            self.grid.flagCell((x, y), debug=self.debug)

        for x, y in self.set_to_reveal:
            self.grid.revealCell((x, y), debug=self.debug)

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
        if not cell.surround[side_idx + 1].touched:
            direction = 1
        elif not cell.surround[side_idx - 1].touched:
            direction = -1
        else:
            return

        # the cell in that cardinal direction must have a value of 1
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
        
        max_prob = 0
        for g in groups:
            if g.max_prob_cell is not None:
                continue
            # work out probabilities for a mine on each cell
            n_combs = 2 ** n - 1
            combs = self.binary_mask_arr(len(g))
            valid_combs = np.array()
            for comb in combs:
                if np.sum(comb) <= self.grid.numMinesRemaining():
                    for idx,cell in enumerate(g):
                        mark = comb[idx]
                        cell.combination_mark = mark
                    is_valid = self.evaluateCombination(g, frontier_cells)
                    if is_valid:
                        valid_combs = np.vstack((validCombs,comb))
            prob = np.divide(np.sum(valid_combs,axis=1),n_combs)
            
            # assign certainties to be actioned
            group_has_certainties = False
            for idx,cell in enumerate(g):
                if prob[idx] == 0:
                    self.set_to_reveal.add(cell.location)
                    group_has_certanties = True
                if prob[idx] == 1:
                    self.set_to_flag.add(cell.location)
                    group_has_certanties = True
                    
            # no certanties in group, store best probability cell
            if not group_has_certanties:
                group_max_prob_idx = np.argmax(prob)
                for idx,cell in enumerate(g):
                    if idx == group_max_prob_idx:
                        g.max_prob = prob[group_max_prob_idx]
                         g.max_prob_cell = max_prob_cell
                        break
                        
            # reset all group cells
            for cell in g:
                cell.combination_mark = None
        
        # no certanties in all groups, pick best option
        if len(self.set_to_flag) == 0 or len(self.set_to_reveal) == 0:
            for g in groups:
                if g.max_prob > max_prob:
                    self.set_to_flag.add(g.max_prob_cell.location)      
        return 1

    def evaluateCombination(self, group:ContiguousGroup, frontier_cells: List[MSGridElement]):
        for cell in frontier_cells:
            for n in cell.surround:
                if not n.touched and not n.isEdge:
                    if n not in group:
                        return

            numMarkedCells = sum(
                1 for n in cell.surround if not n.isEdge and n.combination_mark == 1
            )
            if numMarkedCells != cell.revealedReducedValue:
                return False
        return True

    def randomGuess(self, frontier_cells: List[MSGridElement]):
        return 1
    
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

    # def binary_list(n, k):
    #     if not (0 <= k < 2 ** n):
    #         raise ValueError(f"k must be in the range 0 <= k < 2^{n} (got k={k})")
    #     return [(k >> i) & 1 for i in range(n)]

    def binary_mask_arr(n) -> np.array:
        """
        Returns a 2D numpy array of shape (2**n, n), where each row is the binary representation
        of k (0 <= k < 2**n), least significant bit first.
        """
        return np.array([[(k >> i) & 1 for i in range(n)] for k in range(2 ** n)])


if __name__ == "__main__":
    # Example usage
    grid = MSGrid(20, 8, nMines=40)
    grid.instantiateGrid()
    solver = MineSolver(grid, debug=False)

    # Print the formatted cell
    # grid.print(PrintMode.RevealAll)
    # print()
    grid.print(PrintMode.RevealMines)
    print()
    # solver.solverIteration()
    # grid.print(PrintMode.RevealMines)
    solver.solve(
        print_mode=PrintMode.RevealMines,
    )
    print(solver.n_iterations)
