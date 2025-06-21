import numpy as np
from enum import Enum
from typing import List
from minegrids import MSGrid, MSGridState, MSGridElement
from print_tools import PrintMode


class MineSolverState(Enum):
    START = 0
    FRONTIER_SIMPLE_SOLVE = 1
    FRONTIER_PATTERN_SOLVE = 2
    RANDOM_GUESS = 3
    COMPLETE = 4


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

    def solve(self, until_state=MineSolverState.COMPLETE, print_mode=PrintMode.NoPrint):
        game_over = 0
        while game_over == 0:
            game_over = self.solverIteration()
            if self.state == until_state:
                break
            if print_mode != PrintMode.NoPrint:
                self.grid.print(print_mode)
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

        elif (
            self.state == MineSolverState.FRONTIER_SIMPLE_SOLVE
            or self.state == MineSolverState.FRONTIER_PATTERN_SOLVE
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

            self.iter_touched_cells, self.iter_mines_remaining = (
                self.getCurrentGridState()
            )
            if ret is None:
                raise ValueError("Random guess returned None.")

            return MineSolverState.FRONTIER_SIMPLE_SOLVE

        else:
            raise ValueError(f"Unknown state: {self.state}")

    def getCurrentGridState(self):
        return self.grid.numTouchedCells, self.grid.numMinesRemaining()

    def frontierSimpleSolve(self, frontier_cells: List[MSGridElement]):
        for cell in frontier_cells:
            unrevealed_neighbors = cell.unrevealedUnflaggedNeighbors()
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
        for cell in frontier_cells:
            # 1-2-1 pattern
            self.pattern121Solve(cell)
        return 1

    def pattern121Solve(self, cell: MSGridElement):
        if cell.revealedReducedValue != 2:
            return
        if len(cell.unrevealedUnflaggedNeighbors()) != 3:
            return

        # get target side
        unrevealed_cells = [
            1 if not n.isEdge and not n.touched else 0 for n in cell.cardinalSurround
        ]
        if sum(unrevealed_cells) != 1:
            return
        indices = [2 * i for i, _ in enumerate(unrevealed_cells)]
        side_idx = indices[0]
        side_cell = cell.surround[side_idx]

        # todo check values of other cells
        if (
            cell.surround[side_idx + 2].revealedReducedValue != 1
            or cell.surround[side_idx - 2].revealedReducedValue == 1
        ):
            return

        for j in [side_idx - 1, side_idx, side_idx + 1]:
            if cell.surround[j].touched or cell.surround[j].isEdge:
                return
        i = 1
        # check cells either side

    def randomGuess(self):
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
        until_state=MineSolverState.FRONTIER_PATTERN_SOLVE,
        print_mode=PrintMode.RevealMines,
    )
    print(solver.n_iterations)

    solver.pattern121Solve(grid[10, 1])
