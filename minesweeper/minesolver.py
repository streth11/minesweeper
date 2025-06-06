import numpy as np
from enum import Enum
from minegrids import MSGrid, MSGridState
from print_tools import PrintMode

class MineSolverState(Enum):
    START = 0
    FRONTIER_SIMPLE_SOLVE = 1
    FRONTIER_PATTERN_SOLVE = 2
    RANDOM_GUESS = 3

class MineSolver():
    def __init__(self, grid: MSGrid):
        self.grid = grid
        self.nMines = grid.nMines
        
        self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()
        self.prev_touched_cells = 0
        self.prev_mines_remaining = grid.nMines
        
        self.state = MineSolverState.START
        self.n_iterations = 0

    def solve(self):
        game_over = 0
        while game_over == 0:
            game_over = self.solverIteration()
        return game_over, self.n_iterations

    def solverIteration(self):
        self.state = self.runSolverStateMachine()
        self.prev_touched_cells = self.iter_touched_cells
        self.prev_mines_remaining = self.iter_mines_remaining
        self.n_iterations += 1
        if self.grid.state == MSGridState.IN_PROGRESS:
            return 0
        elif self.grid.state == MSGridState.SOLVED:
            return 1
        elif self.grid.state == MSGridState.FAILED:
            return -1
        else:
            raise ValueError(f"Bad grid state: {self.grid.state}")

    def runSolverStateMachine(self) -> MineSolverState:
        if self.state == MineSolverState.START:
            if self.iter_touched_cells != 0:
                raise ValueError("Solver started with non-zero touched cells.")
            ret = self.startGuess()
            
            self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()
            if ret is None:
                raise ValueError("Start guess returned None.")
            return MineSolverState.FRONTIER_SIMPLE_SOLVE
        
        elif self.state == MineSolverState.FRONTIER_SIMPLE_SOLVE or self.state == MineSolverState.FRONTIER_PATTERN_SOLVE:
            iter_frontier_cells = self.grid.getFrontierCells()
        
        if self.state == MineSolverState.FRONTIER_SIMPLE_SOLVE:
            # execute frontier simple solve
            ret = self.frontierSimpleSolve(iter_frontier_cells)
            if ret is None:
                raise ValueError("Frontier simple solve returned None.")
            
            # see what has changed
            self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()
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
            self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()
            # update state based on touched cells
            if self.prev_touched_cells == self.iter_touched_cells:
                return MineSolverState.RANDOM_GUESS
            return MineSolverState.FRONTIER_SIMPLE_SOLVE

        elif self.state == MineSolverState.RANDOM_GUESS:
            # execute random guess
            ret = self.randomGuess()

            self.iter_touched_cells, self.iter_mines_remaining = self.getCurrentGridState()
            if ret is None:
                raise ValueError("Random guess returned None.")
            
            return MineSolverState.FRONTIER_SIMPLE_SOLVE

        else:
            raise ValueError(f"Unknown state: {self.state}")
            
    def getCurrentGridState(self):
        return self.grid.numTouchedCells, self.grid.numMinesRemaining()

    def frontierSimpleSolve(self, frontier_cells):

        for cell in frontier_cells:
            if cell.value == cell.numFlaggedNeighbors():
                for neighbor in cell.unrevealedUnflaggedNeighbors2():
                    ret = self.grid.revealCell(neighbor.location)
                    if ret == 1:
                        self.grid.print(PrintMode.RevealMines)
                    elif ret == -1:
                        raise ValueError("Revealed a mine.")
            
            cell_temp_list = cell.unrevealedUnflaggedNeighbors()
            if cell.value - cell.numFlaggedNeighbors() == sum(cell_temp_list):
                for i, neighbor in enumerate(cell.surround):
                    if cell_temp_list[i] == 1:
                        self.grid.flagCell(neighbor.location)
                        self.grid.print(PrintMode.RevealMines)
        return 1

    def frontierPatternSolve(self):
        pass

    def randomGuess(self):
        pass

    def startGuess(self):
        ret = self.grid.revealCell((0, 0))
        while self.grid.state == MSGridState.FAILED:
            # If the first guess is a mine, try again
            self.grid.instantiateGrid()
            ret = self.grid.revealCell((0, 0))
        if ret != 1:
            raise ValueError("Start guess did not succeed.")
        return 1
    

if __name__ == "__main__":
    # Example usage
    grid = MSGrid(20, 8, nMines=40)
    grid.instantiateGrid()
    solver = MineSolver(grid)
    
    # Print the formatted cell
    # grid.print(PrintMode.RevealAll)
    # print()
    grid.print(PrintMode.RevealMines)
    print()
    solver.solverIteration()
    grid.print(PrintMode.RevealMines)
    solver.solverIteration()
    solver.solverIteration()