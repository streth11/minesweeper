import numpy as np

from enum import Enum
from grids import GridElement, Grid
from print_tools import PrintMode, TXT_COL, COL_MAP, print_styled

class MSGridElement(GridElement):
    def __init__(self, parent, x, y, edgeENWS = ...):
        super().__init__(parent, x, y, edgeENWS)
        self.value = 0
        self.revealed = False
        self.flagged = False
        self.is_mine = False

    @property
    def touched(self):
        return self.revealed or self.flagged

    def reveal(self):
        self.revealed = True

    def runRecursiveReveal(self):
        """Recursively reveal this cell and its neighbors if it has no mines around."""
        if self.revealed or self.is_mine or self.flagged:
            return
        self.reveal()
        if self.value == 0:
            for neighbor in self.surround:
                if not neighbor.isEdge:
                    neighbor.runRecursiveReveal()

    def hasRevealedNeighbors(self):
        """Check if any non-edge neighbor is revealed."""
        return any(neighbor.touched for neighbor in self.surround if not neighbor.isEdge)

    def numFlaggedNeighbors(self):
        """Count the number of flagged neighbors."""
        return sum(1 for neighbor in self.surround if not neighbor.isEdge and neighbor.flagged)

class MSGridState(Enum):
    UNINITIALIZED = 0
    INIT = 1
    IN_PROGRESS = 2
    SOLVED = 3
    FAILED = 4

class MSGrid(Grid):
    nMines = 0

    def __init__(self, nX, nY, seed=1234, nMines=1):
        super().__init__(nX, nY)
        self.seed = seed
        np.random.seed(seed)
        self.nMines = nMines
        self.mine_positions = set()
        self.state = MSGridState.UNINITIALIZED

    def instantiateGrid(self, *args):
        """Instantiate the grid with a given number of mines."""
        if self.state == MSGridState.FAILED or self.state == MSGridState.SOLVED:
            self.state = MSGridState.UNINITIALIZED
        if self.state != MSGridState.UNINITIALIZED:
            raise ValueError("Cannot re-instantiate in current state.")
        super().instantiateGrid(MSGridElement, *args)
        self.state = MSGridState.INIT

        self.randomizeMines()
        self.state = MSGridState.IN_PROGRESS

    def randomizeMines(self):
        """Randomly place mines in the grid."""
        if self.state != MSGridState.INIT:
            raise ValueError("Grid must be in INIT state to randomize mines.")
        
        if self.nMines < 0:
            raise ValueError("Number of mines cannot be negative.")
        if self.nMines > self.nX * self.nY:
            raise ValueError("Number of mines exceeds grid size.")
        
        self.mine_positions.clear()
        while len(self.mine_positions) < self.nMines:
            x = np.random.randint(0, self.nX)
            y = np.random.randint(0, self.nY)
            self.mine_positions.add((x, y))
        
        for x, y in self.mine_positions:
            self[x, y].is_mine = True
            for neighbor in self[x, y].surround:
                if not neighbor.isEdge and not neighbor.is_mine:
                    neighbor.value += 1

    def revealCell(self, x, y):
        """Reveal a cell at (x, y). 
           Returns 0 if already revealed, -1 if mine, 1 if successful."""
        if x >= self.nX or y >= self.nY or x < 0 or y < 0:
            raise ValueError("Coordinates out of bounds.")
        if self[x, y].revealed:
            return 0
        if self[x, y].is_mine:
            self.state = MSGridState.FAILED
            return -1
        self[x, y].runRecursiveReveal()
        if self.numMinesRemaining(truth=True) == 0:
            self.state = MSGridState.SOLVED
        return 1

    def flagCell(self, x, y):
        """Toggle flag on a cell at (x, y).
           Returns 0 if already revealed, 1 if successful."""
        if x >= self.nX or y >= self.nY or x < 0 or y < 0:
            raise ValueError("Coordinates out of bounds.")
        cell = self[x, y]
        if not cell.revealed:
            cell.flagged = not cell.flagged
            return 1
        return 0

    def valueArray(self):
        """Returns a 2D numpy array of cell values."""
        return np.array([[cell.value for cell in row] for row in self.grid])

    def numMinesRemaining(self, truth=False):
        """Returns the number of mines remaining"""
        if truth:
            return self.nMines - sum(cell.flagged and cell.is_mine for row in self.grid for cell in row)
        else:
            return self.nMines - sum(cell.flagged for row in self.grid for cell in row)

    @property
    def numFlaggedCells(self):
        return sum(cell.flagged for row in self.grid for cell in row)
    
    @property
    def numTouchedCells(self):
        return sum(cell.touched for row in self.grid for cell in row)
    
    @property
    def numRevealedCells(self):
        return sum(cell.revealed for row in self.grid for cell in row)

    def untouchedFrontier(self):
        """Returns a list of cells that are not revealed or flagged."""
        return [cell for row in self.grid for cell in row if not cell.touched]

    def getCellFormat(self, cell:MSGridElement, print_mode:PrintMode=PrintMode.Normal):
        """Returns a formatted string for the cell based on the print mode."""
        if print_mode == PrintMode.Normal:
            if cell.touched:
                if cell.flagged:
                    styled_out = print_styled('X', bold=True, fg_rgb=COL_MAP["red"])
                else:
                    if cell.value == 0:
                        styled_out = print_styled(' ')
                    else:
                        styled_out = print_styled(str(cell.value), bold=True, fg_rgb=TXT_COL[str(cell.value)])
            else:
                styled_out = print_styled(' ', bold=True, bg_rgb=COL_MAP["gray"])
        elif print_mode == PrintMode.RevealMines:
            if cell.is_mine:
                if cell.flagged:
                    styled_out = print_styled('X', bold=True, fg_rgb=COL_MAP["red"])
                else:
                    styled_out = print_styled(' ', bold=True, bg_rgb=COL_MAP["red"])
            else:
                if cell.flagged:
                    styled_out = print_styled('X', bold=True, bg_rgb=COL_MAP["yellow"])
                elif cell.revealed:
                    if cell.value == 0:
                        styled_out = print_styled(' ')
                    else:
                        styled_out = print_styled(str(cell.value), bold=True, fg_rgb=TXT_COL[str(cell.value)])
                else:
                    styled_out = print_styled(' ', bold=True, bg_rgb=COL_MAP["gray"])
        elif print_mode == PrintMode.RevealAll:
            if cell.is_mine:
                if cell.flagged:
                    styled_out = print_styled('X', bold=True, fg_rgb=COL_MAP["red"])
                else:
                    styled_out = print_styled(' ', bold=True, bg_rgb=COL_MAP["red"])
            else:
                if cell.flagged:
                    styled_out = print_styled(str(cell.value), bold=True, bg_rgb=COL_MAP["yellow"])
                elif cell.value == 0:
                    styled_out = print_styled(' ')
                else:
                    styled_out = print_styled(str(cell.value), bold=True, fg_rgb=TXT_COL[str(cell.value)])

        return f" {styled_out} "
    

if __name__ == "__main__":
    # Example usage
    grid = MSGrid(20, 8, nMines=40)
    grid.instantiateGrid()
    
    # Print the formatted cell
    grid.print(PrintMode.RevealAll)
    print()
    grid.print(PrintMode.Normal)
    grid.revealCell(0, 0)
    print(grid.flagCell(2, 3))
    print(grid.flagCell(2, 4))
    grid.print()
    grid.print(PrintMode.RevealMines)
