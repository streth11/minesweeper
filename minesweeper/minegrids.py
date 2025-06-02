import numpy as np

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


class MSGrid(Grid):
    nMines = 0

    def __init__(self, nX, nY, seed=1234):
        super().__init__(nX, nY)
        self.seed = seed
        np.random.seed(seed)
        self.mine_positions = set()

    def instantiateGrid(self, nMines, *args):
        """Instantiate the grid with a given number of mines."""
        super().instantiateGrid(MSGridElement, *args)
        self.nMines = nMines

        self.mine_positions.clear()
        self.randomizeMines()

    def randomizeMines(self):
        """Randomly place mines in the grid."""
        if self.nMines > self.nX * self.nY:
            raise ValueError("Number of mines exceeds grid size.")
        
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
            return -1
        self[x, y].runRecursiveReveal()
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
    grid = MSGrid(20, 8)
    grid.instantiateGrid(40)
    
    # Print the formatted cell
    grid.print(PrintMode.RevealAll)
    print()
    grid.print(PrintMode.Normal)
    grid.revealCell(0, 0)
    print(grid.flagCell(2, 3))
    print(grid.flagCell(2, 4))
    grid.print()
    grid.print(PrintMode.RevealMines)
