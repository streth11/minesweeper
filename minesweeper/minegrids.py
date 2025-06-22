import numpy as np

from enum import Enum
from grids import GridElement, Grid
from print_tools import PrintMode, TXT_COL, COL_MAP, print_styled

class MSEdgeElement:
    def __init__(self, parent, *args):
        self.parent = parent
        self.isEdge = True
        self.value = -1
        self.touched = False


class MSGridElement(GridElement):
    def __init__(self, parent, *args):
        super().__init__(parent, *args)
        self.value = 0
        self.revealed = False
        self.flagged = False
        self.is_mine = False

    @property
    def touched(self):
        return self.revealed or self.flagged

    def reveal(self, debug=False):
        self.revealed = True
        if debug:
            print(f"Revealed cell at ({self.x}, {self.y}).")

    def toggleFlag(self, debug=False):
        self.flagged = not self.flagged
        if debug:
            print(f"Toggled flag on cell at ({self.x}, {self.y}).")

    def runRecursiveReveal(self, debug=False):
        """Recursively reveal this cell and its neighbors if it has no mines around."""
        if self.revealed or self.is_mine or self.flagged:
            return
        self.reveal(debug=debug)
        if self.value == 0:
            for neighbor in self.surround:
                if not neighbor.isEdge:
                    neighbor.runRecursiveReveal(debug=debug)

    @property
    def revealedReducedValue(self):
        """Return the value of the cell, reduced by the number of flagged neighbors."""
        return self.value - self.numFlaggedNeighbors()

    def hasRevealedNeighbors(self):
        """Check if any non-edge neighbor is revealed."""
        return any(
            neighbor.revealed for neighbor in self.surround if not neighbor.isEdge
        )

    def unrevealedNeighbors(self):
        """Return a list of neighbors that are neither revealed nor flagged."""
        return [n for n in self.surround if not n.isEdge and not n.revealed]

    def numFlaggedNeighbors(self):
        """Count the number of flagged neighbors."""
        return sum(
            1 for neighbor in self.surround if not neighbor.isEdge and neighbor.flagged
        )

    def hasUnrevealedUnflaggedNeighbors(self):
        """Return True if any neighbor is neither revealed nor flagged."""
        return any(self.unrevealedUnflaggedNeighbors())

    def numUnrevealedUnflaggedNeighbors(self):
        """Count the number of neighbors that are neither revealed nor flagged."""
        return sum(self.unrevealedUnflaggedNeighbors())

    def unrevealedUnflaggedNeighbors(self):
        """Return a list of neighbors that are neither revealed nor flagged."""
        return [
            n
            for n in self.surround
            if not n.isEdge and not n.revealed and not n.flagged
        ]


class MSGridState(Enum):
    UNINITIALIZED = 0
    INIT = 1
    IN_PROGRESS = 2
    SOLVED = 3
    STALEMATE = 4
    FAILED = 5


class MSGrid(Grid):
    nMines = 0

    def __init__(self, nX, nY, seed=1234, nMines=1):
        super().__init__(nX, nY)
        self.seed = seed
        np.random.seed(seed)
        self.nMines = nMines
        self.mine_positions = set()
        self.state = MSGridState.UNINITIALIZED

    def instantiateGrid(self, mine_coords:set=None, *args):
        """Instantiate the grid with a given number of mines."""
        if self.state == MSGridState.FAILED or self.state == MSGridState.SOLVED:
            self.state = MSGridState.UNINITIALIZED
        if self.state != MSGridState.UNINITIALIZED:
            raise ValueError("Cannot re-instantiate in current state.")
        super().instantiateGrid(GridElemType=MSGridElement, EdgeElementType=MSEdgeElement, *args)
        self.state = MSGridState.INIT

        if mine_coords is None:
            self.randomizeMines()
        else:
            self.mine_positions = mine_coords
            self.nMines = len(mine_coords)

        # instantiate
        for x, y in self.mine_positions:
            self[x, y].is_mine = True
            for neighbor in self[x, y].surround:
                if not neighbor.isEdge and not neighbor.is_mine:
                    neighbor.value += 1

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

    def revealCell(self, pos: tuple, debug=False):
        """Reveal a cell at (x, y).
        Returns 0 if already revealed, -1 if mine, 1 if successful."""
        x, y = pos
        if x >= self.nX or y >= self.nY or x < 0 or y < 0:
            raise ValueError("Coordinates out of bounds.")
        if self[x, y].revealed:
            return 0
        if self[x, y].is_mine:
            self.state = MSGridState.FAILED
            return -1
        self[x, y].runRecursiveReveal(debug=debug)
        if self.numMinesRemaining(truth=True) == 0:
            self.state = MSGridState.SOLVED
        return 1

    def flagCell(self, pos: tuple, debug=False):
        """Toggle flag on a cell at (x, y).
        Returns 0 if already revealed, 1 if successful."""
        x, y = pos
        if x >= self.nX or y >= self.nY or x < 0 or y < 0:
            raise ValueError("Coordinates out of bounds.")
        cell = self[x, y]
        if not cell.revealed:
            cell.toggleFlag(debug=debug)
            return 1
        return 0

    def valueArray(self):
        """Returns a 2D numpy array of cell values."""
        return np.array([[cell.value for cell in row] for row in self.grid])

    def numMinesRemaining(self, truth=False):
        """Returns the number of mines remaining"""
        if truth:
            return self.nMines - sum(
                cell.flagged and cell.is_mine for row in self.grid for cell in row
            )
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

    def untouchedListFlattened(self):
        """Returns a flattened list of untouched cells."""
        return [cell for cell in self.grid.flat if not cell.touched]

    def revealedWithUnrevealedNeighbors(self):
        return [
            cell
            for cell in self.grid.flat
            if cell.revealed and cell.hasUnrevealedUnflaggedNeighbors()
        ]

    def getFrontierCells(self):
        return self.revealedWithUnrevealedNeighbors()

    def getCellFormat(
        self, cell: MSGridElement, print_mode: PrintMode = PrintMode.Normal
    ):
        """Returns a formatted string for the cell based on the print mode."""
        if print_mode == PrintMode.Normal:
            if cell.touched:
                if cell.flagged:
                    styled_out = print_styled("X", bold=True, fg_rgb=COL_MAP["red"])
                else:
                    if cell.value == 0:
                        styled_out = print_styled(" ")
                    else:
                        styled_out = print_styled(
                            str(cell.value), bold=True, fg_rgb=TXT_COL[str(cell.value)]
                        )
            else:
                styled_out = print_styled(" ", bold=True, bg_rgb=COL_MAP["gray"])
        elif print_mode == PrintMode.RevealMines:
            if cell.is_mine:
                if cell.flagged:
                    styled_out = print_styled("X", bold=True, fg_rgb=COL_MAP["red"])
                else:
                    styled_out = print_styled(" ", bold=True, bg_rgb=COL_MAP["red"])
            else:
                if cell.flagged:
                    styled_out = print_styled("X", bold=True, bg_rgb=COL_MAP["yellow"])
                elif cell.revealed:
                    if cell.value == 0:
                        styled_out = print_styled(" ")
                    else:
                        styled_out = print_styled(
                            str(cell.value), bold=True, fg_rgb=TXT_COL[str(cell.value)]
                        )
                else:
                    styled_out = print_styled(" ", bold=True, bg_rgb=COL_MAP["gray"])
        elif print_mode == PrintMode.RevealAll:
            if cell.is_mine:
                if cell.flagged:
                    styled_out = print_styled("X", bold=True, fg_rgb=COL_MAP["red"])
                else:
                    styled_out = print_styled(" ", bold=True, bg_rgb=COL_MAP["red"])
            else:
                if cell.flagged:
                    styled_out = print_styled(
                        str(cell.value), bold=True, bg_rgb=COL_MAP["yellow"]
                    )
                elif cell.value == 0:
                    styled_out = print_styled(" ")
                else:
                    styled_out = print_styled(
                        str(cell.value), bold=True, fg_rgb=TXT_COL[str(cell.value)]
                    )

        return f" {styled_out} "


if __name__ == "__main__":
    # Example usage
    grid = MSGrid(20, 8, nMines=40)
    grid.instantiateGrid()

    # Print the formatted cell
    grid.print(PrintMode.RevealAll)
    print()
    grid.print(PrintMode.Normal)
    grid.revealCell((0, 0))
    print(grid.flagCell((2, 3)))
    print(grid.flagCell((2, 4)))
    grid.print()
    grid.print(PrintMode.RevealMines)
