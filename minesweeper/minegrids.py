import numpy as np
from typing import List

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
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.value = 0
        self.revealed = False
        self.flagged = False
        self.is_mine = False
        self.group_id = None
        self.combination_mark = None

    @property
    def touched(self):
        return self.revealed or self.flagged

    def reveal(self, debug=False):
        self.revealed = True
        if self.group_id is not None:
            g = self.parent.getGroupFromID(self.group_id)
            g.invalidate()
        if debug:
            print(f"Revealed cell at ({self.x}, {self.y}).")

    def toggleFlag(self, debug=False):
        self.flagged = not self.flagged
        if self.group_id is not None:
            g = self.parent.getGroupFromID(self.group_id)
            g.invalidate()
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

    def hasUntouchedNeighbors(self):
        """Return True if any neighbor is neither revealed nor flagged."""
        return any(self.untouchedNeighbors())

    def numUntouchedNeighbors(self):
        """Count the number of neighbors that are neither revealed nor flagged."""
        return sum(self.untouchedNeighbors())

    def untouchedNeighbors(self):
        """Return a list of neighbors that are neither revealed nor flagged."""
        return [n for n in self.surround if not n.isEdge and not n.touched]

    def untouchedCardinalNeighbors(self):
        """Return a list of neighbors that are neither revealed nor flagged."""
        return [n for n in self.cardinalSurround if not n.isEdge and not n.touched]


class MSGridState(Enum):
    UNINITIALIZED = 0
    INIT = 1
    IN_PROGRESS = 2
    SOLVED = 3
    STALEMATE = 4
    FAILED = 5


class ContiguousGroup(set):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid = False
        self.id = -1
        self.initStats()
        
    def initStats(self):
        self.max_prob = 0
        self.min_prob = 1
        self.max_prob_cell = None
        self.min_prob_cell = None
        self.valid_comb_min_mines = np.inf

    def setID(self, id):
        self.id = id
        self.valid = True

    def invalidate(self):
        self.valid = False
        self.initStats()
        for cell in self.__iter__():
            cell.group_id = None

    def add(self, cell: MSGridElement):
        super().add(cell)
        cell.group_id = self.id

    def remove(self, cell: MSGridElement):
        cell.group_id = None
        super().remove(cell)

    def pushGroupIDs(self):
        if self.valid:
            for cell in self.__iter__():
                cell.group_id = self.id


class MSGrid(Grid):
    nMines = 0

    def __init__(self, nX, nY, seed=100, nMines=1):
        super().__init__(nX, nY)
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed=seed)
        self.nMines = nMines
        self.mine_positions = set()
        self.state = MSGridState.UNINITIALIZED
        self.groups = []
        self.groupCounter = 0

    def instantiateGrid(self, mine_coords: set = None, *args, **kwargs):
        """Instantiate the grid with a given number of mines."""
        if self.state == MSGridState.FAILED or self.state == MSGridState.SOLVED:
            self.state = MSGridState.UNINITIALIZED
        if self.state != MSGridState.UNINITIALIZED:
            raise ValueError("Cannot re-instantiate in current state.")
        super().instantiateGrid(
            GridElemType=MSGridElement, EdgeElementType=MSEdgeElement, *args, **kwargs
        )
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
            x = self.rng.integers(0, self.nX)
            y = self.rng.integers(0, self.nY)
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
            if cell.revealed and cell.hasUntouchedNeighbors()
        ]

    def getFrontierCells(self):
        return self.revealedWithUnrevealedNeighbors()

    def notInContiguousGroup(self):
        if len(self.groups) == 0:
            return self.untouchedListFlattened()
        return [cell for cell in self.grid.flat if not cell.touched and cell.group_id is None]

    def establishContiguousCells(self, frontier_cells: List[MSGridElement] = None) -> List[ContiguousGroup]:
        if frontier_cells is None:
            frontier_cells = self.getFrontierCells()

        for cell in frontier_cells:
            potential_targets = cell.unrevealedNeighbors()
            for n in potential_targets:
                g_id = n.group_id

                # in a valid group already - skip
                if g_id in self.getValidGroupIDs():
                    continue

                if g_id is not None:
                    # confirm group invalid
                    if self.getGroupFromID(g_id).valid:
                        raise ValueError(f"Group {g_id} should be invalid")

                # no group, look around
                touchingGroups = {
                    s.group_id for s in n.unrevealedNeighbors() if s.group_id is not None
                }
                if len(touchingGroups) == 0:
                    # no neighboring groups, create a group
                    g = self.createGroup()
                    g.add(n)
                elif len(touchingGroups) == 1:
                    # one group, add to group
                    g = self.getGroupFromID(next(iter(touchingGroups)))
                    g.add(n)
                elif len(touchingGroups) > 1:
                    # more than 1 group found
                    g = self.joinGroups(touchingGroups)
                    g.add(n)

        self.cleanupGroups()
        self.cleanupInvalidGroups()
        return self.groups

    def cleanupGroups(self):
        for g in self.groups:
            badCells = set()
            for cell in g:
                if cell.flagged:
                    badCells.add(cell)
            for cell in badCells:
                g.remove(cell)

    def getValidGroupIDs(self) -> list:
        if len(self.groups) == 0:
            return []
        return [g.id for g in self.groups if g.valid]

    def createGroup(self) -> ContiguousGroup:
        new_group = ContiguousGroup()
        new_group.setID(self.groupCounter)
        self.groupCounter += 1
        self.groups.append(new_group)
        return new_group

    def getGroupFromID(self, id) -> ContiguousGroup:
        for g in self.groups:
            if g.id == id:
                return g
        raise ValueError(f"Undefined group with id {id}")

    def joinGroups(self, group_ids: set):
        new_group = self.createGroup()
        for comb_g_id in group_ids:
            old_group = self.getGroupFromID(comb_g_id)
            old_group.invalidate()
            new_group.update(old_group)
        new_group.pushGroupIDs()
        return new_group

    def cleanupInvalidGroups(self):
        self.groups = [g for g in self.groups if g.valid]

    def getCellFormat(
        self,
        cell: MSGridElement,
        print_mode: PrintMode = PrintMode.Normal,
        show_groups=False,
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
                if cell.combination_mark == 1:
                    styled_out = print_styled("M", bold=True, bg_rgb=COL_MAP["gray"])
                elif cell.combination_mark == 0:
                    styled_out = print_styled("-", bold=True, bg_rgb=COL_MAP["gray"])
                else:
                    styled_out = print_styled(" ", bg_rgb=COL_MAP["gray"])
        elif print_mode == PrintMode.RevealMines:
            if cell.is_mine:
                if cell.flagged:
                    styled_out = print_styled("X", bold=True, fg_rgb=COL_MAP["red"])
                else:
                    if show_groups and cell.group_id is not None:
                        styled_out = print_styled(
                            f"{cell.group_id}", bold=True, bg_rgb=COL_MAP["red"]
                        )
                    else:
                        if cell.combination_mark == 1:
                            styled_out = print_styled("M", bold=True, bg_rgb=COL_MAP["red"])
                        elif cell.combination_mark == 0:
                            styled_out = print_styled("-", bold=True, bg_rgb=COL_MAP["red"])
                        else:
                            styled_out = print_styled(" ", bg_rgb=COL_MAP["red"])
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
                    if show_groups and cell.group_id is not None:
                        styled_out = print_styled(
                            f"{cell.group_id}", bold=True, bg_rgb=COL_MAP["gray"]
                        )
                    else:
                        if cell.combination_mark == 1:
                            styled_out = print_styled("M", bold=True, bg_rgb=COL_MAP["gray"])
                        elif cell.combination_mark == 0:
                            styled_out = print_styled("-", bold=True, bg_rgb=COL_MAP["gray"])
                        else:
                            styled_out = print_styled(" ", bg_rgb=COL_MAP["gray"])
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
    grid = MSGrid(20, 8, nMines=28, seed=100)
    grid.instantiateGrid()

    # Print the formatted cell
    grid.print(PrintMode.RevealAll)
    print()
    grid.print(PrintMode.Normal)
    grid.revealCell((0, 0))
    # grid.print()
    grid.revealCell((6, 7))
    grid.revealCell((8, 4))
    grid.revealCell((11, 5))
    grid.revealCell((13, 5))
    grid.revealCell((13, 7))
    grid.print(PrintMode.RevealMines)

    grid.establishContiguousCells()

    grid.print(PrintMode.RevealMines, show_groups=True)
