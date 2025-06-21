import numpy as np

class NeighbourList(list):
    def __getitem__(self, key):
        if key >= 8 or key <= -8:
            key = key % 8
        return super().__getitem__(key)


class EdgeElement():
    def __init__(self, parent, *args):
        self.parent = parent
        self.isEdge = True
        self.value = -1
    
        
class GridElement():
    def __init__(self, parent, x:int,y:int, edgeENWS:list=[0,0,0,0]):
        self.x = x
        self.y = y
        self.parent = parent
        self.isEdge = False
        self.surround = []
        self.edgeENWS = [bool(e) for e in edgeENWS]
        self.value = x+y

    def getSurrounding(self):
        self.surround = NeighbourList((
            self._east(),
            self._northEast(),
            self._north(),
            self._northWest(),
            self._west(),
            self._southWest(),
            self._south(),
            self._southEast()
        ))

    @property
    def cardinalSurround(self):
        """Return the cardinal (E, N, W, S) neighbors as a list."""
        return [self.surround[i] for i in (0, 2, 4, 6)]

    @property
    def location(self):
        """Returns the (x, y) location of this element."""
        return (self.x, self.y)

    def _east(self):
        if self.edgeENWS[0]:
            return EdgeElement(self.parent)
        return self.parent[self.x+1,self.y]
    def _north(self):
        if self.edgeENWS[1]:
            return EdgeElement(self.parent)
        return self.parent[self.x,self.y+1]
    def _west(self):
        if self.edgeENWS[2]:
            return EdgeElement(self.parent)
        return self.parent[self.x-1,self.y]
    def _south(self):
        if self.edgeENWS[3]:
            return EdgeElement(self.parent)
        return self.parent[self.x,self.y-1]
    def _northEast(self):
        if self.edgeENWS[0] or self.edgeENWS[1]:
            return EdgeElement(self.parent)
        return self.parent[self.x+1,self.y+1]
    def _northWest(self):
        if self.edgeENWS[1] or self.edgeENWS[2]:
            return EdgeElement(self.parent)
        return self.parent[self.x-1,self.y+1]
    def _southWest(self):
        if self.edgeENWS[2] or self.edgeENWS[3]:
            return EdgeElement(self.parent)
        return self.parent[self.x-1,self.y-1]
    def _southEast(self):
        if self.edgeENWS[3] or self.edgeENWS[0]:
            return EdgeElement(self.parent)
        return self.parent[self.x+1,self.y-1]
    
    @property
    def east(self):
        return self.surround[0]
    @property
    def north(self):
        return self.surround[2]
    @property
    def west(self):
        return self.surround[4]
    @property
    def south(self):
        return self.surround[6]
    @property
    def northEast(self):
        return self.surround[1]
    @property
    def northWest(self):
        return self.surround[3]
    @property
    def southWest(self):
        return self.surround[5]
    @property
    def southEast(self):
        return self.surround[7]
    
    def __repr__(self):
        return f'<{self.__class__.__qualname__}(x={self.x},y={self.y}) object at {hex(id(self))}>'


class Grid():
    def __init__(self, nX: int, nY: int):
        self.nX = nX
        self.nY = nY
        self.n = self.nX * self.nY
        
    def instantiateGrid(self, GridElemType=GridElement, *args):
        """Instantiates the grid with GridElemType elements."""
        self.grid = np.empty((self.nX, self.nY), dtype=object)
        
        for i in range(self.nX):
            for j in range(self.nY):
                edgeENWS = [0,0,0,0]
                if i == 0:
                    edgeENWS[2] = 1
                if i == self.nX - 1:
                    edgeENWS[0] = 1
                if j == 0:
                    edgeENWS[3] = 1
                if j == self.nY - 1:
                    edgeENWS[1] = 1
                self.grid[i,j] = GridElemType(self, i, j, edgeENWS, *args)

        for i in range(self.nX):
            for j in range(self.nY):
                self.grid[i,j].getSurrounding()
        return self.grid

    def __getitem__(self, key):
        """Returns the grid element at the specified key."""
        if isinstance(key,tuple):
            if len(key) != 2:
                raise ValueError()
            x,y = key
            if x < 0 or x > self.nX:
                raise ValueError()
            if y < 0 or y > self.nY:
                raise ValueError()
            return self.grid[x,y]
        return self.grid

    def print(self, *args):
        """Prints the grid values with grid lines and colored output."""
        cell_width = 3
        horizontal = "+" + ("-" * cell_width + "+") * self.nX

        for y in reversed(range(self.nY)):
            print(horizontal)
            row = "|"
            for x in range(self.nX):
                cell = self.getCellFormat(self.grid[x,y], *args)
                row += cell.center(cell_width) + "|"
            print(row)
        print(horizontal)

    def getCellFormat(self, cell:GridElement) -> str:
        """Returns the format string for a cell at (x, y)."""
        val = "X"
        return f" {val} "
    
    def flat(self):
        """Returns a flattened view of the grid."""
        return self.grid.flatten()



if __name__ == "__main__":
    g = Grid(3,4)
    g.instantiateGrid()

    g.print()

    x = g[1,1]
    print(g[1,1])

    s = g[1,1].surround
    print(s)
    print(g[1,1].surround)
    print(g[1,1].surround[0])
    print(g[1,1].surround[9])
    print(g[1,1].surround[-9])