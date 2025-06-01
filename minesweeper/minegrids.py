import numpy as np

from grids import GridElement, Grid

class MSGridElement(GridElement):
    def __init__(self, parent, x, y, edgeENWS = ...):
        super().__init__(parent, x, y, edgeENWS)
        self.value = 0
    pass

class MSGrid(Grid):
    nMines = 0

    def __init__(self, nX, nY, seed=1234):
        super().__init__(nX, nY)
        self.seed = seed

    def instantiateGrid(self, nMines, *args):
        super().instantiateGrid(MSGridElement, *args)
        self.nMines = nMines

    