import numpy as np

from minegrids import MSGrid
from print_tools import PrintMode
from minesolver import MineSolver, MineSolverState


def test_instantiate12X():
    grid = MSGrid(5, 8)
    mine_coords = set([
        (3,0),
        (3,1),
        (3,2),
        (3,3),
        (3,4),
        (3,5),
        (3,6),
        (3,7),
        (3,0),
        (2,0),
        (2,2),
        (2,4),
        (2,5),
        (2,7),
    ])

    grid.instantiateGrid(mine_coords)

    return grid

def test_instantiate11X():
    grid = MSGrid(5, 5)
    mine_coords = set([
        (3,0),
        (3,1),
        (3,2),
        (3,3),
        (3,4),
        (3,0),
        (2,0),
        (2,3),
    ])

    grid.instantiateGrid(mine_coords)

    return grid


if __name__ == "__main__":
    grid = test_instantiate11X()

    grid.print(PrintMode.RevealMines)
    grid.revealCell((0, 0))
    grid.print(PrintMode.RevealMines)
    solver = MineSolver(grid)
    solver.set_to_flag.clear()
    solver.set_to_reveal.clear()
    solver.pattern11XSolve(grid[1,0])
    solver.patternSolveExecuteSets()
    grid.print(PrintMode.Normal)

    # solver = MineSolver(grid, debug=False)
    # grid.print(PrintMode.RevealMines)

    # solver.solve(
    #     print_mode=PrintMode.RevealMines
    # )
    # print(solver.n_iterations)