from ramanujantools.cmf import pFq
from ramanujantools import Position
import sympy as sp

def main():
    ccmf = pFq(6,5,1)
    x0, x1, x2, x3, x4, x5 = sp.symbols('x0 x1 x2 x3 x4 x5')
    y0, y1, y2, y3, y4, y5 = sp.symbols('y0 y1 y2 y3 y4 y5')
    # Stuck
    traj = Position({x0: 8, x1: 0, x2: 7, x3: 7, x4: 1, x5: 6, y0: 13, y1: 14, y2: 8, y3: 14, y4: 19})
    # Hila
    traj = Position({x0: 11, x1: 12, x2: 13, x3: 14, x4: 15, x5: 16, y0: 30, y1: 29, y2: 28, y3: 27, y4: 26})
    # Another
    traj = Position({x0: 11, x1: 12, x2: 13, x3: 14, x4: 15, x5: 16, y0: 25, y1: 25, y2: 25, y3: 25, y4: 25})
    # Another 2
    traj = Position({x0: 8, x1: 0, x2: 7, x3: 7, x4: 1, x5: 6, y0: 10, y1: 10, y2: 10, y3: 10, y4: 10})
    start = Position({x0: 1, x1: 1, x2: 1, x3: 1, x4: 1, x5: 1, y0: 2, y1: 2, y2: 2, y3: 2, y4: 2})
    traj_mat = ccmf.trajectory_matrix(traj, start)
    print(traj_mat)

if __name__ == '__main__':
    main()