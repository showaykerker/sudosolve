"""
__author__ = Yash Patel
__name__   = solve.py
__description__ = Given a sudoku board, solves the boarda
"""

import copy

def check_if_valid(arr):
    seen_elems = set()
    for elem in arr:
        if elem in seen_elems:
            return False
        if elem != 0:
            seen_elems.add(elem)
    return True

def is_valid(board):
    to_fill = [[] for _ in range(9)]
    rows    = copy.deepcopy(to_fill)
    cols    = copy.deepcopy(to_fill)
    squares = copy.deepcopy(to_fill)

    for i in range(len(board)):
        for j in range(len(board[i])):
            rows[i].append(board[i][j])
            cols[j].append(board[i][j])
            squares[3 * (i // 3) + j // 3].append(board[i][j])

    for i in range(len(rows)):
        valid_row = check_if_valid(rows[i])
        valid_col = check_if_valid(cols[i])
        valid_square = check_if_valid(squares[i])

        if not valid_row or not valid_col or not valid_square:
            return False
    return True

def _solve(board,i,j):
    if (i,j) == (9,0):
        if is_valid(board):
            return board
        return None

    if j == 8:
        next_i = i + 1
        next_j = 0
    else:
        next_i = i
        next_j = j + 1

    if is_valid(board):
        if board[i][j] == 0:
            for possible in range(1,10):
                board[i][j] = possible
                if _solve(board,next_i,next_j) is not None:
                    return board
            board[i][j] = 0
            return None
        else:
            return _solve(board,next_i,next_j)
    else:
        return None

def solve(board):
    return _solve(board,0,0)

if __name__ == "__main__":
    board = [
        [0,2,3, 0,9,0, 8,0,0],
        [6,0,1, 0,3,8, 0,0,0],
        [5,0,0, 4,0,0, 0,0,0],

        [3,0,0, 0,0,0, 2,0,4],
        [0,0,6, 0,0,0, 3,0,0],
        [7,0,8, 0,0,0, 0,0,1],

        [0,0,0, 0,0,3, 0,0,8],
        [0,0,0, 7,8,0, 9,0,6],
        [0,0,2, 0,5,0, 1,4,0],
    ]

    print(solve(board))