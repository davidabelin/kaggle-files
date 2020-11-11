def prunerBD(obs, config):

    #config is dict: {'rows': 6, 'columns': 7, 'inarow': 4}
    # obs.board is last move of opponent, obs.mark is current player
    # return column that max's next grid's score

    ################################
    # Imports and helper functions #
    ################################

    import numpy as np
    import random

    ########################### Regular pruner ################
    # constants (given by game)
    ROWS = 6
    COLUMNS = 7
    CNCTX = 4
    ## coefficients (weights on variable future outcomes)
    A = 1     #my twos
    B = 10    #my threes
    C = 1000   #my fours         
    D = -10    #opp-threes
    E = -100   #opp-fours

    # vary lookahead depth according to state of play:
    if obs.board.count(0) >= ROWS*COLUMNS/2:
        N_STEPS =      2
    else:
        N_STEPS =      3  # deeper search after half the board is filled

    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark):
        next_grid = grid.copy()
        for row in range(ROWS-1, -1, -1):       ###row in range(0,ROWS)??
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece):
        return (window.count(piece) == num_discs and window.count(0) == CNCTX-num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece):
        num_windows = 0
        # horizontal
        for row in range(ROWS):
            for col in range(COLUMNS-(CNCTX-1)):
                window = list(grid[row, col:col+CNCTX])
                if check_window(window, num_discs, piece):
                    num_windows += 1
        # vertical
        for row in range(ROWS-(CNCTX-1)):
            for col in range(COLUMNS):
                window = list(grid[row:row+CNCTX, col])
                if check_window(window, num_discs, piece):
                    num_windows += 1
        # positive diagonal
        for row in range(ROWS-(CNCTX-1)):
            for col in range(COLUMNS-(CNCTX-1)):
                window = list(grid[range(row, row+CNCTX), range(col, col+CNCTX)])
                if check_window(window, num_discs, piece):
                    num_windows += 1
        # negative diagonal
        for row in range(CNCTX-1, ROWS):
            for col in range(COLUMNS-(CNCTX-1)):
                window = list(grid[range(row, row-CNCTX, -1), range(col, col+CNCTX)])
                if check_window(window, num_discs, piece):
                    num_windows += 1
        return num_windows

    # Helper function for minimax: calculates value of heuristic for grid
    def get_score(grid, mark):
        num_twos = count_windows(grid, 2, mark) #A
        num_threes = count_windows(grid, 3, mark)  #B
        num_fours = count_windows(grid, 4, mark)   #C
        num_threes_opp = count_windows(grid, 3, mark%2+1) #D
        num_fours_opp = count_windows(grid, 4, mark%2+1)  #E     
        score = A*num_twos + B*num_threes + C*num_fours + D*num_threes_opp + E*num_fours_opp
        is_terminal = (not num_fours == 0) or (not num_fours_opp == 0) or (list(grid[0, :]).count(0) == 0)
        return score, is_terminal

    # Minimax implementation was here:
    def alphabeta(node, depth, alpha, beta, maximizingPlayer, mark):
        node_score, is_terminal = get_score(node, mark)
        if depth == 0 or is_terminal:
             return node_score
            
        valid_moves = [c for c in range(COLUMNS) if node[0][c] == 0]

        if maximizingPlayer:
            value = -np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark)
                value = max(value, alphabeta(child, depth-1, alpha, beta, False, mark))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        else: #minimizing player
            value = np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1)
                value = min(value, alphabeta(child, depth-1, alpha, beta, True, mark))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    # Uses minimax to calculate value of dropping piece in selected column
    def score_move(grid, col, mark, nsteps):
        next_grid = drop_piece(grid, col, mark)
        score = alphabeta(next_grid, nsteps-1, -np.Inf, np.Inf, False, mark)     
        return score
    #########################
    # Agent makes selection #
    #########################

    # Get list of valid moves
    valid_moves = [c for c in range(COLUMNS) if obs.board[c] == 0]

    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(ROWS, COLUMNS)

    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, N_STEPS) for col in valid_moves]))

    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    # Select at random from the maximizing columns
    return random.choice(max_cols)
