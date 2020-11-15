def test_agent(obs, config, debug=False):

    import numpy as np
    import random
    import time

    # constants (given by game)
    ROWS = config.rows
    COLUMNS = config.columns
    CNCTX = config.inarow
    ## coefficients (weights on variable future outcomes)
    A = 1     #my twos
    B = 100    #my threes
    C = 10000   #my fours         
    D = -10    #opp-threes
    E = -1000   #opp-fours
    
    # vary lookahead depth according to state of play:
    if obs.board.count(1) < 2:
        N_STEPS =      1
    elif obs.board.count(1) <= ROWS*COLUMNS//(2*3):   # 2 up to one third
        N_STEPS =      2 
    elif obs.board.count(1) <= 3*ROWS*COLUMNS//(2*4): # 3 up to three fouths
        N_STEPS =      3
    else:                                             # 4 last 25% of game
        N_STEPS =      4

    if debug:
        if obs.board.count(1) == 0:
            print(f'"configuration":{config}')  
        print(f'\n###### Agent mark {obs.board.count(1):02} ######') 
        print(f'###### Game marks {(obs.board.count(2) + obs.board.count(1)):02} ######') 
        print(f'Using {N_STEPS} step lookahead')

    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark):
        next_grid = grid.copy()
        for row in range(ROWS-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for get_score: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece):
        return (window.count(piece) == num_discs and window.count(0) == CNCTX-num_discs)

    # Helper function for get_score: counts number of windows satisfying specified heuristic conditions
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

    # Helper function for minimax: calculates value of heuristic score
    # for grid and checks if the grid is terminal
    def get_score(grid, mark):
        num_twos = count_windows(grid, 2, mark) #A
        num_threes = count_windows(grid, 3, mark)  #B
        num_fours = count_windows(grid, 4, mark)   #C
        num_threes_opp = count_windows(grid, 3, mark%2+1) #D
        num_fours_opp = count_windows(grid, 4, mark%2+1)  #E     
        score = A*num_twos + B*num_threes + C*num_fours + D*num_threes_opp + E*num_fours_opp
        is_terminal = (not num_fours == 0) or (not num_fours_opp == 0) or (list(grid[0, :]).count(0) == 0)
        return score, is_terminal

    # Minimax algorithm with alphabeta pruning implementation:
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

    # Uses alphabeta pruning to calculate value
    # of dropping piece in selected column
    def score_move(grid, col, mark, depth):
        if debug:
            column_time = time.time()
        next_grid = drop_piece(grid, col, mark)
        score = alphabeta(next_grid, depth-1, -np.Inf, np.Inf, False, mark)      
        if debug:
            column_time = time.time() - column_time 
            summary_stats = {
                'column': col,
                'score': score,
                'column_time': round(column_time, 4),
                #'time_left': round(time_left, 3),
                #'time_elapsed': round(time.time() - START_TIME, 3)
            }
            print(f'"summary_stats":{summary_stats}')
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
    
    #select column in order of preference
    for pref in [3,4,2,6,0,5,1]: 
        if pref in max_cols:
            choice = pref
            break

    if debug:
        print("Chosen column:", choice)

    return choice
