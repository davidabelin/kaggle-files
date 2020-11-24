#@ title debug_agent
def debug_agent(obs, config, N_STEPS=2, debug=False):

    import numpy as np
    import random
    import time

    ########################### Regular pruner ################
    # constants (given by game)
    ROWS = config.rows
    COLUMNS = config.columns
    CNCTX = config.inarow
    ## coefficients
    A = 2     #my twos
    B = 20    #my threes
    C = 200   #my fours
    D = -1    #opp-twos
    E = -10    #opp-threes
    F = -100   #opp-fours
    
    # vary lookahead depth according to state of play:
    if obs.board.count(0) >= 5*(ROWS*COLUMNS//6):
        N_STEPS =       2     
    elif obs.board.count(0) >= (ROWS*COLUMNS//2):
        N_STEPS =       4  
    elif obs.board.count(0) >= (ROWS*COLUMNS//6):
        N_STEPS =       6
    else:    
        N_STEPS =       8
    
    if debug:
        if obs.board.count(1) == 0:
            print(f'"configuration":{config}')  
        print(f'\n###### Agent Turn {obs.board.count(1):02} ######') 
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
        
    # Quickly checks to see if the game could be won or lost in next step
    def check_terminal(grid, mark):
        num_fours = count_windows(grid, 4, mark)   #C
        num_fours_opp = count_windows(grid, 4, mark%2+1)  #F
        is_terminal = (num_fours != 0) or (num_fours_opp != 0) or (list(grid[0, :]).count(0) == 0)
        return is_terminal
    
    # Helper function for alphabeta: calculates value of heuristic for grid
    def get_score(grid, mark):
        num_fours = count_windows(grid, 4, mark)   #C
        num_fours_opp = count_windows(grid, 4, mark%2+1)  #F
        num_twos = count_windows(grid, 2, mark) #A
        num_threes = count_windows(grid, 3, mark)  #B
        num_twos_opp = count_windows(grid, 2, mark%2+1) #D
        num_threes_opp = count_windows(grid, 3, mark%2+1) #E
        score = A*num_twos + B*num_threes + C*num_fours + D*num_twos_opp + E*num_threes_opp + F*num_fours_opp
        is_terminal = (num_fours != 0) or (num_fours_opp != 0) or (list(grid[0, :]).count(0) == 0)
        return score, is_terminal

    # Minimax with alphabeta pruning implementation:
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
        go = time.time()
        next_grid = drop_piece(grid, col, mark)
        score = alphabeta(next_grid, depth-1, -np.Inf, np.Inf, False, mark)
        if debug:
            summary_stats = {
                'column': col,
                'score': score,
                'column time': round(time.time() - go, 5),
                'time_left': round(2.0 - (time.time() - choice_time), 5)           
            }
            print(f'"summary_stats":{summary_stats}')
        return score
    
    def first_pass(grid, col, mark):
        player_is_terminal = check_terminal(drop_piece(grid, col, mark), mark)
        opp_is_terminal = check_terminal(drop_piece(grid, col, mark%2+1), mark%2+1)
        return opp_is_terminal or player_is_terminal
    
    #########################
    # Agent makes selection #
    #########################

    # Get list of valid moves
    valid_moves = [c for c in range(COLUMNS) if obs.board[c] == 0]

    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(ROWS, COLUMNS)

    # Do a quick pass to see if there is a terminal node on the surface 
    choice_time = time.time()
    quick_pick = False
    for col in valid_moves:
        quick_pick = first_pass(grid, col, obs.mark)
        if quick_pick:
            choice = col   
            if debug:
                print("Column {} is terminal.".format(choice))
            break
    
    if not quick_pick:   
        # Use the heuristic to assign a score to each possible board in the next step
        scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, N_STEPS) for col in valid_moves]))

        # Get a list of columns (moves) that maximize the heuristic
        max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
        
        #select column in order of preference
        for pref in [3,4,2]: 
            if pref in max_cols:
                choice = pref
                break
            else:
                choice = random.choice(max_cols)
    if debug:
        print("Chosen column:", choice)
        print("Choice took:", round(time.time()-choice_time,5))
    return choice
