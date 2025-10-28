def guesstimator(obs, config, N_STEPS=2, debug=False):

    ###########################
    # constants (given by game)
    ROWS = 6
    COLUMNS = 7
    CNCTX = 4
    #config = {"rows":ROWS, "columns":COLUMNS, "inarow":CNCTX}
    ## coefficients
    A = 2     #my twos
    B = 20    #my threes
    C = 200   #my fours
    D = -1    #opp-twos
    E = -10    #opp-threes
    F = -100   #opp-fours
    
    # printout every few
    

    # vary lookahead depth according to number finished columns:
    if obs.board.count(0) < ROWS*COLUMNS//2 + 6:
        N_STEPS =      3   
    if obs.board[:7].count(0) < 4:
        N_STEPS =      5      
    elif obs.board[:7].count(0) < 6:
        N_STEPS =      3          

    if debug:
        print(f'\n###### Agent Turn: {obs.board.count(1):02} ######') 
        print(f'Using {N_STEPS} step lookahead')

    # Gets obs.board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark):
        next_grid = grid.copy()
        for row in range(ROWS-1, -1, -1):    
            if next_grid[row][col] == 0:
                next_grid[row][col] = mark
                break
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

    def check_fours(grid, mark):
        num_fours = count_windows(grid, 4, mark)   #C
        num_fours_opp = count_windows(grid, 4, mark%2+1)  #F
        is_terminal = (num_fours != 0) or (num_fours_opp != 0) or (list(grid[0, :]).count(0) == 0)
        return num_fours, num_fours_opp, is_terminal
    
    # Helper function for alphabeta: calculates value of heuristic for grid
    def get_score(grid, mark):
        is_terminal = False
        num_fours, num_fours_opp, is_terminal = check_fours(grid,mark)
        if is_terminal:
            return C*num_fours + F*num_fours_opp, is_terminal
        num_twos = count_windows(grid, 2, mark) #A
        num_threes = count_windows(grid, 3, mark)  #B
        num_twos_opp = count_windows(grid, 2, mark%2+1) #D
        num_threes_opp = count_windows(grid, 3, mark%2+1) #E
        score = A*num_twos + B*num_threes + C*num_fours + D*num_twos_opp + E*num_threes_opp + F*num_fours_opp
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
        next_grid = drop_piece(grid, col, mark)
        score = alphabeta(next_grid, depth-1, -np.Inf, np.Inf, False, mark)
        if debug:
            summary_stats = {'column': col, 'score': score}
            print(f'"summary_stats":{summary_stats}')
        return score
    
    def first_pass(grid, col, mark):
        _,_, self_is_terminal = check_fours(drop_piece(grid, col, mark), mark)
        #if self_is_terminal:
        return self_is_terminal
        #_,_, opp_is_terminal = check_fours(drop_piece(grid, col, mark%2+1), mark%2+1)
        #return opp_is_terminal
        
    def get_probs(valid_scores, valid_moves):
        scores = list(valid_scores.values())
        min_score = np.min(scores)
        max_score = np.max(scores)
        if min_score <= 0.0:
            if max_score <= 0.0:
                scores = [s-min_score+0.001 for s in scores]
            else:
                scores = [0.001 if s<=0 else s for s in scores]
        #else: no action
        sum_scores = np.sum(scores)
        valid_probs = {m:s for m,s in zip(valid_moves, scores/sum_scores)}
        probs = [valid_probs[i] if i in valid_moves else 0.0 for i in range(7)]
        return probs

    #########################
    # Agent makes selection #
    #########################

    # Get list of valid moves
    valid_moves = [c for c in range(COLUMNS) if obs.board[c] == 0]

    # Convert the obs.board to a 2D grid
    grid = np.asarray(obs.board).reshape(ROWS, COLUMNS)

    # Do a quick pass at depth zero to see if there is a positive terminal node
    quick_pick = False
    for col in valid_moves:
        quick_pick = first_pass(grid, col, obs.mark)
        if quick_pick:
            choice = col  
            probs = [1.0 if i == col else 0.0 for i in range(7)] 
            if debug:
                print("Column {} is terminal.".format(choice))
            break
    
    if not quick_pick:  
        #if np.random.choice([False, True, True]):
        valid_scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, N_STEPS) for col in valid_moves]))
        probs = get_probs(valid_scores, valid_moves)
        choice = int(np.random.choice(list(range(7)), p=probs))
        #else:
            #probs = ["random"]
            #choice = int(np.random.choice(valid_moves))     
    
    if debug:
        print("Probabilities:", probs,"\nChoice:",choice)

    return choice