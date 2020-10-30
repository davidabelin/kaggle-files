def my_agent(obs, config):
    ### first derived from agent_2q() ###
    ### agent to be submitted  ###
    import random
    good_moves = []
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    for piece in [obs.mark, obs.mark%2+1]:
        for col in valid_moves:
            if check_winning_move(obs, config, col, piece):
                good_moves.append(col)
    if len(good_moves) == 0:
        good_moves = valid_moves
    return random.choice(good_moves)
