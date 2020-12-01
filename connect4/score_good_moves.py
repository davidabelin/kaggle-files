def score(agent, max_lines = 1000):
    ''' scores an agent against a set of "perfect moves" '''
    
    global env
    # required imports
    import json
    #!pip install kaggle_environments
    import kaggle_environments
    from kaggle_environments.utils import structify
    
    #also requires a copy of 'refmoves1k_kaggle.csv'
    
    def win_loss_draw(score):
        if score>0: 
            return 'win'
        if score<0: 
            return 'loss'
        return 'draw'


    print("scoring ",agent)
    count = 0
    good_move_count = 0
    perfect_move_count = 0
    observation = structify({'mark': None, 'board': None})
    #with open("/kaggle/input/1k-connect4-validation-set/refmoves1k_kaggle") as f:
    with open("/content/refmoves1k_kaggle.csv") as f:
        for line in f:
            count += 1
            data = json.loads(line)
            observation.board = data["board"]
            # find out how many moves are played to set the correct mark.
            ply = len([x for x in data["board"] if x>0])
            if ply&1:
                observation.mark = 2
            else:
                observation.mark = 1
            
            #call the agent
            agent_move = agent(observation,env.configuration)
            
            moves = data["move score"]
            perfect_score = max(moves)
            perfect_moves = [ i for i in range(7) if moves[i]==perfect_score]

            if(agent_move in perfect_moves):
                perfect_move_count += 1

            if win_loss_draw(moves[agent_move]) == win_loss_draw(perfect_score):
                good_move_count += 1

            if count == max_lines:
                break

        print("perfect move percentage: ",perfect_move_count/count)
        print("good moves percentage: ",good_move_count/count)

#to call:
#from kaggle_environments import make
#env = make("connectx")
#score(env.agents["random"],100)
#score(agentX,1000)