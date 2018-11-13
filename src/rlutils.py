import numpy as np
import random

def give_reward(battle):
    return 1

def action_index_to_tuple(index,type):
    action = np.zeros(10)
    if type == 'move':
        action[index] = 1
    elif type == 'switch':
        action[index+4] = 1
    return action

def action_tuple_to_index(tuple):
    for i,t in enumerate(tuple):
        if t == 1:
            if i <= 3:
                return i
            else:
                return i-4

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions, sess, DQNetwork):
    """
    This function will do the part
    With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
    Initialize the weights
    Init the environment
    Initialize the decay rate (that will use to reduce epsilon) 

    For episode to max_episode do
        Make new episode
        Set step to 0
        Observe the first state $s_0$ 

        While step < max_steps do:
            Increase decay_rate
            With $\epsilon$ select a random action $a_t$, otherwise select $a_t = \mathrm{argmax}_a Q(s_t,a)$
            Execute action $a_t$ in simulator and observe reward $r_{t+1}$ and new state $s_{t+1}$
            Store transition $
            Sample random mini-batch from $D$: $$
            Set $\hat{Q} = r$ if the episode ends at $+1$, otherwise set $\hat{Q} = r + \gamma \max_{a'}{Q(s', a')}$
            Make a gradient descent step with loss $(\hat{Q} - Q(s, a))^2$
        endfor 
    endfor
    """
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(actions)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = actions[int(choice)]
                
    return action, explore_probability

def make_state(battle):
    """
    Constructs the state vector given the battle object.
    What each value in the vector represents:
    
    
    pkmn[0:18] isPokemonTypeOne{x}  x => fire, water, etc
    pkmn[18:36] isPokemonTypeTwo{x}  x => fire, water, etc
    pkmn[36,37,38,39,40,41,42] => HP, atk, def, spa, spd, spe, lvl
    pkmn[43,44,45,46,47,48] => isFnt, isSlp, isPsn, isBrn, isPar, isFrz
    pkmn[49] => current pokemon
    50 params

    For each move: 
    mv[0:18] moveOneIsType{x} x => fire,water,etc
    mv[18,19,20] => isPhys, isSpec, isStat
    mv[21,22] => power, accuracy
    23 params

    each pokemon has 4 moves 
    50 + 23*4 = 142 params

    6 pokemon on each team, 2 teams
    142 * 6 * 2 = 1704 features in state

    """
    state = make_team_vector(battle.bot_team)
    state = np.append(state, [make_team_vector(battle.enemy_team)])
    return state

def make_team_vector(team):
    team_vector = np.empty(0)
    ctr = 0

    if team:
        for pokemon in team.pokemons:
            ctr += 1
            team_vector = np.append(team_vector, [make_pkmn_vector(pokemon)])

    while True:
        if ctr > 5:
            break
        ctr += 1
        team_vector = np.append(team_vector, [np.full(142,-1)])
    
    return team_vector

def make_pkmn_vector(pokemon):
    #print('POKEMON:')
    #print(pokemon.name)
    pokemon_vector = np.full(50,-1)
    
    pokemon_vector[0:18] = make_element_type_vector(pokemon.types[0])
    if len(pokemon.types) > 1:
        pokemon_vector[18:36] = make_element_type_vector(pokemon.types[1])
    else:
        pokemon_vector[18:36] = np.zeros(18)

    pokemon_vector[36] = pokemon.stats['hp']
    pokemon_vector[37] = pokemon.stats['atk']
    pokemon_vector[38] = pokemon.stats['def']
    pokemon_vector[39] = pokemon.stats['spa']
    pokemon_vector[40] = pokemon.stats['spd']
    pokemon_vector[41] = pokemon.stats['spe']
    pokemon_vector[42] = pokemon.level

    pokemon_vector[43:49] = make_status_vector(pokemon)
    #print(pokemon_vector)
    ctr = 0
    for move in pokemon.moves:
        if ctr > 3:
            break
        pokemon_vector = np.append(pokemon_vector,[make_move_vector(move)])
        ctr += 1
    
    while ctr < 4: 
        ctr += 1
        pokemon_vector = np.append(pokemon_vector,[np.full(23,-1)])

    #print('MOVES: '+str(ctr))
    return pokemon_vector

def make_move_vector(move):
    move_vector = np.full(23, 0)
    move_vector[0:18] = make_element_type_vector(move['type'])
    
    if move['category'] == 'Physical':
        move_vector[18] = 1
    elif move['category'] == 'Special': 
        move_vector[19] = 1
    elif move['category'] == 'Status':
        move_vector[20] = 1
    
    move_vector[21] = move['basePower']
    move_vector[22] = move['accuracy']
    return move_vector

def make_element_type_vector(type):
    types = ['Bug','Dark','Dragon','Electric','Fairy','Fighting','Fire','Flying','Ghost','Grass','Ground','Ice','Normal','Poison','Psychic','Rock','Steel','Water' ]
    type_vector = np.full(18,-1)

    if type is not None:
        for i,t in enumerate(types):
            if type == t:
                type_vector[i] = 1
            else:
                type_vector[i] = 0

    return type_vector

def make_status_vector(status):
    status_vector = np.full(6,-1)
    if status is not None:
        for i,r in enumerate(status_vector):
            if(status == i+1):
                status_vector[i] = 1
            else:
                status_vector[i] = 0  
    return status_vector