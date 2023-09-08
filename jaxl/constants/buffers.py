OBSERVATIONS = "observations"
ACTIONS = "actions"
ACT_DIM = "act_dim"
HIDDEN_STATES = "hidden_states"
REWARDS = "rewards"
DONES = "dones"
TERMINATEDS = "terminateds"
TRUNCATEDS = "truncateds"
INFOS = "infos"
NEXT_OBSERVATION = "next_observation"
NEXT_OBSERVATIONS = "next_observations"
NEXT_HIDDEN_STATE = "next_hidden_state"
NEXT_HIDDEN_STATES = "next_hidden_states"
LAST_OBSERVATIONS = "last_observations"
LAST_HIDDEN_STATES = "last_hidden_states"

BUFFER_SIZE = "buffer_size"
POINTER = "pointer"
COUNT = "count"
DTYPE = "dtype"
RNG = "rng"
BURN_IN_WINDOW = "burn_in_window"

CURR_EPISODE_LENGTH = "curr_episode_length"
EPISODE_IDXES = "episode_idxes"
EPISODE_LENGTHS = "episode_lengths"
EPISODE_START_IDXES = "episode_start_idxes"

CONST_DEFAULT = "default"
CONST_MEMORY_EFFICIENT = "memory_efficient"
CONST_TRAJECTORY = "trajectory"
VALID_BUFFER = [CONST_DEFAULT, CONST_MEMORY_EFFICIENT, CONST_TRAJECTORY]

DEFAULT_LOAD_BUFFER_KWARGS = {
    "buffer_size": 0,
    "obs_dim": (0,),
    "h_state_dim": (0,),
    "act_dim": (0,),
    "rew_dim": (0,),
}
