"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
All credit too open-ai's examples: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
"""

import gym
import numpy as np
import retro

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        assert(type(env) == 'retro.retro_env.RetroEnv')
        assert(isinstance(env.action_space, gym.spaces.MultiBinary))
        assert(type(combos) == list)
        
        super().__init__(env)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        self._combos = combos
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        assert(type(act) == int)
        return self._decode_discrete_action[act].copy()

    def get_action_meaning(self, act):
        assert(type(act) == int)
        return self._combos[act]

class StreetFighter2Discretizer(Discretizer):
    """
    Use Street Fighter 2
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[[], 
                                         ['UP'], 
                                         ['DOWN'], 
                                         ['LEFT'], 
                                         ['UP', 'LEFT'],
                                         ['DOWN', 'LEFT'],
                                         ['RIGHT'], 
                                         ['UP', 'RIGHT'], 
                                         ['DOWN', 'RIGHT'],
                                         ['A'],
                                         ['A', 'UP'],
                                         ['A', 'DOWN'],
                                         ['A', 'LEFT'],
                                         ['A', 'RIGHT'],
                                         ['A', 'DOWN', 'LEFT'],
                                         ['A', 'DOWN', 'RIGHT'],
                                         ['B'],
                                         ['B', 'UP'],
                                         ['B', 'DOWN'],
                                         ['B', 'LEFT'],
                                         ['B', 'RIGHT'],
                                         ['B', 'DOWN', 'LEFT'],
                                         ['B', 'DOWN', 'RIGHT'],
                                         ['C'],
                                         ['C', 'UP'],
                                         ['C', 'DOWN'],
                                         ['C', 'LEFT'],
                                         ['C', 'RIGHT'],
                                         ['C', 'DOWN', 'LEFT'],
                                         ['C', 'DOWN', 'RIGHT'],
                                         ['X'],
                                         ['X', 'UP'],
                                         ['X', 'DOWN'],
                                         ['X', 'LEFT'],
                                         ['X', 'RIGHT'],
                                         ['X', 'DOWN', 'LEFT'],
                                         ['X', 'DOWN', 'RIGHT'],
                                         ['Y'],
                                         ['Y', 'UP'],
                                         ['Y', 'DOWN'],
                                         ['Y', 'LEFT'],
                                         ['Y', 'RIGHT'],
                                         ['Y', 'DOWN', 'LEFT'],
                                         ['Y', 'DOWN', 'RIGHT'],
                                         ['Z'],
                                         ['Z', 'UP'],
                                         ['Z', 'DOWN'],
                                         ['Z', 'LEFT'],
                                         ['Z', 'RIGHT'],
                                         ['Z', 'DOWN', 'LEFT'],
                                         ['Z', 'DOWN', 'RIGHT']])

"""
    Initializes an example discrete environment and randomly selects moves for the agent to make.
    The meaning of each selected move in terms of what buttons are being pressed is also displayed.
"""
def main():
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    env = StreetFighter2Discretizer(env)
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        _, _, _, _ = env.step(action)
        print(env.get_action_meaning(action))

    env.close()


if __name__ == '__main__':
    main()
