import retro
import numpy
import time
from Discretizer import StreetFighter2Discretizer
def main(game= 'StreetFighterIISpecialChampionEdition-Genesis',  state= "single_player_ryuVSchunli"):
    #env = retro.make(game= game, state= state, players= 1, use_restricted_actions= retro.Actions.MULTI_DISCRETE)
    env = retro.make(game= game, state= state, players= 1)
    env = StreetFighter2Discretizer(env)
    obs = env.reset()
    while True:
        actionList = [env.action_space.sample() for i in range(env.players)]
        obs, rew, done, info = env.step(actionList)
        env.render()
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()
