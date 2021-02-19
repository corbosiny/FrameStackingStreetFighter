import retro
import numpy
import time
from Discretizer import StreetFighter2Discretizer
def main(game= 'StreetFighterIISpecialChampionEdition-Genesis',  state= "two_player_ryuVSdhalism"):
    env = retro.make(game= game, state= state, players= 2)
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
