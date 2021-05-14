import retro
import numpy
import time
import gym
from Discretizer import StreetFighter2Discretizer

def main(game= 'StreetFighterIISpecialChampionEdition-Genesis',  state= "two_player_ryuVSguile"):
    env = retro.make(game= game, state= state, players= 2)
    env = StreetFighter2Discretizer(env)
    env.reset()
    while True:
        action = [env.action_space.sample() for i in range(2)]
        env.step(action)
        env.render()
if __name__ == "__main__":
    main()