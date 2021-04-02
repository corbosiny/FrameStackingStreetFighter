import retro
import numpy
import time
import gym
from Discretizer import StreetFighter2Discretizer
from stable_baselines.common.vec_env import SubprocVecEnv

def makeEnv(game= 'StreetFighterIISpecialChampionEdition-Genesis',  state= "two_player_ryuVSguile"):
    env = retro.make(game= game, state= state, players= 2)
    env = StreetFighter2Discretizer(env)
    return env 

def main():
    envs = [makeEnv for i in range(1)]
    env = SubprocVecEnv(envs)
    env.reset()
    while True:
        env.step_async([[2, 4], [5, 7]])
        env.env_method('render')
        time.sleep(.0001)

if __name__ == "__main__":
    main()