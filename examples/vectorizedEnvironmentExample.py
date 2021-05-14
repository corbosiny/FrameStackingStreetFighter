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
    num_envs = 20
    num_players_per_env = 2
    envs = [makeEnv for i in range(num_envs)]
    actionSpace = makeEnv().action_space
    env = SubprocVecEnv(envs)
    env.reset()
    gameFinished = [False] * num_envs
    while not all(gameFinished):
        inputs = []
        for game in range(num_envs):
            if not gameFinished[game]: inputs.append([actionSpace.sample() for player in range(num_players_per_env)])
            else: inputs.append([0] * num_players_per_env)

        _, _, done, info = env.step(inputs)
        
        gameFinished = [gameElem or doneElem for gameElem, doneElem in list(zip(gameFinished, done))]
        time.sleep(.0001)

if __name__ == "__main__":
    main()