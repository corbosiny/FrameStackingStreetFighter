import retro
import numpy
import time
from Discretizer import StreetFighter2Discretizer
def main(game= 'StreetFighterIISpecialChampionEdition-Genesis',  state= "single_player_ryuVSchunli"):
    #env = retro.make(game= game, state= state, players= 1, use_restricted_actions= retro.Actions.MULTI_DISCRETE)
    env = retro.make(game= game, state= state, players= 2)
    env = StreetFighter2Discretizer(env, env.players)
    print(env.action_space)
    input()
    obs = env.reset()
    while True:
        actionList = [env.action_space.sample() for i in range(env.players)]
        obs, rew, done, info = env.step(actionList)
        #print(env.get_action_meaning(actionList))
        print(rew)
        if rew[0] != 0: input()
        env.render()
        time.sleep(.0001)
        if done:
            obs = env.reset()

if __name__ == "__main__":
    main()
