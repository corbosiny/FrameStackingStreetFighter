import retro

def main(game= 'StreetFighterIISpecialChampionEdition-Genesis',  state= "chunli", record= 'test'):
    env = retro.make(game= game, state= state)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        env.render()
    env.close()
    env.viewer.close()

if __name__ == "__main__":
    main()
