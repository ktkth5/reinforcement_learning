import retro

def main():
    env = retro.make(game='Airstriker-Genesis', state='Level1')
    obs = env.reset()
    print(obs.shape) # (224,320,3) to (3,224,320)
    while True:
        # print(env.action_space.sample().shape)
        obs, rew, done, info = env.step(env.action_space.sample())
        obs = obs.transpose(2,0,1)
        env.render()
        if done:
            obs = env.reset()

if __name__ == '__main__':
    main()