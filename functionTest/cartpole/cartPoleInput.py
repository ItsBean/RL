import gym

env = gym.make('CartPole-v1', render_mode="human")
env.reset()
for _ in range(500):
    env.render()

    action = input("input " + _.__str__() + " :")  # env.action_space.sample()
    if (action == ""): continue
    action = int(action)
    if (action % 10 == 0):
        action = 0
    else:
        action = 1

    observation, reward, done, info, unknown = env.step(action)  # take a random action
    print(action, observation, reward, sep=" , ")
    if done:
        break
env.close()
