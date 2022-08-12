import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import nasim

def update_actions(env):
    env.action_space.actions = [action for action in env.action_space.actions if env.current_state.host_discovered(action.target)] #FL
    env.action_space.n = len(env.action_space.actions) #FL
    print("Actions: " + str(env.action_space.actions)) #FL
    print("n: " + str(env.action_space.n)) #FL


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("env_name", type=str, help="benchmark scenario name")
    # parser.add_argument("--render_eval", action="store_true",
    #                     help="Renders final policy")
    # parser.add_argument("--lr", type=float, default=0.001,
    #                     help="Learning rate (default=0.001)")
    # parser.add_argument("-t", "--training_steps", type=int, default=10000,
    #                     help="training steps (default=10000)")
    # parser.add_argument("--batch_size", type=int, default=32,
    #                     help="(default=32)")
    # parser.add_argument("--seed", type=int, default=0,
    #                     help="(default=0)")
    # parser.add_argument("--replay_size", type=int, default=100000,
    #                     help="(default=100000)")
    # parser.add_argument("--final_epsilon", type=float, default=0.05,
    #                     help="(default=0.05)")
    # parser.add_argument("--init_epsilon", type=float, default=1.0,
    #                     help="(default=1.0)")
    # parser.add_argument("-e", "--exploration_steps", type=int, default=10000,
    #                     help="(default=10000)")
    # parser.add_argument("--gamma", type=float, default=0.99,
    #                     help="(default=0.99)")
    # parser.add_argument("--quite", action="store_false",
    #                     help="Run in Quite mode")
    # args = parser.parse_args()

    env = gym.make("nasim:Tiny-v2")
    # update_actions(env)
    # print(env.action_space.actions)
    # print(env.action_space.n)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("a2c_nasim_test")

    del model # remove to demonstrate saving and loading

    model = A2C.load("a2c_nasim_test")

    obs = env.reset()
    update_actions(env)

    total_reward = 0
    number_of_steps = 0

    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    print(total_reward)
    print(dones)
    number_of_steps = number_of_steps + 1
    env.render(mode='readable')
    
    while dones != True:
        action, _states = model.predict(obs)
        number_of_steps = number_of_steps + 1
        obs, rewards, dones, info = env.step(action)
        print("Action: " + str(action) + ", reward: " + str(rewards))
        total_reward += rewards
        print("Total reward: " + str(total_reward))
        print(dones)
        print("Steps: " + str(number_of_steps))
        env.render(mode='readable')
