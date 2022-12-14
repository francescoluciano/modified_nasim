import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import nasim
from nasim.scenarios.generator import ScenarioGenerator

def update_actions(env):
    env.action_space.actions = [action for action in env.action_space.actions if env.current_state.host_discovered(action.target)] #FL
    env.action_space.n = len(env.action_space.actions) #FL
    print("Actions: " + str(env.action_space.actions)) #FL
    print("n: " + str(env.action_space.n)) #FL


if __name__ == "__main__":

    # Change the name of the scenario here to change
    # the testing scenario
    num_hosts = 6
    num_services = 8
    env = nasim.generate(num_hosts=num_hosts, num_services=num_services, fully_obs=True, flat_actions=False)

    # Build the PPO algorithm. Change the timesteps here.
    # Change the name of the saved policy.
    model = PPO("MlpPolicy", env,  num_hosts=num_hosts, num_services=num_services, verbose=1, ent_coef=0.025)
    model.learn(total_timesteps=10)
    model.save("generated_6_8_test")

    del model # remove to demonstrate saving and loading

    model = PPO.load("generated_0_8_test", num_hosts=num_hosts, num_services=num_services)

    # Init the environment
    obs = env.reset()
    update_actions(env)

    # Init variables
    total_reward = 0
    number_of_steps = 0
    done = False
    
    # Do the test
    while done != True:
        action, _states = model.predict(obs)
        number_of_steps = number_of_steps + 1
        obs, rewards, done, info = env.step(action)
        print("Action: " + str(action) + ", reward: " + str(rewards))
        total_reward += rewards
        print("Total reward: " + str(total_reward))
        print(done)
        print("Steps: " + str(number_of_steps))
        env.render(mode='readable')
