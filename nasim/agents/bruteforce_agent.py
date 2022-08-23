"""An bruteforce agent that repeatedly cycles through all available actions in
order.

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python bruteforce_agent.py tiny

This will run the agent and display progress and final results to stdout.

To see available running arguments:

$ python bruteforce_agent.py --help
"""

from itertools import product

import nasim
import numpy as np

LINE_BREAK = "-"*60


def run_bruteforce_agent(env, step_limit=1e6, verbose=True):
    """Run bruteforce agent on nasim environment.

    Parameters
    ----------
    env : nasim.NASimEnv
        the nasim environment to run agent on
    step_limit : int, optional
        the maximum number of steps to run agent for (default=1e6)
    verbose : bool, optional
        whether to print out progress messages or not (default=True)

    Returns
    -------
    int
        timesteps agent ran for
    float
        the total reward recieved by agent
    bool
        whether the goal was reached or not
    """
    if verbose:
        print(LINE_BREAK)
        print("STARTING EPISODE")
        print(LINE_BREAK)
        print("t: Reward")

    env.reset()
    total_reward = 0
    done = False
    steps = 0
    cycle_complete = False

    if env.flat_actions:
        act = 0
    else:
        act_iter = product(*[range(n) for n in env.action_space.nvec])

    while not done and steps < step_limit:
        if env.flat_actions:
            act = (act + 1) % env.action_space.n
            cycle_complete = (steps > 0 and act == 0)
        else:
            try:
                act = next(act_iter)
                cycle_complete = False
            except StopIteration:
                act_iter = product(*[range(n) for n in env.action_space.nvec])
                act = next(act_iter)
                cycle_complete = True

        _, rew, done, _ = env.step(act)
        total_reward += rew

        if cycle_complete and verbose:
            print(f"{steps}: {total_reward}")
        steps += 1

    if done and verbose:
        print(LINE_BREAK)
        print("EPISODE FINISHED")
        print(LINE_BREAK)
        print(f"Goal reached = {env.goal_reached()}")
        print(f"Total steps = {steps}")
        print(f"Total reward = {total_reward}")
    elif verbose:
        print(LINE_BREAK)
        print("STEP LIMIT REACHED")
        print(LINE_BREAK)

    if done:
        done = env.goal_reached()

    return steps, total_reward, done


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("-p", "--param_actions", action="store_true",
                        help="Use Parameterised action space")
    parser.add_argument("-f", "--box_obs", action="store_true",
                        help="Use 2D observation space")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="number of random runs to perform (default=1)")
    args = parser.parse_args()

    num_hosts = 12      #FL
    num_services = 8    #FL
    run_steps = []
    run_rewards = []
    run_goals = 0
    for i in range(args.runs):
        env = nasim.generate(num_hosts=num_hosts, num_services=num_services, fully_obs=True, flat_actions=False)    #FL
        steps, reward, done = run_bruteforce_agent(env, verbose=False)
        run_steps.append(steps)
        run_rewards.append(reward)
        run_goals += int(done)

        if args.runs > 1:
            print(f"Run {i}:")
            print(f"\tSteps = {steps}")
            print(f"\tReward = {reward}")
            print(f"\tGoal reached = {done}")

    run_steps = np.array(run_steps)
    run_rewards = np.array(run_rewards)

    print(LINE_BREAK)
    print("Random Agent Runs Complete")
    print(LINE_BREAK)
    print(f"Mean steps = {run_steps.mean():.2f} +/- {run_steps.std():.2f}")
    print(f"Mean rewards = {run_rewards.mean():.2f} "
          f"+/- {run_rewards.std():.2f}")
    print(f"Goals reached = {run_goals} / {args.runs}")
