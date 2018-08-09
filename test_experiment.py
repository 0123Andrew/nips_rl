import os, random
import argparse
from ast import literal_eval
import numpy as np

from model import build_model, build_model_test, Agent
from environments import RunEnv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy', dest='accuracy', action='store', default=5e-5, type=float)
    parser.add_argument('--modeldim', dest='modeldim', action='store', default='2D', choices=('3d', '2d', '3D', '2D'), type=str)
    parser.add_argument('--prosthetic', dest='prosthetic', action='store', default=False, type=bool)
    parser.add_argument('--difficulty', dest='difficulty', action='store', default=0, type=int)
    parser.add_argument('--episodes', type=int, default=10, help="Number of test episodes.")
    parser.add_argument('--critic_layers', type=str, default='(512,512)', help="critic hidden layer sizes as tuple")
    parser.add_argument('--actor_layers', type=str, default='(512,512)', help="actor hidden layer sizes as tuple")
    parser.add_argument('--layer_norm', action='store_true', help="Use layer normalization.")
    parser.add_argument('--weights', type=str, default=None, help='weights to load')
    args = parser.parse_args()
    args.modeldim = args.modeldim.upper()
    return args

def test_agent(args, num_test_episodes, model_params):
    env = RunEnv2(visualize=True, model=args.modeldim, prosthetic=args.prosthetic, difficulty=args.difficulty, skip_frame=3)
    test_rewards = []

    # train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = build_model(**model_params)
    # actor_fn, params_actor, params_crit, actor_lr, critic_lr = build_model(**model_params)
    actor_fn, params_actor, params_crit = build_model_test(**model_params)
    weights = [p.get_value() for p in params_actor]
    actor = Agent(actor_fn, params_actor, params_crit)
    actor.set_actor_weights(weights)
    if args.weights is not None:
        actor.load(args.weights)

    for ep in range(num_test_episodes):
        seed = random.randrange(2**32-2)
        state = env.reset(seed=seed, difficulty=0)
        test_reward = 0
        while True:
            state = np.asarray(state, dtype='float32')
            # state = np.concatenate((state,state,state))[:390]  # ndrw tmp
            action = actor.act(state)  # ndrw tmp
            # if args.prosthetic:
            #     action = np.zeros(19)  # ndrw tmp
            # else:
            #     action = np.zeros(22)  # ndrw tmp
            state, reward, terminal, _ = env._step(action)
            test_reward += reward
            if terminal:
                break
        test_rewards.append(test_reward)
    mean_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)

    global_step = 0
    test_str ='global step {}; test reward mean: {:.2f}, std: {:.2f}, all: {} '.\
        format(global_step.value, float(mean_reward), float(std_reward), test_rewards)

    print(test_str)
    with open(os.path.join('test_report.log'), 'a') as f:
        f.write(test_str + '\n')

def main():
    args = get_args()
    args.critic_layers = literal_eval(args.critic_layers)
    args.actor_layers = literal_eval(args.actor_layers)

    if args.prosthetic:
        num_actions = 19
    else:
        num_actions = 22

    env = RunEnv2(model=args.modeldim, prosthetic=args.prosthetic, difficulty=args.difficulty, skip_frame=3)
    env.change_model(args.modeldim, args.prosthetic, args.difficulty)
    state = env.reset(seed=42, difficulty=0)
    # obs = env.get_observation()
    d = env.get_state_desc()
    state_size = len(env.dict_to_vec(d))
    del env

    model_params = {
        'state_size': state_size,
        'num_act': num_actions,
        'gamma': 0,
        'actor_layers': args.actor_layers,
        'critic_layers': args.critic_layers,
        'actor_lr': 0,
        'critic_lr': 0,
        'layer_norm': args.layer_norm
    }

    test_agent(args, args.episodes, model_params)

if __name__ == '__main__':
    main()

