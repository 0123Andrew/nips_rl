import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['THEANO_FLAGS'] = 'device=cpu'
os.nice(9)

import argparse
from ast import literal_eval
import numpy as np
from model import build_model, Agent
from time import sleep
from multiprocessing import Process, cpu_count, Value, Queue
import queue
from memory import ReplayMemory
from agent import run_agent
# from state import StateVelCentr
from state import NormState
import lasagne
import random
from environments import RunEnv2
from datetime import datetime
from time import time
import config
import shutil

#  cd /vol/atari-rl/prj/nips_rl/
#  python run_experiment.py --n_threads 24 --weights weights/06082018-1312/weights_updates_21390_reward_78.8_pelvis_X_1.0.pkl
#  python run_experiment.py --n_threads 22

def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--modeldim', dest='modeldim', action='store', default='2D', choices=('3d', '2d', '3D', '2D'), type=str)
    parser.add_argument('--prosthetic', dest='prosthetic', action='store', default=False, type=bool)
    parser.add_argument('--difficulty', dest='difficulty', action='store', default=0, type=int)
    parser.add_argument('--gamma', type=float, default=0.95, help="Discount factor for reward.")
    parser.add_argument('--n_threads', type=int, default=cpu_count(), help="Number of agents to run (n_threads - 2).")
    parser.add_argument('--sleep', type=int, default=0.2, help="Sleep time in seconds before start each worker.")
    parser.add_argument('--max_steps', type=int, default=10000000, help="Number of steps.")
    parser.add_argument('--test_period_min', default=30, type=int, help="Test interval int min.")
    parser.add_argument('--save_period_min', default=30, type=int, help="Save interval int min.")
    parser.add_argument('--num_test_episodes', type=int, default=1, help="Number of test episodes.")
    parser.add_argument('--batch_size', type=int, default=200, help="Batch size.")
    parser.add_argument('--start_train_steps', type=int, default=5000, help="Number of steps to start training.")
    parser.add_argument('--critic_lr', type=float, default=2e-3, help="critic learning rate")
    parser.add_argument('--critic_lr_end', type=float, default=5e-5, help="critic learning rate")
    parser.add_argument('--actor_lr', type=float, default=1e-3, help="actor learning rate.")
    parser.add_argument('--actor_lr_end', type=float, default=5e-5, help="actor learning rate.")
    parser.add_argument('--critic_layers', type=str, default='(512,512)', help="critic hidden layer sizes as tuple")
    parser.add_argument('--actor_layers', type=str, default='(512,512)', help="actor hidden layer sizes as tuple")
    parser.add_argument('--flip_prob', type=float, default=0., help="Probability of flipping.")
    parser.add_argument('--layer_norm', action='store_true', help="Use layer normalization.")
    parser.add_argument('--param_noise_prob', type=float, default=0.4, help="Probability of parameters noise.")
    parser.add_argument('--exp_name', type=str, default=datetime.now().strftime("%d%m%Y-%H%M"), help='Experiment name')
    parser.add_argument('--weights', type=str, default=None, help='weights to load')
    args = parser.parse_args()
    args.modeldim = args.modeldim.upper()
    return args


def test_agent(args, testing, num_test_episodes, model_params, weights, best_reward, updates, global_step, save_dir):
    env = RunEnv2(model=args.modeldim, prosthetic=args.prosthetic, difficulty=args.difficulty, skip_frame=3)
    test_rewards_all = []
    test_pelvis_X_all = []

    train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = build_model(**model_params)
    actor = Agent(actor_fn, params_actor, params_crit)
    actor.set_actor_weights(weights)
    # if args.weights is not None:
    #     actor.load(args.weights)

    for ep in range(num_test_episodes):
        seed = random.randrange(2**32-2)
        state = env.reset(seed=seed, difficulty=0)
        test_reward = 0
        while True:
            state = np.asarray(state, dtype='float32')
            action = actor.act(state)
            state, reward, terminal, info = env._step(action)
            test_reward += reward
            if terminal:
                break
        test_rewards_all.append(test_reward)
        test_pelvis_X_all.append(info['pelvis_X'])
    test_reward_mean = np.mean(test_rewards_all)
    mean_pelvis_X = np.mean(test_pelvis_X_all)
    std_reward = np.std(test_rewards_all)

    test_str ='global step {}; test_reward_mean: {:.2f}, test_rewards_all: {}; mean_pelvis_Xmean: {:.2f}, test_pelvis_X_all: {} '.\
        format(global_step.value, float(test_reward_mean), test_rewards_all, float(mean_pelvis_X), test_pelvis_X_all)

    print(test_str)
    try:
        with open(os.path.join(save_dir, 'test_report.log'), 'a') as f:
            f.write(test_str + '\n')
    except:
        print('#############################################')
        print('except  »  f.write(test_str )')
        print('#############################################')

    if test_reward_mean > best_reward.value or test_reward_mean > 30 * env.reward_mult:
        if test_reward_mean > best_reward.value:
            best_reward.value = test_reward_mean
        fname = os.path.join(save_dir, 'weights_updates_{}_reward_{:.1f}_pelvis_X_{:.1f}.pkl'.format(updates.value, test_reward_mean, mean_pelvis_X))
        actor.save(fname)
    testing.value = 0


def main():
    args = get_args()
    args.critic_layers = literal_eval(args.critic_layers)
    args.actor_layers = literal_eval(args.actor_layers)

    # create save directory
    save_dir = os.path.join('weights', args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.move(save_dir, save_dir + '.backup')
        os.makedirs(save_dir)

    # state_transform = StateVelCentr(obstacles_mode='standard', exclude_centr=True, vel_states=[])
    # num_actions = 18
    # state_transform = NormState(args.prosthetic)
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

    # build model
    model_params = {
        'state_size': state_size,
        'num_act': num_actions,
        'gamma': args.gamma,
        'actor_layers': args.actor_layers,
        'critic_layers': args.critic_layers,
        'actor_lr': args.actor_lr,
        'critic_lr': args.critic_lr,
        'layer_norm': args.layer_norm
    }
    print('building model')
    train_fn, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = build_model(**model_params)
    actor = Agent(actor_fn, params_actor, params_crit)

    if args.weights is not None:
        actor.load(args.weights)  # set_actor_weights & set_crit_weights

    actor_lr_step = (args.actor_lr - args.actor_lr_end) / args.max_steps
    critic_lr_step = (args.critic_lr - args.critic_lr_end) / args.max_steps

    # build actor
    weights = [p.get_value() for p in params_actor]

    # build replay memory
    memory = ReplayMemory(state_size, num_actions, 5000000)

    # init shared variables
    global_step = Value('i', 0)
    updates = Value('i', 0)
    best_reward = Value('f', -1e8)
    testing = Value('i', 0)

    # init agents
    data_queue = Queue()
    workers = []
    weights_queues = []
    num_agents = args.n_threads - 2
    print('starting {} agents'.format(num_agents))
    for i in range(num_agents):
        w_queue = Queue()
        worker = Process(target=run_agent,
                         args=(args, model_params, weights, data_queue, w_queue,
                               i, global_step, updates, best_reward, args.param_noise_prob, save_dir, args.max_steps)
                         )
        worker.daemon = True
        worker.start()
        sleep(args.sleep)
        workers.append(worker)
        weights_queues.append(w_queue)

    prev_steps = 0
    start_save = time()
    start_test = time()
    weights_rew_to_check = []
    while global_step.value < args.max_steps:

        # get all data
        try:
            i, batch, weights_check, reward = data_queue.get_nowait()
            if weights_check is not None:
                weights_rew_to_check.append((weights_check, reward))
            weights_queues[i].put(weights)
            # add data to memory
            memory.add_samples(*batch)
        except queue.Empty:
            pass

        # training step
        # TODO: consider not training during testing model
        if len(memory) > args.start_train_steps:
            batch = memory.random_batch(args.batch_size)

            # if np.random.rand() < args.flip_prob:
            #     states, actions, rewards, terminals, next_states = batch
            #
            #     states_flip = state_transform.flip_states(states)
            #     next_states_flip = state_transform.flip_states(next_states)
            #     actions_flip = np.zeros_like(actions)
            #     actions_flip[:, :num_actions//2] = actions[:, num_actions//2:]
            #     actions_flip[:, num_actions//2:] = actions[:, :num_actions//2]
            #
            #     states_all = np.concatenate((states, states_flip))
            #     actions_all = np.concatenate((actions, actions_flip))
            #     rewards_all = np.tile(rewards.ravel(), 2).reshape(-1, 1)
            #     terminals_all = np.tile(terminals.ravel(), 2).reshape(-1, 1)
            #     next_states_all = np.concatenate((next_states, next_states_flip))
            #     batch = (states_all, actions_all, rewards_all, terminals_all, next_states_all)

            actor_loss, critic_loss = train_fn(*batch)
            updates.value += 1
            if np.isnan(actor_loss):
                raise Value('actor loss is nan')
            if np.isnan(critic_loss):
                raise Value('critic loss is nan')
            target_update_fn()
            weights = actor.get_actor_weights()

        delta_steps = global_step.value - prev_steps
        prev_steps += delta_steps

        actor_lr.set_value(lasagne.utils.floatX(max(actor_lr.get_value() - delta_steps*actor_lr_step, args.actor_lr_end)))
        critic_lr.set_value(lasagne.utils.floatX(max(critic_lr.get_value() - delta_steps*critic_lr_step, args.critic_lr_end)))

        # check if need to save and test
        if (time() - start_save)/60. > args.save_period_min:
            fname = os.path.join(save_dir, 'weights_updates_{}.pkl'.format(updates.value))
            actor.save(fname)
            start_save = time()

        # start new test process
        weights_rew_to_check = [(w, r) for w, r in weights_rew_to_check if r > best_reward.value and r > 0]
        weights_rew_to_check = sorted(weights_rew_to_check, key=lambda x: x[1])
        if ((time() - start_test) / 60. > args.test_period_min or len(weights_rew_to_check) > 0) and testing.value == 0:
            testing.value = 1
            print('start test')
            if len(weights_rew_to_check) > 0:
                _weights, _ = weights_rew_to_check.pop()
            else:
                _weights = weights
            worker = Process(target=test_agent,
                             args=(args, testing, args.num_test_episodes,
                                   model_params, _weights, best_reward,
                                   updates, global_step, save_dir)
                             )
            worker.daemon = True
            worker.start()
            start_test = time()

    # end all processes
    for w in workers:
        w.join()


if __name__ == '__main__':
    main()
