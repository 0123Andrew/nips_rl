import os
os.environ['OMP_NUM_THREADS'] = '2'

import argparse
import numpy as np
from model import build_model, Agent
from memory import ReplayMemory
from state import StateVelCentr
import lasagne
from datetime import datetime
from time import time
import Pyro4
import yaml
import sys

sys.excepthook = Pyro4.util.excepthook


def get_args():
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('--exp_name', type=str, default=datetime.now().strftime("%d.%m.%Y-%H:%M"),
                        help='Experiment name')
    parser.add_argument('--weights', type=str, default=None, help='weights to load')
    return parser.parse_args()


def find_workers(prefix):
    workers = []
    with Pyro4.locateNS() as ns:
        for sampler, sampler_uri in ns.list(prefix="{}.".format(prefix)).items():
            print("found {}".format(prefix), sampler)
            workers.append(Pyro4.Proxy(sampler_uri))
    if not workers:
        raise ValueError("no {} found!".format(prefix))
    print('found total {} {}s'.format(len(workers), prefix))
    return workers


def init_samplers(samplers, config, weights):
    results = []
    print('start samplers initialization')
    for sampler in samplers:
        res = Pyro4.Future(sampler.initialize)(config, weights)
        results.append(res)

    while len(results) > 0:
        for res in results:
            if res.ready:
                results.remove(res)
    print('finish samplers initialization')


def process_results(sampler, res, memory, weights, weights_from_samplers):
    # first set new weights
    sampler.set_actor_weights(weights)

    # start sampling process
    new_res = Pyro4.Future(sampler.run_episode)()

    # add data to memory
    memory.add_samples(res['states'], res['actions'], res['rewards'], res['terminals'])

    # add weights for tester check
    if 'weights' in res:
        weights_from_samplers.append((res['weights'], res['total_reward']))

    return new_res


def main():
    args = get_args()

    # create save directory
    save_dir = os.path.join('weights', args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read config
    with open('config.yaml') as f:
        config = yaml.load(f)

    # add savedir to config
    config['test_params']['save_dir'] = save_dir

    # save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # init state transform
    state_transform = StateVelCentr(**config['env_params']['state_transform'])

    # init model
    config['model_params']['state_size'] = state_transform.state_size
    train_fn, train_actor, train_critic, actor_fn, target_update_fn, params_actor, params_crit, actor_lr, critic_lr = \
        build_model(**config['model_params'])
    actor = Agent(actor_fn, params_actor, params_crit)
    actor.summary()
    if args.weights is not None:
        actor.load(args.weights)
    weights = [[w.tolist() for w in weights] for weights in actor.get_weights()]
    weights_actor, weights_critic = weights

    # initialize samplers
    samplers = find_workers('sampler')
    init_samplers(samplers, config, weights_actor)

    # init tester
    tester = find_workers('tester')[0]
    tester.initialize(config, weights)

    # init replay memory
    memory = ReplayMemory(state_transform.state_size, 18, **config['repay_memory'])

    # learning rate decay step
    def get_lr_step(lr, lr_end, max_steps):
        return (lr - lr_end) / max_steps

    actor_lr_step = get_lr_step(
        config['model_params']['actor_lr'],
        config['train_params']['actor_lr_end'],
        config['train_params']['max_steps']
    )
    critic_lr_step = get_lr_step(
        config['model_params']['critic_lr'],
        config['train_params']['critic_lr_end'],
        config['train_params']['max_steps']
    )

    # start sampling
    samplers_results = {s: Pyro4.Future(s.run_episode)() for s in samplers}

    # common statistic
    total_steps = 0
    prev_steps = 0
    updates = 0
    best_reward = -1e8
    weights_from_samplers = []
    samplers_weights = False
    start = time()
    start_save = start
    start_test = start
    tester_res = Pyro4.Future(tester.test_model)(weights, best_reward)

    start_train_steps_actor = config['train_params']['start_train_steps_actor']
    start_train_steps_critic = config['train_params']['start_train_steps_critic']
    start_train_steps_min = min(start_train_steps_actor, start_train_steps_critic)
    start_train_steps_max = max(start_train_steps_actor, start_train_steps_critic)

    # main train loop
    print('start train loop')
    while total_steps < config['train_params']['max_steps']:
        for s, res in samplers_results.items():
            if res.ready:
                res = res.value
                total_steps += res['steps']
                s.set_best_reward(best_reward)

                # start new job
                new_res = process_results(s, res, memory, weights[0], weights_from_samplers)
                samplers_results[s] = new_res

                # report progress on this episode
                report_str = 'Global step: {}, steps/sec: {:.2f}, updates: {}, episode len {}, ' \
                             'reward: {:.2f}, original_reward {:.4f}; best reward: {:.2f} noise {}'. \
                    format(total_steps, 1. * total_steps / (time() - start), updates,
                           res['steps'], res['total_reward'], res['total_reward_original'],
                           best_reward, 'actions' if res['action_noise'] else 'params')
                print(report_str)

        # check if enough samles and can start training
        if total_steps > start_train_steps_min:
            # batch2 if faster than batch1
            batch = memory.random_batch2(config['train_params']['batch_size'])
            states, actions, rewards, terminals, next_states = batch

            # flip states
            if np.random.rand() < config['train_params']['flip_prob']:

                states_flip = state_transform.flip_states(states)
                next_states_flip = state_transform.flip_states(next_states)
                actions_flip = np.zeros_like(actions)
                actions_flip[:, :9] = actions[:, 9:]
                actions_flip[:, 9:] = actions[:, :9]

                states = np.concatenate((states, states_flip))
                actions = np.concatenate((actions, actions_flip))
                rewards = np.tile(rewards.ravel(), 2).reshape(-1, 1)
                terminals = np.tile(terminals.ravel(), 2).reshape(-1, 1)
                next_states = np.concatenate((next_states, next_states_flip))
                batch = (states, actions, rewards, terminals, next_states)

            # choose actor or critic to train
            actor_loss = critic_loss = 0
            if total_steps > start_train_steps_max:
                actor_loss, critic_loss = train_fn(*batch)
            elif total_steps > start_train_steps_actor:
                actor_loss = train_actor(states)
            elif total_steps > start_train_steps_critic:
                critic_loss = train_critic(*batch)

            updates += 1
            target_update_fn()
            if np.isnan(actor_loss):
                raise ValueError('actor loss is nan')
            if np.isnan(critic_loss):
                raise ValueError('critic loss is nan')

            weights = [[w.tolist() for w in weights] for weights in actor.get_weights()]
            weights_actor, weights_critic = weights

            delta_steps = total_steps - prev_steps
            prev_steps += delta_steps

            actor_lr.set_value(lasagne.utils.floatX(
                max(actor_lr.get_value() - delta_steps*actor_lr_step, config['train_params']['actor_lr_end'])))
            critic_lr.set_value(lasagne.utils.floatX(
                max(critic_lr.get_value() - delta_steps*critic_lr_step, config['train_params']['critic_lr_end'])))

        # check if need to save and test
        if (time() - start_save)/60. > config['test_params']['save_period_min']:
            fname = os.path.join(save_dir, 'weights_updates_{}.h5'.format(updates))
            actor.save(fname)
            start_save = time()

        # check if can start new test process
        weights_from_samplers = [(w, r) for w, r in weights_from_samplers if r > best_reward and r > 0]
        if ((time() - start_test) / 60. > config['test_params']['test_period_min'] or len(weights_from_samplers) > 0) \
                and tester_res.ready:
            # save best reward
            test_reward, test_weights = tester_res.value
            if test_reward > best_reward:
                best_reward = test_reward
                if test_weights is not None and samplers_weights:
                    print('set sampler actor weights')
                    actor.set_actor_weights(test_weights)

            if len(weights_from_samplers) > 0:
                _weights = [weights_from_samplers.pop()[0], weights_critic]
                samplers_weights = True
            else:
                _weights = weights
                samplers_weights = False

            tester_res = Pyro4.Future(tester.test_model)(_weights, best_reward)
            start_test = time()


if __name__ == '__main__':
    main()
