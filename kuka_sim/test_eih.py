import torch
from kuka_eih import KukaCamEnv1, KukaCamEnv2, KukaCamEnv3
from agent import base1, base2, base3, np_to_tensor, opt_cuda, base3_ensemble
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import argparse

def test_critic(task, log, base_ratio=1.0, label='', render=False, n_episodes=1, mode='de', use_fast=True, mean=True):
    log_dir = 'saves/t' + str(task) + label + '/' + str(log)
    with open(log_dir + '/critic.pt', 'rb') as fc:
        critic = torch.load(fc, map_location=torch.device('cpu'))
    with open(log_dir + '/actor.pt', 'rb') as fa:
        actor = torch.load(fa, map_location=torch.device('cpu'))
    if task == 1:
        env = KukaCamEnv1(renders=render, image_output=not use_fast, mode=mode, width=128)
        base = base1
    elif task == 2:
        env = KukaCamEnv2(renders=render, image_output=not use_fast, mode=mode, width=128)
        base = base2
    else:
        env = KukaCamEnv3(renders=render, image_output=not use_fast, mode=mode, width=128)
        base = base3
    for n in range(n_episodes):
        o, s = env.reset()
        frame = 0
        R = 0
        q_a_record = []
        q_b_record = []
        while True:
            a = base(s)
            if not use_fast:
                o_t = torch.tensor(o).type(torch.FloatTensor).unsqueeze(dim=0)
                s_t = torch.tensor(s[:8]).type(torch.FloatTensor).unsqueeze(dim=0)
            s = torch.tensor(s).type(torch.FloatTensor).unsqueeze(dim=0)
            with torch.no_grad():
                if not use_fast:
                    action = actor(o_t, s_t, mean=mean)
                else:
                    action = actor(s, mean=mean)
                q_a_record.append(critic(s, action).item())
                q_b_record.append(critic(s, torch.tensor(a).type(torch.FloatTensor).unsqueeze(dim=0)))
                action = action.squeeze().numpy()
                print(action)
            if np.random.uniform(0, 1) < base_ratio:
                o_next, s_next, r, done = env.step(a)
            else:
                o_next, s_next, r, done = env.step(action)
            s = s_next
            o = o_next
            R += r
            frame += 1
            if done or frame >= 100:
                print('episode', n + 1, 'ends in', frame, 'frames, return =', R)
                plt.plot(q_a_record, label='agent')
                plt.plot(q_b_record, c='gray', alpha=0.5, label='base')
                plt.legend()
                plt.show()
                break


def test_actor(task, log, n_episodes=100, label='', base_ratio=1.0, render=False, mode='de', use_fast=True):
    print("inloading")
    with open('saves/t' + str(task) + label + '_eih/' + str(log) + '/actor_best.pt', 'rb') as fa:
        actor = opt_cuda(torch.load(fa, map_location=torch.device('cpu')), 0)
    print("finish loading")
    if task == 1:
        env = KukaCamEnv1(renders=render, image_output=not use_fast, mode=mode, width=128)
        base = base1
    elif task == 2:
        env = KukaCamEnv2(renders=render, image_output=not use_fast, mode=mode, width=128)
        base = base2
    else:
        env = KukaCamEnv3(renders=render, image_output=not use_fast, mode=mode, width=128)
        base = base3
    success_count = 0
    sum_L = 0
    misbehavior_count = 0
    print("*******************************************")
    for n in range(n_episodes):
        o, s = env.reset()
        frame = 0
        R = 0
        while True:
            if np.random.uniform(0, 1) < base_ratio:
                o_next, s_next, r, done = env.step(base(s))
            else:
                if not use_fast:
                    o_t = np_to_tensor(o, actor.device).unsqueeze(dim=0)
                    s_t = np_to_tensor(s[:8], actor.device).unsqueeze(dim=0)
                s = np_to_tensor(s, actor.device).unsqueeze(dim=0)
                with torch.no_grad():
                    if not use_fast:
                        a = actor(o_t, s_t)
                    else:
                        a = actor(s)
                a = a.cpu().squeeze().numpy()
                o_next, s_next, r, done = env.step(a)
            s = s_next
            o = o_next
            R += r
            frame += 1
            #if frame == 1 or frame == 30 or done:
            #    time.sleep(10)
            if done or frame >= 100:
                #print('episode', n+1, 'ends in', frame, 'frames, return =', R)
                if done:
                    if R == 1:
                        sum_L += frame
                        success_count += 1
                    else:
                        misbehavior_count += 1
                break
    print('saves/t', task, label,log)
    print('Average time in executing the task is', sum_L / success_count, ';\n'
          'Success rate in', n_episodes, 'episodes is', success_count / n_episodes, ';\n'
          'Misbehavior rate in', n_episodes, 'episodes is', misbehavior_count / n_episodes, ';\n')
    print("*******************************************")
    return sum_L / success_count, success_count / n_episodes, misbehavior_count / n_episodes


def test_base(task = 1,n_episodes=1000, render=True, add_noise=False):
    if task == 1:
        env = KukaCamEnv1(renders=render,image_output = False)
        base = base1
    elif task == 2:
        env = KukaCamEnv2(renders=render, image_output=False)
        base = base2
    elif task == 3:
        env = KukaCamEnv3(renders=render, image_output=False)
        base = base3

    success_count = 0
    sum_L = 0
    misbehavior_count =0
    print("*******************************************")
    for n in range(n_episodes):
        o, s = env.reset()
        frame = 0
        R = 0
        while True:
            print(s)
            a = base(s)
            if add_noise:
                a += 0.1 * np.random.normal(0, 1, 5)
            o_next, s_next, r, done = env.step(a)
            s = s_next
            frame += 1
            R += r
            if done or frame >= 100:
                print('episode', n+1, 'ends in', frame, 'frames, return =', R)
                if done:
                    if R == 1:
                        sum_L += frame
                        success_count += 1
                    else:
                        misbehavior_count += 1
                break

    print('Average time in executing the task is', sum_L / success_count, ';\n'
                                                                          'Success rate in', n_episodes,
          'episodes is', success_count / n_episodes, ';\n'
                                                     'Misbehavior rate in', n_episodes, 'episodes is',
          misbehavior_count / n_episodes, ';\n')
    print("*******************************************")
            #return sum_L / success_count, success_count / n_episodes, misbehavior_count / n_episodes




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-l', '--log', type=int, default=1)
    parser.add_argument('-a', '--imitate', action='store_true')
    parser.add_argument('-t', '--task', type=int, default=1)
    parser.add_argument('-q', '--mixed_q', action='store_true')
    parser.add_argument('-b', '--label', type=str, default='')
    args = parser.parse_args()
    print("in test")
    test_actor(task=args.task, log=args.log, label=args.label, render=False, base_ratio=0.0, n_episodes=1000, mode='de',use_fast=False)