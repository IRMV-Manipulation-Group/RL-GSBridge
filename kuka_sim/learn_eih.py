import numpy as np
import torch
from kuka_eih import KukaCamEnv1, KukaCamEnv2, KukaCamEnv3
from agent import WBAgent, base1, base2, base3, ReplayBufferFast
import csv
import pickle
import argparse
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import os
from tqdm import tqdm


def collect_demo(env, log_dir, n_episodes=500, task=1):
    if task == 1:
        base = base1
    if task == 2:
        base = base2
    if task == 3:
        base = base3
    demo = ReplayBufferFast(20, 5, size=n_episodes * 100)
    count = 0
    for n in range(int(n_episodes * 1.5)):
        if count == n_episodes:
            break
        o, s = env.reset()
        frame = 0
        R = 0
        temp = ReplayBufferFast(20, 5, size=100)
        while True:
            action = base(s)
            o_next, s_next, r, done = env.step(action)
            temp.store(s, action, s_next, [r], [done])
            s = s_next
            R += r
            frame += 1
            if done or frame == 100:
                if R == 1:
                    demo.merge(temp)
                    count += 1
                break
    with open(log_dir + '/demo.pkl', 'wb') as fd:
        pickle.dump(demo, fd)


def train(env, agent, log_dir, start_n=0):
    log_file = log_dir + '/log.csv'
    
    if os.path.exists(log_dir + '/critic.pt'):
        with open(log_dir + '/critic.pt', 'rb') as fc:
            critic = torch.load(fc, map_location=torch.device('cpu'))
        with open(log_dir + '/actor.pt', 'rb') as fa:
            actor = torch.load(fa, map_location=torch.device('cpu'))

    if os.path.exists(log_file):
        with open(log_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
    else:
        with open(log_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['episode', 'frames', 'return', 'Lc', 'La', 'Lbc', 'ratio', 'test'])
    n_episodes = 30000
    frames = 0
    test = 0
    test_old = 0
    average_frame = 0
    termi = 100
    if agent.ensemble:
        max_frames = 1e6
    elif agent.use_fast:
        max_frames = 2e5
    else:
        max_frames = 4e5
    max_frames = int(max_frames)
    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False,
        speed_estimate_period=300.0,
        transient=True
    )
    progress_frames = 0
    with progress:
        task = progress.add_task('[red]'+log_dir.ljust(15), total=max_frames)
        #### 3000 episodes, each episodes should reach max frames or done. So episode can be loaded again?
        for n in range(start_n, n_episodes):
            o, s = env.reset()
            frame = 0
            R = 0
            ratio = 0
            if frames >= max_frames:
                break
            while True:
                action, flag = agent.act(o, s)
                if flag:
                    ratio += 1
                action += 0.1 * np.random.normal(0, 1, 5)
                action = np.clip(action, -1, 1)
                o_next, s_next, r, done = env.step(action)
                agent.remember(o, s, action, o_next, s_next, r, done)
                o = o_next
                s = s_next
                R += r
                frame += 1
                #### if get the obj or reach max t range, train the agent; 
                #### progress is the training frames progress
                if done or frame == termi:
                    av_Lc, av_La, av_Lbc = agent.train(frame)
                    frames += frame
                    if not agent.use_fast or ((n + 1) % 5 == 0 and agent.use_fast) or frames >= max_frames:
                        advance = min(max_frames, frames)-progress_frames
                        progress.update(task, advance=advance)
                        progress.refresh()
                        progress_frames += advance

                    # if frames <= 6e5:
                    #     for p in agent.optimizer_actor.param_groups:
                    #         p['lr'] = 1e-3 * (1 - 0.9 * frames / 6e5)
                    # else:
                    #     p['lr'] = 1e-4

                    if (n+1) % 30 == 0:
                        success_count = 0
                        average_frames = 0
                        for _ in range(100):
                            o, s = env.reset()
                            frame_t = 0
                            R_t = 0
                            while True:
                                a, _ = agent.act(o, s, test=True)
                                o_next, s_next, r, done = env.step(a)
                                o = o_next
                                s = s_next
                                frame_t += 1
                                R_t += r
                                if done or frame_t == termi:
                                    if R_t == 1:
                                        success_count += 1
                                        average_frames += frame_t
                                    break
                        test = success_count / 100
                    new_line = np.array([n + 1, frames, R, av_Lc, av_La, av_Lbc, ratio / frame, test])
                    with open(log_file, "a+", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(new_line)
                    with open(log_dir + '/critic.pt', 'wb') as fc:
                        torch.save(agent.critic, fc)
                    with open(log_dir + '/actor.pt', 'wb') as fa:
                        torch.save(agent.actor, fa)
                    if test > test_old:
                        with open(log_dir + '/actor_best.pt', 'wb') as fb:
                            torch.save(agent.actor, fb)
                        test_old = test
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='de')
    parser.add_argument('-w', '--width', type=int, default=128)
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-l', '--log', type=int, default=1)
    parser.add_argument('-i', '--use_image', action='store_true')
    parser.add_argument('-a', '--imitate', action='store_true')
    parser.add_argument('-t', '--task', type=int, default=1)
    parser.add_argument('-q', '--mixed_q', action='store_true')
    parser.add_argument('-b', '--base_boot', action='store_true')
    parser.add_argument('-c', '--behavior_clone', action='store_true')
    parser.add_argument('-e', '--ensemble', action='store_true')
    args = parser.parse_args()
    exp = ['vanilla', 'wMQ', 'wBB', 'nBC', 'wBC', 'nBB', 'nMQ', '']
    idx = args.mixed_q + args.base_boot * 2 + args.behavior_clone * 4
    if args.task == 1:
        env = KukaCamEnv1(renders=False, image_output=args.use_image, mode=args.mode, width=args.width)
    elif args.task == 2:
        env = KukaCamEnv2(renders=False, image_output=args.use_image, mode=args.mode, width=args.width)
    elif args.task == 3:
        env = KukaCamEnv3(renders=False, image_output=args.use_image, mode=args.mode, width=args.width)
    if args.imitate:
        log_dir = 'saves/t' + str(args.task) + 'wD/' + str(args.log)
        # collect_demo(env, log_dir=log_dir, task=args.task)
    elif args.ensemble:
        log_dir = 'saves/t' + str(args.task) + 'e' + exp[idx] + '/' + str(args.log)
    elif args.use_image:
        log_dir = 'saves/t' + str(args.task) + 'i_eih/' + str(args.log)
    else:
        log_dir = 'saves/t' + str(args.task) + exp[idx] + '/' + str(args.log)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    start_n = args.start
    agent = WBAgent(log_dir=log_dir, mode=args.mode, width=args.width, device=args.gpu, use_fast=not args.use_image,
                    task=args.task, mixed_q=args.mixed_q, base_boot=args.base_boot,
                    behavior_clone=args.behavior_clone, imitate=args.imitate, ensemble=args.ensemble)
    train(env, agent, log_dir=log_dir, start_n = start_n)
