import numpy as np
import torch
from kuka_eih_robotiq import KukaCamEnv1, KukaCamEnv2, KukaCamEnv3, KukaCamEnv4
from SAC_agent_robotiq import SACWBAgent, ReplayBufferFast, Base2, Base3, Base4
import csv
import pickle
import argparse
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import os
from tqdm import tqdm
import cv2
from gs_rendering import SimGaussian
import json

# import sys
# sys.path.append('/data/dinov2')

def collect_demo(env, log_dir, n_episodes=500, task=1):
    if task == 1:
        #base = base1
        raise NotImplementedError("Task 1 not implement yet!")
    if task == 2:
        base = Base2
    if task == 3:
        base = Base3
    if task == 4:
        base = Base4

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
            if done or frame == 115:
                if R == 1:
                    demo.merge(temp)
                    count += 1
                break
    with open(log_dir + '/demo.pkl', 'wb') as fd:
        pickle.dump(demo, fd)

def train(env, agent, log_dir, start_n=0, use_render = False, simmodel = None, refine = False):
    log_file = log_dir + '/log.csv'

    ###### TODO: 从中间开始训，网络在agent里，更新需要重新写一下
    if os.path.exists(log_dir + '/critic.pt'):
        with open(log_dir + '/critic.pt', 'rb') as fc:
            critic = torch.load(fc, map_location=torch.device('cuda'))
        with open(log_dir + '/actor.pt', 'rb') as fa:
            actor = torch.load(fa, map_location=torch.device('cuda'))
        agent.load_ckpt(actor, critic)

    if os.path.exists(log_file):
        with open(log_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
    else:
        with open(log_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(['episode', 'frames', 'return', 'Lc', 'La', 'Lbc', 'ratio', 'epsilon', 'test'])
    n_episodes = 30000
    frames = 0
    test = 0
    test_old = 0
    average_frame = 0
    termi = 115 #wyx change:100
    
    if agent.use_fast:
        max_frames = 2e5
    else:
        max_frames = 5e5#4e5
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
            if frames > max_frames/2 and refine:
                ### domain refine
                scale1 = np.random.uniform(0.9, 1.1)
                scale2 = np.random.uniform(scale1-0.05, scale1+0.05)
                o, s, simdata = env.reset(scale1=scale1, scale2=scale2)
                if simmodel is not None:
                    simmodel.reset_scale([scale1, scale2])
            else:
                o, s, simdata = env.reset()
            if use_render:  
                ## render_process        
                rgb = simdata['rgb']
                rgbImg = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
                grip_mask = simdata['mask']
                extrinsic = simdata['extrinsic']
                obj_trans_list = simdata['obj_trans_list']
                obj_rot_list = simdata['obj_rot_list']
                mask_invert = 1 - grip_mask[:, :, None]
                grip_img = grip_mask[:, :, None]*rgbImg

                simmodel.update_camera(extrinsic)
                render_img = simmodel.update_and_render(obj_trans_list, obj_rot_list)
                render_img = render_img[:, :, 80:560]
                render_img = cv2.resize(np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255, (128, 128), interpolation=cv2.INTER_LINEAR)
                render_img = mask_invert*render_img + grip_img
                #render_img = mask_invert*np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255 + grip_img
                #render_img = cv2.rotate(render_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                o[0] = render_img[:, :, 0]
                o[1] = render_img[:, :, 1]
                o[2] = render_img[:, :, 2]
                o[3] = render_img[:, :, 0]
                o[4] = render_img[:, :, 1]
                o[5] = render_img[:, :, 2]
            frame = 0
            R = 0
            ratio = 0
            if frames >= max_frames:
                break
            while True:
                action, flag = agent.act(o, s)
                if flag:
                    ratio += 1
                #action += 0.1 * np.random.normal(0, 1, 5)
                action = np.clip(action, -1, 1)
                o_next, s_next, r, done, simdata = env.step(action)
                if use_render:
                    ## render_process        
                    rgb = simdata['rgb']
                    rgbImg = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
                    grip_mask = simdata['mask']
                    extrinsic = simdata['extrinsic']
                    obj_trans_list = simdata['obj_trans_list']
                    obj_rot_list = simdata['obj_rot_list']
                    mask_invert = 1 - grip_mask[:, :, None]
                    grip_img = grip_mask[:, :, None]*rgbImg

                    simmodel.update_camera(extrinsic)
                    render_img = simmodel.update_and_render(obj_trans_list, obj_rot_list)
                    render_img = render_img[:, :, 80:560]
                    render_img = cv2.resize(np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255, (128, 128), interpolation=cv2.INTER_LINEAR)
                    render_img = mask_invert*render_img + grip_img
                    #render_img = mask_invert*np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255 + grip_img
                    
                    # output_path = './test_out/RLlearntest'
                    # if not os.path.exists(output_path):
                    #     os.mkdir(output_path)
                    #     os.mkdir(output_path+'/GS')
                    #     os.mkdir(output_path+'/gt')
                    # #torchvision.utils.save_image(render_img/255, os.path.join(output_path, 'GS', '{0:05d}'.format(t) + ".png"))
                    # #print(render_img.shape)
                    # save_img = cv2.cvtColor(render_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    # cv2.imwrite(os.path.join(output_path, 'GS', '{0:05d}'.format(frame) + ".png"), save_img)
                    # cv2.imwrite(os.path.join(output_path, 'gt', '{0:05d}'.format(frame) + ".png"), cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR))

                    o_next[0] = render_img[:, :, 0]
                    o_next[1] = render_img[:, :, 1]
                    o_next[2] = render_img[:, :, 2]
                    o_next[3] = render_img[:, :, 0]
                    o_next[4] = render_img[:, :, 1]
                    o_next[5] = render_img[:, :, 2]
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

                    if (n+1) % 50 == 0:
                        success_count = 0
                        average_frames = 0
                        for _ in range(100):
                            o, s, simdata = env.reset()
                            if use_render:  
                                ## render_process        
                                rgb = simdata['rgb']
                                rgbImg = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
                                grip_mask = simdata['mask']
                                extrinsic = simdata['extrinsic']
                                obj_trans_list = simdata['obj_trans_list']
                                obj_rot_list = simdata['obj_rot_list']
                                mask_invert = 1 - grip_mask[:, :, None]
                                grip_img = grip_mask[:, :, None]*rgbImg

                                simmodel.update_camera(extrinsic)
                                render_img = simmodel.update_and_render(obj_trans_list, obj_rot_list)
                                render_img = render_img[:, :, 80:560]
                                render_img = cv2.resize(np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255, (128, 128), interpolation=cv2.INTER_LINEAR)
                                render_img = mask_invert*render_img + grip_img
                                #render_img = mask_invert*np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255 + grip_img
                                #render_img = cv2.rotate(render_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                o[0] = render_img[:, :, 0]
                                o[1] = render_img[:, :, 1]
                                o[2] = render_img[:, :, 2]
                                o[3] = render_img[:, :, 0]
                                o[4] = render_img[:, :, 1]
                                o[5] = render_img[:, :, 2]
                            frame_t = 0
                            R_t = 0
                            while True:
                                a, _ = agent.act(o, s, test=True)
                                o_next, s_next, r, done, simdata = env.step(a)
                                if use_render:
                                    ## render_process        
                                    rgb = simdata['rgb']
                                    rgbImg = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
                                    grip_mask = simdata['mask']
                                    extrinsic = simdata['extrinsic']
                                    obj_trans_list = simdata['obj_trans_list']
                                    obj_rot_list = simdata['obj_rot_list']
                                    mask_invert = 1 - grip_mask[:, :, None]
                                    grip_img = grip_mask[:, :, None]*rgbImg

                                    simmodel.update_camera(extrinsic)
                                    render_img = simmodel.update_and_render(obj_trans_list, obj_rot_list)
                                    render_img = render_img[:, :, 80:560]
                                    render_img = cv2.resize(np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255, (128, 128), interpolation=cv2.INTER_LINEAR)
                                    render_img = mask_invert*render_img + grip_img
                                    #render_img = mask_invert*np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255 + grip_img

                                    o_next[0] = render_img[:, :, 0]
                                    o_next[1] = render_img[:, :, 1]
                                    o_next[2] = render_img[:, :, 2]
                                    o_next[3] = render_img[:, :, 0]
                                    o_next[4] = render_img[:, :, 1]
                                    o_next[5] = render_img[:, :, 2]
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
                    new_line = np.array([n + 1, frames, R, av_Lc, av_La, av_Lbc, ratio / frame, agent.epsilon, test])
                    with open(log_file, "a+", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(new_line)
                    with open(log_dir + '/critic.pt', 'wb') as fc:
                        torch.save(agent.critic, fc)
                    with open(log_dir + '/actor.pt', 'wb') as fa:
                        torch.save(agent.actor, fa)
                    if test > test_old or test == test_old:
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
    parser.add_argument('-l', '--log', type=str, default='1')
    parser.add_argument('-i', '--use_image', action='store_true')
    parser.add_argument('-t', '--task', type=int, default=2)
    parser.add_argument('-q', '--mixed_q', action='store_true')
    parser.add_argument('-b', '--base_boot', action='store_true')
    parser.add_argument('-c', '--behavior_clone', action='store_true')
    parser.add_argument('--color_refine', action='store_true')
    parser.add_argument('--strain', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--use_force', action='store_true')
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('--mesh', action='store_true')
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()
    exp = ['vanilla', 'wMQ', 'wBB', 'nBC', 'wBC', 'nBB', 'nMQ', '']

    use_render = args.render
    if use_render:
        obj_names = ['bg_meshGS', 'banana', 'cake'] ## here choose the object and bgs
        obj_path = []
        obj_scale = []
        obj_trans = []
        obj_after_pos = []
        obj_heights = []
        with open('./obj_trans.json','r',encoding='utf8')as fp:
            json_data = json.load(fp)[0]
        #print(json_data)
        for obj_name in obj_names:
            obj_path.append(json_data[obj_name]['path'])
            obj_scale.append(json_data[obj_name]['scale'])
            mesh_trans = np.array(json_data[obj_name]['mesh_trans'])
            plane_trans = np.array(json_data[obj_name]['plane_trans'])
            trans_obj = mesh_trans @ plane_trans
            obj_trans.append(trans_obj)
            if 'bg' in obj_name:
                obj_after_pos.append(np.array(json_data[obj_name]['pos_mod']))
            else:
                obj_after_pos.append(None)
                obj_heights.append(json_data[obj_name]['height'])
        
        if 'bg_meshGS' in obj_names:
            degree_list = [3, 0, 0]
        else:
            degree_list = [0, 0, 0]

        params = {
            'model_list': obj_path, 
            'convert_SHs_python':False, 
            'white_background':True, 
            'obj_scale_list': obj_scale, # [0.3, 0.075, 0.02]
            'init_trans_list': obj_trans,# [trans_bg, trans_small_cube, trans_cake],  #### 高斯模型坐标系到物体中心系的变换矩阵
            'after_pos_list': obj_after_pos,  
            'camera_setting':{
                'FovX':54.8, # 58
                'FovY':42.5, 
                'img_H':480, #args.width, 
                'img_W':640, #args.width
            }, 
            'degree_list': degree_list, 
            'shs_num':1
        }
        fov = params['camera_setting']['FovY']
        simmodel = SimGaussian(params)
        bg_pos = params['after_pos_list'][0]
    else:
        simmodel = None
        bg_pos = None
        fov = 50

    idx = args.mixed_q + args.base_boot * 2 + args.behavior_clone * 4
    if args.task == 1:
        env = KukaCamEnv1(renders=False, image_output=args.use_image, mode=args.mode, width=args.width, loadmesh = args.mesh)
    elif args.task == 2:
        env = KukaCamEnv2(renders=False, image_output=args.use_image, mode=args.mode, width=args.width, loadmesh = args.mesh, bg_pos=bg_pos, fov=fov)
    elif args.task == 3:
        env = KukaCamEnv3(renders=False, image_output=args.use_image, mode=args.mode, width=args.width, loadmesh = args.mesh, bg_pos=bg_pos, fov=fov)
    elif args.task == 4:
        env = KukaCamEnv4(renders=False, image_output=args.use_image, mode=args.mode, width=args.width, loadmesh = args.mesh, bg_pos=bg_pos, fov=fov)
   
    if args.use_image:
        log_dir = 'saves/SAC_t' + str(args.task) + 'i_eih/' + args.log#str(args.log)
    else:
        log_dir = 'saves/SAC_t' + str(args.task) + exp[idx] + '/' + args.log#str(args.log)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    start_n = args.start
    agent = SACWBAgent(log_dir=log_dir, mode=args.mode, width=args.width, device=args.gpu, use_fast=not args.use_image,
                    task=args.task, mixed_q=args.mixed_q, base_boot=args.base_boot,
                    behavior_clone=args.behavior_clone, Qb_strain=args.strain, color_jitter=args.color_refine, pretrain=args.pretrain, use_force=args.use_force)#, refine = args.refine)
    
    train(env, agent, log_dir=log_dir, start_n = start_n, use_render = use_render, simmodel = simmodel, refine = args.refine)
