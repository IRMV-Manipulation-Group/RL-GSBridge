import torch
from kuka_eih_robotiq import KukaCamEnv1, KukaCamEnv2, KukaCamEnv3, KukaCamEnv4
#import agent
from SAC_agent_robotiq import  np_to_tensor, opt_cuda, base3_ensemble, Base2, Base4, Base3
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import argparse
import cv2
from gs_rendering import SimGaussian
import json

def test_actor(n_episodes=100, render=False, mode='de', use_fast=True, use_render=False, simmodel=None, load_mesh=False, bg_pos=None, refine = False, fov = 50, width = 140, obj_infos=None):
    
    env = KukaCamEnv4(renders=render, image_output=not use_fast, mode=mode, loadmesh=load_mesh, bg_pos=bg_pos, fov = fov, width=width, obj_infos=obj_infos)
    base = Base4(refine=False)

    success_count = 0
    sum_L = 0
    misbehavior_count = 0
    print("*******************************************")
    for n in range(n_episodes):
        if refine:
            ### domain refine
            scale1 = np.random.uniform(0.9, 1.1)
            scale2 = np.random.uniform(scale1-0.05, scale1+0.05)
            o, s, simdata = env.reset_quad(math.floor(n*4/n_episodes)+1, scale1=scale1, scale2=scale2)
            if simmodel is not None:
                simmodel.reset_scale([scale1, scale2])
        else:
            if n < 11:
                quad = 1
            elif n < 22:
                quad = 2
            elif n < 27:
                quad = 3
            else:
                quad = 4
            #o, s, simdata = env.reset_quad(quad)
            o, s, simdata = env.reset_quad(math.floor(n*4/n_episodes)+1)
            #o, s, simdata = env.reset_fix([0.60, -0.05, 0.044], [0,0,0],[0.4, 0.15,0.027],[0,0,0.021957018536702427])
    
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
            #render_img = cv2.rotate(render_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            o[0] = render_img[:, :, 0]
            o[1] = render_img[:, :, 1]
            o[2] = render_img[:, :, 2]
            o[3] = render_img[:, :, 0]
            o[4] = render_img[:, :, 1]
            o[5] = render_img[:, :, 2]
        frame = 0
        R = 0
        while True:
            action = [0,0,0,0,0]
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
                #print()
                render_img = render_img[:, :, 80:560]
                render_img = cv2.resize(np.transpose(render_img.detach().cpu().numpy(), (1, 2, 0))*255, (128, 128), interpolation=cv2.INTER_LINEAR)
                render_img = mask_invert*render_img + grip_img
                #render_img = cv2.rotate(render_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                import os
                output_path = './test_out/render_test'
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                    os.mkdir(output_path+'/GS')
                    os.mkdir(output_path+'/gt')
                #torchvision.utils.save_image(render_img/255, os.path.join(output_path, 'GS', '{0:05d}'.format(t) + ".png"))
                #print(render_img.shape)
                save_img = cv2.cvtColor(render_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_path, 'GS', '{0:05d}'.format(frame) + ".png"), save_img)
                #rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(os.path.join(output_path, 'gt', '{0:05d}'.format(frame) + ".png"), cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR))
                
                o_next[0] = render_img[:, :, 0]
                o_next[1] = render_img[:, :, 1]
                o_next[2] = render_img[:, :, 2]
                o_next[3] = render_img[:, :, 0]
                o_next[4] = render_img[:, :, 1]
                o_next[5] = render_img[:, :, 2]
            s = s_next
            o = o_next
            R += r
            frame += 1
            #if frame == 1 or frame == 30 or done:
            #    time.sleep(10)
        
    return 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-w', '--width', type=int, default=128)
    parser.add_argument('-r', '--render', action='store_true')
    parser.add_argument('--mesh', action='store_true')
    args = parser.parse_args()
    print("in test")
    use_render = args.render
    if use_render:
        obj_names = ['bg_meshGS', 'bear', 'cake']
        obj_path = []
        obj_scale = []
        obj_trans = []
        obj_after_pos = []
        obj_heights = []
        obj_urdfs = []
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
                obj_urdfs.append(json_data[obj_name]['urdf'])
        
        if 'bg_meshGS' in obj_names:
            degree_list = [3, 0, 0]
        else:
            degree_list = [0, 0, 0]

        params = {
            'model_list': obj_path, 
            'convert_SHs_python':False, 
            'white_background':True, 
            'obj_scale_list': obj_scale,
            'init_trans_list': obj_trans,# [trans_bg, trans_small_cube, trans_cake],  #### GS cord to object center cord
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

    test_actor(render=True, n_episodes=32, mode='de',use_fast=False, use_render=use_render, simmodel=simmodel, load_mesh=args.mesh, bg_pos = bg_pos, refine=False, fov = fov, width = args.width, obj_infos = [obj_urdfs, obj_heights])