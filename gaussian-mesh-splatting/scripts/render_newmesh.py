#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#
import sys
sys.path.append('/data/gaussian-mesh-splatting')
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from renderer.gaussian_animated_renderer import render
import torchvision
import trimesh
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from games.mesh_splatting_norm_aug.scene.gaussian_mesh_norm_aug_model import GaussianMeshNormAugModel


def transform_ficus_sinus(vertices, t, idxs):
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 0] * 2 *torch.pi + t)  # sinus
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)  # sinus
    return vertices


def transform_hotdog_fly(vertices, t, idxs):
    vertices_new = vertices.clone()
    f = torch.sin(t) * 0.5
    #vertices_new[:, 2] += f * vertices[:, 0] ** 2 # parabola
    #vertices_new[:, 2] += 0.3 * torch.sin(vertices[:, 0] * torch.pi + t)
    vertices_new[:, 2] += t * (vertices[:, 1] ** 2 + vertices[:, 1] ** 2) ** (1 / 2) * 0.01
    return vertices_new


def transform_ficus_pot(vertices, t, idxs):
    if t > 8 * torch.pi:
        vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)
    else:
        vertices[idxs, 2] -= (0.005+t) * (vertices[idxs, 0]/10) ** 2
    return vertices


def transform_ship_sinus(vertices, t, idxs=None):
    f = torch.sin(t) * 0.5
    vertices[:, 2] += 0.05 * torch.sin(vertices[:, 0] * torch.pi + f) # sinus
    return vertices


def make_smaller(vertices, t, idxs=None):
    vertices_new = vertices.clone()
    f = torch.sin(t) + 1
    vertices_new = f * vertices_new
    return vertices_new

def do_not_transform(vertices, t):
    return vertices

def transform_fron_file(path):
    mesh_scene = trimesh.load(path, force='mesh')
    #mesh_scene = trimesh.load("/home/s184/gaussian-mesh-splatting/data/horsehead/models/modify.obj", force='mesh')
    #mesh_scene = trimesh.load("/home/s184/gaussian-mesh-splatting/data/horsehead/models/modify_unicorn.obj", force='mesh')
    #mesh_scene = trimesh.load("/home/s184/gaussian-mesh-splatting/data/xinzhou_small/xinzhou-100wfaces-modify-model/xinzhou-100wfaces-120meter-modify.obj", force='mesh')
    #mesh_scene = trimesh.load("/home/s184/gaussian-mesh-splatting/data/xinzhou_small/xinzhou-100wfaces-modify-model/new_modify.obj", force='mesh')
    # vertices = mesh_scene.vertices
    # faces = mesh_scene.faces
    # triangles = torch.tensor(mesh_scene.triangles).float()  # equal vertices[faces]

    vertices = torch.tensor(mesh_scene.vertices)
    # vertices = transform_vertices_function_colmap(
    #     torch.tensor(vertices),
    # )
    faces = mesh_scene.faces
    #triangles = vertices[torch.tensor(mesh_scene.faces).long()].float()
    return vertices, faces

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mesh_path, norm_aug=False):
    file_name = os.path.split(mesh_path)[-1].replace('.obj', '')
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), file_name)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    t = torch.linspace(0, 10 * torch.pi, len(views))

    vertices = gaussians.vertices
    print("raw vertices:", vertices.shape)

    # chose indexes if you want change partly
    idxs = None

    new_vertices, new_faces = transform_fron_file(mesh_path)#transform_hotdog_fly(vertices, t[idx], idxs)
    new_vertices = new_vertices.cuda()
    triangles = new_vertices[torch.tensor(gaussians.faces).long()].float().cuda()
    #triangles = new_vertices[torch.tensor(faces).long()].float().cuda()
    #_xyz = gaussians._xyz
    if norm_aug:
        _xyz = torch.matmul(
                torch.concat((gaussians.alpha, gaussians.soft_normalpha), dim=-1),       #[N,K,3]]
                torch.concat((gaussians.triangles, gaussians.v0.unsqueeze(1)), dim=-2)
            )
    else:
        _xyz = torch.matmul(
            gaussians.alpha,
            gaussians.triangles
        )

    _xyz = _xyz.reshape(
        _xyz.shape[0] * _xyz.shape[1], 3
    )
    gaussians._xyz = _xyz
    gaussians._save_ply(os.path.join(model_path, file_name+"_pc.ply"))
    print(norm_aug)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(idxs, triangles, view, gaussians, pipeline, background, norm_aug=norm_aug)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, mesh_path, norm_aug : bool):
    with torch.no_grad():
        if norm_aug:
            gaussians = GaussianMeshNormAugModel(dataset.sh_degree)
        else:
            gaussians = GaussianMeshModel(dataset.sh_degree)
        #print("eval:", dataset.eval)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'prepare_v0'):
            gaussians.prepare_v0()
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #mesh_scene = trimesh.load(f'{dataset.source_path}/mesh.obj', force='mesh')

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, mesh_path, norm_aug)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, mesh_path, norm_aug)

#def read_


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--mesh_path", default='/data/exp_obj_data/bear/models/mesh_2.obj', type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--norm_aug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)
    #print("eval:", args.eval)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mesh_path, args.norm_aug)