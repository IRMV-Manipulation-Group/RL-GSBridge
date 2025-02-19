import pymeshfix as mf

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import aspose.threed as a3d
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='')
args = parser.parse_args()
folder_path = "/data/exp_obj_data/"+ args.name +"/models"
os.makedirs(folder_path, exist_ok=True)
scene = a3d.Scene.from_file("/data/exp_obj_data/"+ args.name +"/scene_dense_mesh_refine.ply")
scene.save(folder_path + "/mesh.obj")
