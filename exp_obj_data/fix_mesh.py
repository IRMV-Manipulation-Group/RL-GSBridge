import pymeshfix as mf

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
import aspose.threed as a3d
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='')
args = parser.parse_args()

bunny = pv.read("/data/exp_obj_data/"+ args.name +"/scene_dense_mesh_refine.ply")

# Define a camera position that shows the holes in the mesh

# Show mesh
#bunny.plot()
print(bunny)
meshfix = mf.MeshFix(bunny)
holes = meshfix.extract_holes()
#print(holes)
p = pv.Plotter()
p.add_mesh(bunny, color=True)
p.add_mesh(holes, color="r", line_width=8)
p.enable_eye_dome_lighting()  # helps depth perception
#p.show()

meshfix.repair(verbose=True)
#meshfix.mesh.plot()

meshfix.mesh.save("/data/exp_obj_data/"+ args.name +"/mesh.ply")

folder_path = "/data/exp_obj_data/"+ args.name +"/models"
os.makedirs(folder_path, exist_ok=True)
scene = a3d.Scene.from_file("/data/exp_obj_data/"+ args.name +"/mesh.ply")
scene.save(folder_path + "/mesh.obj")
