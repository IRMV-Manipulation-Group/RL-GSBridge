import numpy as np
import pymeshlab as pml
import torch

def decimate_mesh(
    verts, faces, target, backend="pymeshlab", remesh=False, optimalplacement=True
):
    # optimalplacement: default is True, but for flat mesh must turn False to prevent spike artifect.

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    if backend == "pyfqmr":
        import pyfqmr

        solver = pyfqmr.Simplify()
        solver.setMesh(verts, faces)
        solver.simplify_mesh(target_count=target, preserve_border=False, verbose=False)
        verts, faces, normals = solver.getMesh()
    else:
        m = pml.Mesh(verts, faces)
        ms = pml.MeshSet()
        ms.add_mesh(m, "mesh")  # will copy!

        # filters
        # ms.meshing_decimation_clustering(threshold=pml.PercentageValue(1))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), optimalplacement=optimalplacement
        )

        if remesh:
            # ms.apply_coord_taubin_smoothing()
            ms.meshing_isotropic_explicit_remeshing(
                iterations=3, targetlen=pml.PercentageValue(1)
            )

        # extract mesh
        m = ms.current_mesh()
        verts = m.vertex_matrix()
        faces = m.face_matrix()

    print(
        f"[INFO] mesh decimation: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces


def clean_mesh(
    verts,
    faces,
    v_pct=1,
    min_f=64,
    min_d=20,
    repair=True,
    remesh=True,
    remesh_size=0.01,
):
    # verts: [N, 3]
    # faces: [N, 3]

    _ori_vert_shape = verts.shape
    _ori_face_shape = faces.shape

    m = pml.Mesh(verts, faces)
    ms = pml.MeshSet()
    ms.add_mesh(m, "mesh")  # will copy!

    # filters
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces

    if v_pct > 0:
        ms.meshing_merge_close_vertices(
            threshold=pml.PercentageValue(v_pct)
        )  # 1/10000 of bounding box diagonal

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_d > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(min_d)
        )

    if min_f > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_f)

    if repair:
        # ms.meshing_remove_t_vertices(method=0, threshold=40, repeat=True)
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    if remesh:
        # ms.apply_coord_taubin_smoothing()
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, targetlen=pml.PureValue(remesh_size)
        )

    # extract mesh
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()

    print(
        f"[INFO] mesh cleaning: {_ori_vert_shape} --> {verts.shape}, {_ori_face_shape} --> {faces.shape}"
    )

    return verts, faces

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-30)#1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det
    # det = (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f)
    # inv_a = (d * f - e**2) / det
    # inv_b = (e * c - b * f) / det
    # inv_c = (e * b - c * d) / det
    # inv_d = (a * f - c**2) / det
    # inv_e = (b * c - e * a) / det
    # inv_f = (a * d - b**2) / det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    #print(power[power > 0])
    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)


# # Gather gaussian parameters
#         close_gaussian_centers = self.points[closest_gaussians_idx]
#         close_gaussian_inv_scaled_rotation = self.get_covariance(
#             return_full_matrix=True, return_sqrt=True, inverse_scales=True
#             )[closest_gaussians_idx]
#         close_gaussian_strengths = self.strengths[closest_gaussians_idx]
        
#         # Compute the density field as a sum of local gaussian opacities
#         shift = (x[:, None] - close_gaussian_centers)
#         warped_shift = close_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
#         neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
#         neighbor_opacities = density_factor * close_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
#         densities = neighbor_opacities.sum(dim=-1)
    
def gaussian_3d_coeff_mat(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 3, 3]
    #warped_shift = covs.transpose(-1, -2) @ xyzs[..., None]
    #warped_shift = covs @ xyzs[..., None]
    #print('warped_shift', (warped_shift[..., 0] * warped_shift[..., 0]).shape)
    #neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
    #print('single:', xyzs[10, None] @ covs[10] @ xyzs[10, :, None])
    #print('all:', (xyzs[:, None] @ covs @ xyzs[..., None])[10])
    #neighbor_opacities = (xyzs[:, None] @ covs @ xyzs[..., None]).clamp(min=0., max=1e8)
    neighbor_opacities = (xyzs[:, None] @ covs @ xyzs[..., None])#.clamp(min=0., max=1e8)
    neighbor_opacities[neighbor_opacities < 0] = 1e10
    
    return torch.exp(-1. / 2 * neighbor_opacities.squeeze())
    
