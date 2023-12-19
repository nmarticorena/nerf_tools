import torch
from typing import Tuple, List
from tqdm import tqdm
from skimage import measure
import trimesh
import time
from torch.autograd import grad
import numpy as np

class min_sdf():
    def __init__(self, pcd_points) -> None:
        if not isinstance(pcd_points, np.ndarray):
            pcd_points = np.array(pcd_points.points)
        self.point_cloud = torch.from_numpy(pcd_points)#
        self.point_cloud.requires_grad_(True)
        self.pcd = self.point_cloud.to('cuda', dtype=torch.float32)

    def sdf_forward(self, points_robot):
        ti = time.time()
        diff = torch.cdist(points_robot, self.pcd)
        min_dist, _ = torch.min(diff, axis=1)
        tf = time.time()
        dt = (tf-ti) * 1000
        print(f"Time to compute distance {dt} ms")
        return min_dist
       
    def gradient(self, inputs, outputs):
        d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
        points_grad = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return points_grad

def marching_cubes(grid:torch.Tensor, model, scale, dim, aabb, levels = [0.0]) -> List[trimesh.Trimesh]:
    '''
    Perform marching cubes on the sdf to get a level set mesh
    '''
    # final SDF
    sdf = torch.zeros(grid.shape[0]).cuda()
    
    # batch size
            


    # Get SDF values
    grid = grid.cuda()
    
    # separate the grid for each run:
    batch_size = 5000
    
    steps = range(0,grid.shape[0],batch_size)
    
    for i in tqdm(steps):
        sdf[i:i+batch_size] = model.sdf_forward(grid[i:i+batch_size]).detach().squeeze()
    
    sdf = sdf.view(dim,dim,dim).detach().cpu().numpy()
    # sdf = sdf.detach().cpu().numpy().reshape((dim,dim,dim), order= 'F')

    
    print(f"The grid haves the values : {sdf.min().item():2f} : {sdf.max().item():2f}")
    meshes = []

    for level in levels:

        vertices, faces, vertex_normals, _ = measure.marching_cubes(
        sdf, level=level,spacing=scale
        )
        vertices = vertices + aabb[0] 
        # dim = sdf.shape[0]
        # vertices = vertices / (dim - 1)
        mesh = trimesh.Trimesh(vertices=vertices,
                            vertex_normals=vertex_normals,
                            faces=faces, process= False)

        meshes.append(mesh)

    # del(grid)
    return meshes

def grid_from_aabb(aabb, dim) -> Tuple[torch.Tensor, torch.Tensor]:
    x_range = (aabb[0,0], aabb[1,0])
    y_range = (aabb[0,1], aabb[1,1])
    z_range = (aabb[0,2], aabb[1,2])
    grid, scale = generate_3d_grid(x_range, y_range, z_range, dim)
    return grid, scale
    

def generate_3d_grid(x_range, y_range, z_range, dim)-> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Generate a 3D grid of size dim x dim x dim
    '''
    x = torch.linspace(x_range[0], x_range[1], dim)
    y = torch.linspace(y_range[0], y_range[1], dim)
    z = torch.linspace(z_range[0], z_range[1], dim)
    x, y, z = torch.meshgrid(x, y, z)
    grid = torch.stack([x, y, z], dim=3)
    grid = grid.view(-1, 3)
    
    
    scale = torch.tensor([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]) / (dim - 1)
    return grid, scale