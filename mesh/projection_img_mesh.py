import cv2
from itertools import islice
import numpy as np
import os
from PIL import Image, ImageDraw
from pytorch3d.structures import Meshes
import torch

from misc.point2mesh import *


def draw_projected_points(xyz, x_img, name, result_path, paint_img = True):
    
    Vt = torch.FloatTensor(xyz)
    uvd_verts = xyz2uvd(Vt)
    
    # u,v are in norm image coords (-1,-1, top left)
    u, v, d = torch.unbind(uvd_verts.squeeze(0), dim=1) 
          
    # to pixels
    u = (u+1.0) * (x_img.shape[2]/2)
    v = (v+1.0) * (x_img.shape[1]/2)

    u = np.array(u, dtype=np.uint16)
    v = np.array(v, dtype=np.uint16)

    uv = np.stack((u, v), axis=-1)
       
    if paint_img:
        dump = (x_img.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
        dump = np.array(dump)
        dump = Image.fromarray(dump).save('temp.jpg')
        dump = cv2.imread('temp.jpg')
        os.remove('temp.jpg')
    else:
        dump=np.zeros((x_img.shape[1],x_img.shape[2],3),np.uint8)

    for p in uv:
        dump = cv2.circle(dump, (p[0],p[1]), radius=1, color=(255, 255, 255), thickness=-1)

    cv2.imwrite(result_path + name + '_projected.jpg', dump)
    cv2.imshow(name, dump)
    cv2.waitKey(5000)


def draw_mesh(vertices, triangles, x_img, name, img_path, result_path, num_samples=10000, not_draw_edges=False):
    
    mesh = Meshes(verts=vertices.unsqueeze(0), faces=triangles.unsqueeze(0))

    #points_gt = sample_points_from_meshes(mesh, num_samples=5000)  
    e_points = sample_points_from_edges(mesh, th = 0.0, num_samples=num_samples, no_sampling=not_draw_edges)              
    draw_projected_points(e_points, x_img, name, result_path, paint_img = True)


def get_sharpen_edges(meshes, th = 0.5, num_samples = 5000):
    """
    Args:
        meshes: Meshes object with a batch of meshes.

    Returns:
        list of edges which sharpeness is above th
        Returns [] if meshes contains no meshes or all empty meshes.
    """
    N = len(meshes) ####default: 1

    edges = []
    edge_points = torch.zeros((N, num_samples, 3), device=meshes.device)

    if meshes.isempty():
        return edges, edge_points
        
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 2)
    verts_packed_to_mesh_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    
    face_to_edge = meshes.faces_packed_to_edges_packed()  # (sum(F_n), 3)
    E = edges_packed.shape[0]  # sum(E_n)
    F = faces_packed.shape[0]  # sum(F_n)
        
    with torch.no_grad():
        edge_idx = face_to_edge.reshape(F * 3)  # (3 * F,) indexes into edges
        vert_idx = (
            faces_packed.view(1, F, 3).expand(3, F, 3).transpose(0, 1).reshape(3 * F, 3)
        )
        edge_idx, edge_sort_idx = edge_idx.sort()
        vert_idx = vert_idx[edge_sort_idx]
               
        edge_num = edge_idx.bincount(minlength=E)
        # Create pairs of vertices associated to e. We generate a list of lists:
        # each list has the indices of the vertices which are opposite to one edge.
        # The length of the list for each edge will vary.
        vert_edge_pair_idx = split_list(
            list(range(edge_idx.shape[0])), edge_num.tolist()
        )
        # For each list find all combinations of pairs in the list. This represents
        # all pairs of vertices which are opposite to the same edge.
        vert_edge_pair_idx = [
            [e[i], e[j]]
            for e in vert_edge_pair_idx
            for i in range(len(e) - 1)
            for j in range(1, len(e))
            if i != j
        ]
        vert_edge_pair_idx = torch.tensor(
            vert_edge_pair_idx, device=meshes.device, dtype=torch.int64
        )

    v0_idx = edges_packed[edge_idx, 0]
    v0 = verts_packed[v0_idx]
    v1_idx = edges_packed[edge_idx, 1]
    v1 = verts_packed[v1_idx]

    # two of the following cross products are zeros as they are cross product
    # with either (v1-v0)x(v1-v0) or (v1-v0)x(v0-v0)
    n_temp0 = (v1 - v0).cross(verts_packed[vert_idx[:, 0]] - v0, dim=1)
    n_temp1 = (v1 - v0).cross(verts_packed[vert_idx[:, 1]] - v0, dim=1)
    n_temp2 = (v1 - v0).cross(verts_packed[vert_idx[:, 2]] - v0, dim=1)
    n = n_temp0 + n_temp1 + n_temp2

    n0 = n[vert_edge_pair_idx[:, 0]]
    n1 = -n[vert_edge_pair_idx[:, 1]]
    
    e_sharp = 1 - torch.cosine_similarity(n0, n1, dim=1)
    # print('cosine',torch.cosine_similarity(n0, n1, dim=1), v0_idx, v1_idx)
            
    for i in range(len(e_sharp)):
        if(e_sharp[i] > th):
            edges.append(edges_packed[i])
    
    NE = len(edges)
    TS = num_samples  
    
    edge_vertices = []
    
    if(NE>0):
        ES = TS // NE
        ER = TS % NE

        t_points = []
        for i in range(NE):                                    
            v0_idx = edges[i][0]
            v0 = verts_packed[v0_idx]
            v1_idx = edges[i][1]
            v1 = verts_packed[v1_idx]

            edge_vertices.append(v0)
            edge_vertices.append(v1)

            e_points = interpolate_points(v0, v1, ES)
            t_points.append(e_points)
            # print('inter points per edge', e_points.shape)
        
        # add points from edge 0 to pad tensor 
        if(ER>0):
            v0_idx = edges[0][0]
            v0 = verts_packed[v0_idx]

            v1_idx = edges[0][1]
            v1 = verts_packed[v1_idx]
            
            e_points = interpolate_points(v0, v1, ER)
            t_points.append(e_points)

        edge_points = torch.cat(t_points, 1)
        edge_points = edge_points.to(meshes.device)

        # output edge vertices
        edge_vertices = torch.stack(edge_vertices, dim = 0)
        edge_vertices = edge_vertices.unsqueeze(0)
        edge_vertices = edge_vertices.to(meshes.device)
    else:
        print('WARNING: no edges found for th:',th)
        edge_vertices = torch.zeros((N, len(e_sharp), 3), device=meshes.device)
        
    return edge_vertices, edge_points


def interpolate_points(v0, v1, steps):
    e_points = []
            
    e_points_x = torch.linspace(v0[0], v1[0], steps=steps)
    e_points.append(e_points_x)
    e_points_y = torch.linspace(v0[1], v1[1], steps=steps)
    e_points.append(e_points_y)
    e_points_z = torch.linspace(v0[2], v1[2], steps=steps)
    e_points.append(e_points_z)

    e_points = torch.stack(e_points, dim = 1)
    e_points = e_points.unsqueeze(0)

    return e_points


def split_list(input, length_to_split):
    inputt = iter(input)
    return [list(islice(inputt, elem)) for elem in length_to_split]


def sample_points_from_edges(meshes, num_samples: int = 5000, th=0.5, no_sampling = False):      
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()

    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")
     
    num_meshes = len(meshes)
    ##num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Intialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    edge_samples = []
    for i in range(num_meshes):
        edge_vertices, sharp_points = get_sharpen_edges(meshes.__getitem__(i), th=th, num_samples = num_samples)
        
        if no_sampling:
            edge_samples.append(edge_vertices)
        else:
            edge_samples.append(sharp_points)
    
    if len(edge_samples)>0:
        samples = torch.cat(edge_samples, 0)
                   
    return samples
    



