import matplotlib.pyplot as plt
import open3d as o3d
import os
import pyvista as pv
from tqdm import tqdm
from vedo import *
import vtk

from misc import tools
from mesh.projection_img_mesh import draw_mesh


def create_o3d_mesh(vertices, triangles):
    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles)
    )


def check_properties(name, mesh):
    '''
        The code below tests a number of triangle meshes against those properties and visualizes the results. 
        Non-manifold edges are shown in red, boundary edges in green, non-manifold vertices are visualized 
        as green points, and self-intersecting triangles are shown in pink.
    '''
    print("Try to render a mesh with normals (exist: " +
      str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      str(mesh.has_vertex_colors()) + ")")
    o3d.visualization.draw_geometries([mesh])
    print("A mesh with no normals and no colors does not look good.")
    
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 1)))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)
    print("Try to render a mesh with normals (exist: " +
      str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      str(mesh.has_vertex_colors()) + ")")


def check_mesh_properties(dataset):
    
    for x_c, vertices, triangles, f_name, img_path, is_val in tqdm(dataset): 
        mesh = create_o3d_mesh(vertices, triangles)
        check_properties(f_name, mesh)


def mesh_empty(mesh, mesh_path):
        '''
            If there is an empty list of vertices the mesh is empty (the program ends),
            otherwise the mesh is created (return False).
        '''
        if not mesh.GetPoints():
            print('Mesh file not found: ', mesh_path)
            sys.exit()
        return False

def load_polydata(file_name):
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

    # todo better generic load
    if file_extension == "vtk":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "vtp":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "fib":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "ply":
        reader = vtk.vtkPLYReader()
    elif file_extension == "stl":
        reader = vtk.vtkSTLReader()
    elif file_extension == "xml":
        reader = vtk.vtkXMLPolyDataReader()
    elif file_extension == "obj":
        reader = vtk.vtkOBJReader()
        #try:  # try to read as a normal obj
        #    reader = vtk.vtkOBJReader()
        #except:  # than try load a MNI obj format
        #    reader = vtk.vtkMNIObjectReader()
    else:
        raise "polydata " + file_extension + " is not suported"

    reader.SetFileName(file_name)
    reader.Update()
    print ('Mesh', file_extension, 'loaded:', file_name)
    return reader.GetOutput()


def load_mesh(mesh_path):
    mesh = load_polydata(mesh_path)
    mesh_empty(mesh, mesh_path)
    return mesh


def plot_img(img, index, name):
    plt.figure(index)
    plt.title(name)
    plt.imshow(img)  
    plt.show()  


def plot_mesh(vertices, triangles):
    mesh3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles)
    )

    mesh3d.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh3d])  


def plot_subdivisions(mesh, a=1, b=3):

    display_args = dict(show_edges=True, color=True)
    p = pv.Plotter(shape=(3,3))

    for i in range(3):
        p.subplot(i,0)
        p.add_mesh(mesh, **display_args)
        p.add_text("Original Mesh")

    def row_plot(row, subfilter):
        subs = [a, b]
        for i in range(2):
            p.subplot(row, i+1)
            p.add_mesh(mesh.subdivide(subs[i], subfilter=subfilter), **display_args)
            p.add_text(f"{subfilter} subdivision of {subs[i]}")
    row_plot(0, "linear")
    row_plot(1, "butterfly")
    row_plot(2, "loop")

    p.link_views()
    p.view_isometric()
    return p


def plotter_sub(camera, mesh, nsub, subfilter):
    display_args = dict(show_edges=True, color=True)
    p = pv.Plotter()
    p.add_mesh(mesh, **display_args)
    #p.add_mesh(camera, color='magenta')
    p.add_text(f"{subfilter} subdivision of {nsub}")
    p.show()


def projection_mesh_to_image(dataset, result_path, not_draw_edges=False, render_mesh=False):

    i = 0
    for x_c, vertices, triangles, f_name, img_path, is_val in tqdm(dataset): 

        f_name = os.path.split(f_name)[-1].split('.')[0]
                           
        x_img_c = tools.x2image(x_c)                                                                               

        draw_mesh(vertices, triangles, x_c, f_name, img_path, result_path, not_draw_edges=not_draw_edges)                                                 
                            
        if render_mesh:                    
            plot_mesh(vertices, triangles)
            plot_img(x_img_c, i, f_name)
            i = i+1


def visualizer_mesh_furniture(args, mesh_names, furniture_ext='obj'):
    
    mesh_path = args.mesh_path
    furniture_path = args.furniture_path
    vtk_path = args.vtk_path
    
    for mesh_name in mesh_names:

        furniture_name = mesh_name[:-3] + furniture_ext

        mesh_vtk = vtk_path + '%svtk' % mesh_name[:-3]
        furniture_vtk = vtk_path + '%s_f.vtk' % mesh_name[:-4]

        mesh = load_polydata(mesh_path + mesh_name)
        write_vtk_file(mesh, mesh_vtk)

        furniture = load_polydata(furniture_path + furniture_name)
        write_vtk_file(furniture, furniture_vtk)
        
        mesh = Mesh(mesh_vtk, c='grey').wireframe()
        furniture = Mesh(furniture_vtk, c='grey')
        
        show(mesh_name, mesh, furniture)


def write_vtk_file(mesh, mesh_vtk):
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(mesh_vtk)
    writer.Write()