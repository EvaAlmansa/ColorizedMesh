import numpy as np
import sys
from vedo import *
import vtk
import pyvista as pv

from utils.save_results import save_results
from mesh.manager import *
from colorized_mesh.obb_tree import OBBTree

    
class ColorizedMesh:

    GREY = (192,192,192)
    YELLOW = (255,250,67)
    RED = (255,0,0)

    def __init__(
        self, args
    ):
        self.name = args.mesh_name[:-4]
        self.result_path = args.result_path 

        self.mesh_name = '%s' % self.name
        self.mesh_path = args.mesh_path + '%s.obj' % self.mesh_name
        self.mesh_vtk = args.vtk_path + '%s.vtk' % self.name
        
        self.furniture_path = None
        self.furniture_vtk = None
        if args.furniture_path:
            self.furniture_path = args.furniture_path + '%s.obj' % self.name
            self.furniture_vtk = args.vtk_path + '%s_f.vtk' % self.name

        self.visible_vertices = []
        self.invisible_vertices = []
        self.occlused_vertices = []
        self.n_vertices = 0

        self.camera = self._origin_position()
        self.tree_m = None
        self.tree_f = None

        self.remesh = False
        if args.nsubd > 0:
            self.remesh = True
            self.nsub = args.nsubd
            remesh = self._subdivide_mesh(nsub=self.nsub)
            self._save_mesh(args, remesh)

    def _distance_points(self, vertex, intersect_points):
        camera = self.camera
        versor = (vertex-camera) / np.linalg.norm(vertex-camera)

        vertex_inter_t = []
        for j in range(intersect_points.GetNumberOfPoints()):
            inter = [0,0,0]
            intersect_points.GetPoint(j, inter)

            # Parametric coordinate of inter WRT the camera-vertex line
            t = (inter[0]-camera[0]) / versor[0] / np.linalg.norm(vertex-camera)

            vertex_inter_t.append(t)
        return vertex_inter_t

    def _initialization_occluded_point_map(self):
        '''
            Load an obj file and save a vtk file for the bounding mesh and the furniture mesh.
            Return a boundary mesh and a furniture mesh, both are vtk mesh.

            Note: if there is not a furniture mesh as input, it will not save neither return this mesh.
        '''
        mesh = load_mesh(self.mesh_path)
        write_vtk_file(mesh, self.mesh_vtk)
        self.tree_m = OBBTree(mesh)

        furniture = None
        if self.furniture_path:
            furniture = load_mesh(self.furniture_path)
            write_vtk_file(furniture, self.furniture_vtk)
            self.tree_f = OBBTree(furniture)
        return mesh, furniture

    def _keep_points(self, vertex, vertex_inter_t):
        # So we keep points whose line intersect the mesh only at the end of the vertex-camera segment
        if vertex_inter_t:
            tt = min(vertex_inter_t)
            if tt > 0.999 and tt < 1.001:
                visible = True
                if self.tree_f:
                    intersect_points, cell_ids = self.tree_f.intersection_with_triangles(self.camera, vertex)
                    if len(cell_ids) > 0:
                        self.occlused_vertices.append(vertex)
                        visible = False
                if visible:
                        self.visible_vertices.append(vertex)
            else:
                self.invisible_vertices.append(vertex)

    def _origin_position(self):
        'Get Camera position/Origin'
        return np.array([0, 0, 0])

    def _save_mesh(self, args, remesh, plot=False):

        self.mesh_path = args.remesh_path + self.mesh_name + '.obj'
        self.mesh_vtk = args.remesh_vtk_path + self.mesh_name + '.vtk'

        pv.save_meshio(self.mesh_path, remesh)
        # remesh.save(remesh_path)

        if plot:
            remesh = pv.read(self.mesh_path)
            remesh.plot()

    def _subdivide_mesh(self, nsub=1, subfilter='linear', plot=False):
        '''
            Increase the number of triangles in a single, connected triangular mesh.

            Uses one of the following vtk subdivision filters to subdivide a mesh:
            - Linear subdivision results in the fastest mesh subdivision, but it does not smooth mesh edges, 
            but rather splits each triangle into 4 smaller triangles.
            - Butterfly and loop subdivision perform smoothing when dividing, and may introduce artifacts 
            into the mesh when dividing.

            Subdivision filter appears to fail for multiple part meshes. Should be one single mesh.
        '''
        print('Subdividing mesh...')
        mesh = load_mesh(self.mesh_path)
        mesh = pv.PolyData(mesh)

        remesh = mesh.subdivide(nsub=nsub, subfilter=subfilter)

        if plot:
            plotter_sub(self.camera, remesh, nsub, subfilter)
        print('... Subdividing mesh done')
        
        return remesh
        

    def occluded_point_map(self, plot_rays_casting=True):

        # Initialization
        mesh, furniture = self._initialization_occluded_point_map()

        vertices = mesh.GetPoints()
        self.n_vertices = vertices.GetNumberOfPoints()

        print('Coloring mesh...')

        # Vertices are repeated due to triangle-fan representation
        print('N vertices: ', self.n_vertices)
        undup_vertices = set()
        for i in range(self.n_vertices):
            vertex = [0,0,0]
            vertices.GetPoint(i, vertex)

            # Convert the vertex list into a tuple
            # This is because lists are not hashable
            vertex_tuple = (*vertex,)
            if not vertex_tuple in undup_vertices:
                undup_vertices.add( vertex_tuple )
                
                intersect_points = self.tree_m.intersection_with_line(self.camera, vertex)
                vertex_inter_t = self._distance_points(np.array(vertex), intersect_points)

                self._keep_points(vertex, vertex_inter_t)
                
        print('Visible vertices = ', len(self.visible_vertices))
        print('Invisible vertices = ', len(self.invisible_vertices))
        print('Occlused vertices = ', len(self.occlused_vertices))

        print('... Coloring mesh done')

    def plotter_point_map(self, plot_rays_casting=False):
        
        visibles = Points(self.visible_vertices, r=8, c=self.GREY)
        invisibles = Points(self.invisible_vertices, r=8, c=self.RED)
        occluded = Points(self.occlused_vertices, r=8).color(self.YELLOW)
        
        mesh = Mesh(self.mesh_vtk, c=self.GREY).wireframe()
        furniture = Mesh(self.furniture_vtk, c=self.GREY)

        camera = Point(self.camera, r=12, c='magenta')
        
        if plot_rays_casting:
            rays_casting = [Line(camera, vertex, c='magenta') for vertex in self.visible_vertices]
            show(camera, mesh, furniture, visibles, rays_casting, invisibles, occluded)
        else:
            show(camera, mesh, furniture, visibles, invisibles, occluded)
            show(camera, mesh, visibles, invisibles, occluded)
            show(mesh, visibles, invisibles, occluded)

            mesh = Mesh(self.mesh_vtk, c=self.GREY).wireframe()
            show(camera, mesh)

            mesh = Mesh(self.mesh_vtk)
            colors = (self.GREY, self.YELLOW, self.RED)
            scalar = [0]*len(self.visible_vertices) + [1]*len(self.occlused_vertices) + [2]*len(self.invisible_vertices)
            vertices = self.visible_vertices + self.occlused_vertices + self.invisible_vertices
            points = Points(vertices, r=1).cmap(colors, scalar)
            #mesh.interpolateDataFrom(points, N=2).cmap(colors).addScalarBar()
            mesh.interpolateDataFrom(points, N=2).cmap(colors)
            show(mesh, points)
            show(mesh).screenshot(self.result_path + self.name + '.jpg')

            mesh = Mesh(self.mesh_vtk, c=self.GREY).wireframe()
            colors = (self.GREY, self.RED)
            scalar = [0]*len(self.visible_vertices) + [2]*len(self.invisible_vertices)
            vertices = self.visible_vertices + self.invisible_vertices
            points = Points(vertices, r=1).cmap(colors, scalar)
            mesh.interpolateDataFrom(points, N=2).cmap(colors)
            show(mesh)

    def save_number_points(self):
        save_results(
            self.name, self.n_vertices, len(self.visible_vertices), 
            len(self.invisible_vertices), len(self.occlused_vertices), self.result_path
        )

    def write_occluded_point_map(self, wireframe=False):

        mesh = Mesh(self.mesh_vtk)
        colors = (self.GREY, self.YELLOW, self.RED)
        scalar = [0]*len(self.visible_vertices) + [1]*len(self.occlused_vertices) + [2]*len(self.invisible_vertices)
        vertices = self.visible_vertices + self.occlused_vertices + self.invisible_vertices
        points = Points(vertices, r=1).cmap(colors, scalar)
        mesh.interpolateDataFrom(points, N=2).cmap(colors)
        mesh.write(self.result_path + self.name + '_color.ply')

        if wireframe:
            mesh = Mesh(self.mesh_vtk, c=self.GREY).wireframe()
            colors = (self.GREY, self.RED)
            scalar = [0]*len(self.visible_vertices) + [2]*len(self.invisible_vertices)
            vertices = self.visible_vertices + self.invisible_vertices
            points = Points(vertices, r=1).cmap(colors, scalar)
            mesh.interpolateDataFrom(points, N=2).cmap(colors)
            mesh.write(self.result_path + self.name + '_color2.ply')
    
    # ---------------------------------------------------------------
    # Doesn't work, I haven't been able to fix the suggestion by Enrico
    #
    # def occluded_point_map_v2(self, camera, vertices):
    #     eps = np.finfo(float).eps
    #     ray_origin = camera

    #     for j in range(vertices.GetNumberOfPoints()):
    #         ray_extremity = [0,0,0]
    #         vertices.GetPoint(j, ray_extremity)
    #         ray_direction = (ray_extremity-camera) / np.linalg.norm(ray_extremity-camera)
    #         #ray_direction = np.linalg.norm(ray_extremity-camera)
    #         ray_extremity = ray_extremity - eps * ray_direction; # shift extremity towards origin to avoid fake self-intersections 

    #         is_visible_when_empty = self.tree_m.IntersectWithLine(ray_origin, ray_extremity, None, None)
    #         is_visible_when_not_cluttered = is_visible_when_empty and self.tree_f.IntersectWithLine(ray_origin, ray_extremity, None, None)
    #         if is_visible_when_not_cluttered:
    #             self.visible_vertices.append(ray_extremity)
    #         # elif is_visible_when_empty:
    #         #     self.occlused_vertices.append(ray_extremity)
    #         else:
    #             self.invisible_vertices.append(ray_extremity)

    #     print('Visible vertices = ', len(self.visible_vertices))
    #     print('Invisible vertices = ', len(self.invisible_vertices))
    #     print('Occlused vertices = ', len(self.occlused_vertices))

    #     self.plotting_point_map(camera, rays=True)

# ---------------------- End colorized mesh

def examples_vedo(cylinder = False):
    
    if cylinder:

        cyl = Cylinder() # vtkActor

        cyl.alpha(0.5).pos(3,3,3).orientation([2,1,1])

        p1, p2 = (0,0,0), (4,4,5)

        ipts_coords = cyl.intersectWithLine(p1, p2)
        print('hit coords are', ipts_coords)

        pts = Points(ipts_coords, r=10).color("yellow")
        # print(pts.polydata())  # is the vtkPolyData object

        origin = Point()
        ln = Line(p1,p2)

        show(origin, cyl, ln, pts, axes=True)

        def fibonacci_sphere(n):
            s = np.linspace(0, n, num=n, endpoint=False)
            theta = s * 2.399963229728653
            y = 1 - s * (2/(n-1))
            r = np.sqrt(1 - y * y)
            x = np.cos(theta) * r
            z = np.sin(theta) * r
            return [x,y,z]

        Points(fibonacci_sphere(1000)).show(axes=1)

def testing(mesh):
    '''
        Triangle mesh contains vertices and triangles represented by the indices to the vertices. 
    '''
    print('Vertices:', len(np.asarray(mesh.vertices)))
    print(np.asarray(mesh.vertices))
    print('Triangles:', len(np.asarray(mesh.triangles)))
    print(np.asarray(mesh.triangles))
    print('Center: ', mesh.get_center())

    
    # import copy
    # mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
    # mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
    # print(f'Center of mesh: {mesh.get_center()}')
    # print(f'Center of mesh tx: {mesh_tx.get_center()}')
    # print(f'Center of mesh ty: {mesh_ty.get_center()}')
    # o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])
    
    # b = Mesh(np.asarray(mesh.vertices))
    # b.computeNormals().clean().lw(0.1)

    # pids = b.boundaries(returnPointIds=True)
    # bpts = b.points()[pids]

    # pts = Points(bpts, r=10, c='red')

    # labels = b.labels('id', scale=10).c('dg')

    # show(b, pts, labels, __doc__, zoom=2).close()
