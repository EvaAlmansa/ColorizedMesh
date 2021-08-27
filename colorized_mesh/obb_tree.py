import vtk


class OBBTree:

    def __init__(self, mesh):
        self.obb_tree = vtk.vtkOBBTree()
        self.obb_tree.SetDataSet(mesh)
        self.obb_tree.BuildLocator()

    def intersection_with_line(self, p_source, p_target):
        intersect_points = vtk.vtkPoints()
        self.obb_tree.IntersectWithLine(p_source, p_target, intersect_points, None)
        return intersect_points

    def intersection_with_triangles(self, p_source, p_target):
        
        points = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()

        # Perform intersection test
        code = self.obb_tree.IntersectWithLine(p_source, p_target, points, cell_ids)
        
        point_data = points.GetData()
        no_points = point_data.GetNumberOfTuples()
        no_ids = cell_ids.GetNumberOfIds()
        
        points_inter = []
        cellids_inter = []
        for idx in range(no_points):
            points_inter.append(point_data.GetTuple3(idx))
            cellids_inter.append(cell_ids.GetId(idx))
        
        return points_inter, cellids_inter