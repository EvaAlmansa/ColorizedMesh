from colorized_mesh.colorized_mesh import ColorizedMesh
from dataloader.mesh_dataset import MeshDataset
from mesh.manager import check_mesh_properties, projection_mesh_to_image, visualizer_mesh_furniture
from settings.arguments import argument_loader


if __name__ == '__main__':
    
    args = argument_loader(print_args=True)

    # These functions obtain a dataset and check the properties of all meshes
    dataset = MeshDataset(args)
    check_mesh_properties(dataset)

    # These functions are only for a selected mesh
    visualizer_mesh_furniture(args, dataset.mesh_fnames)
    projection_mesh_to_image(dataset, args.result_path, not_draw_edges=False, render_mesh=False)

    colorized_map = ColorizedMesh(args)
    colorized_map.occluded_point_map()
    colorized_map.plotter_point_map()
    colorized_map.save_number_points()
    colorized_map.write_occluded_point_map()
    
    

    
                

