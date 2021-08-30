import argparse


def _cuda_parser(parser):
    parser.add_argument(
        '--cuda_devices', 
        help='Configuration of os.environ[CUDA_VISIBLE_DEVICES]. (Default: 0,1)'
    )


def _general_settings_parser(parser):
    conf_modes = ['DEFAULT']

    parser.add_argument(
        '--config_file', 
        required = True,
        help = 'Configuration file.'
    )
    parser.add_argument(
        '--config', 
        default='DEFAULT',
        help='Configuration mode which is one option in configuration file: ' + ' | '.join(conf_modes) + '. (Default: DEFAULT)'
    )
    parser.add_argument(
        '--complete_data_path', 
        help='If the dataset is in the same path as the project this flag should be False'
        ' otherwise should be a True and complete all the paths. (Default: True)'
    )
    parser.add_argument(
        '--result_path'
    )


def _initialization_parser(parser):
    parser.add_argument(
        '--img_path', 
    )
    parser.add_argument(
        '--mesh_path', 
    )
    parser.add_argument(
        '--furniture_path', 
    )
    parser.add_argument(
        '--vtk_path'
    )
    parser.add_argument(
        '--mesh_name', 
        help ='Pick a mesh name to visualize (OBJ file). None for visualize all data. (Default: None)'
    )


def _subdivide_parser(parser):
    parser.add_argument(
        '--nsubd', 
        type=int,
        help='Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting'  
        ' triangles is nface*4**nsub where nface is the current number of faces. (Default: 0)'
    )
    parser.add_argument(
        '--remesh_path', 
    )
    parser.add_argument(
        '--remesh_vtk_path', 
    )
    

def argument_parser():
    parser = argparse.ArgumentParser()
    
    _cuda_parser(parser)
    _general_settings_parser(parser)
    _initialization_parser(parser)
    _subdivide_parser(parser)

    return parser.parse_args()