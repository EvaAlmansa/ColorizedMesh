import configparser
import os

from .argument_parser import argument_parser


def get_root_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def get_complete_dir(args, config, arg_name):
    if args.complete_data_path:
        return ''
    return get_root_dir() + config.get(arg_name)


def _cuda_arguments(args, config):
    args.cuda_devices = config.get('cuda_devices', fallback='0,1')


def _general_settings_arguments(args, config):
    args.complete_data_path = config.getboolean('complete_data_path', fallback=True)
    args.result_path = get_complete_dir(args, config, 'result_path')


def _initialization_arguments(args, config):
    args.img_path = get_complete_dir(args, config, 'img_path')
    args.mesh_path = get_complete_dir(args, config, 'mesh_path')
    args.furniture_path = get_complete_dir(args, config, 'furniture_path')
    args.vtk_path = get_complete_dir(args, config, 'vtk_path')
    args.mesh_name = config.get('mesh_name', fallback=None)


def _subdivide_arguments(args, config):
    args.nsubd = config.getint('nsubd', fallback=0)
    args.remesh_path = get_complete_dir(args, config, 'remesh_path')
    args.remesh_vtk_path = get_complete_dir(args, config, 'remesh_vtk_path')
    

def _dataset_arguments(args, config):
    args.v_pad_max = config.get('v_pad_max', fallback=None)
    args.f_pad_max = config.get('f_pad_max', fallback=None) 
    args.return_name = config.getboolean('return_name', fallback=False) 
    args.return_valid = config.getboolean('return_valid', fallback=False) 


def _argument_loader(args, config):
    _cuda_arguments(args, config)
    _general_settings_arguments(args, config)
    _initialization_arguments(args, config)
    _subdivide_arguments(args, config)
    _dataset_arguments(args, config)
    

def argument_loader(print_args=False):

    args = argument_parser()

    config = configparser.ConfigParser()
    if len(config.read(args.config_file)) == 0:
        raise (RuntimeError("Can't open configuration file: " + args.config_file)) 

    _argument_loader(args, config[args.config])

    if print_args:
        print('args:')
        for key, val in vars(args).items():
            print('    {:16} {}'.format(key, val))
    
    return args


def print_configuration (
        args, 
        print_args=['cuda_devices', 'img_path', 'mesh_path', 'furniture_path'], 
        print_section=False
    ):
    
    print('args:')
    for key in print_args:
        print('    {:16} {}'.format(key, vars(args)[key]))

    if print_section:
        print('Sections in configuration file:')
        print(config.sections())
    
    