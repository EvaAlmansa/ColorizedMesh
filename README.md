
<img src='embryo.html' align='center' width=325>
<br><br><br>

# Table of Contents
1. [Quick start](#quick-start)
1. [Source Code Structure](#source-code-structure)
1. [Installation](#installation)

# Quick start 

Run next command line:
```bash
$ make quick-start
```

# Source Code Structure

```
├── colorized_mesh
|   ├── colorized_mesh.py
|   ├── obb_tree.py
├── dataloader
|   ├── mesh_dataset.py
├── dataset
|   ├── [...]
|   ├── README.md
├── mesh
│   ├── manager.py
│   ├── projection_img_mesh.py
├── misc
│   ├── panostretch.py
│   ├── point2mesh.py
|   ├── tools.py
├── result
|   ├── README.md
├── settings
│   ├── argument_parser.py
│   ├── arguments.py
|   ├── config_files
|       ├── local_config.ini
├── utils
│   ├── save_results.py
├── .gitignore
├── main.py
├── Makefile
├── README.md
└── requirements.txt
```

# Installation

1. [Virtual Environment (Optional)](#virtual-environment-optional) 
1. [Requirements](#requirements)


## Virtual Environment (Optional) 

> Install pip3 if it is already not installed on your system:
```bash
$ sudo apt install python3-pip
```
> Install Python3-venv if it is already not installed on your system:
```bash
$ sudo apt install -y python3-venv
```
> To create a virtual environment, decide upon a directory where you want to place it, and run the venv module as a script with the directory path (Note: the second **venv** is the name of your virtual environment, so you can choose the one you want):
```bash
$ python3 -m venv venv
```
> Once you’ve created a virtual environment, you may activate it and run it:
```bash
$ source venv/bin/activate
```
> Install the requirements file:
```bash
(venv) $ pip3 install -r requirements.txt
```

## Requirements 
This code was tested with the following tools: 
* Python 3.8.10 
* [Open3D](http://www.open3d.org/docs/release/index.html) 0.13.0 (which includes the installation of Numpy and more) 
* [Trimesh](https://github.com/mikedh/trimesh) 3.9.27
* Pytorch3D 0.3.0 (which includes the installation of torchvision) 
* [Vedo](https://github.com/marcomusy/vedo) 2021.0.3
* [VTK](https://kitware.github.io/vtk-examples/site/Python/) 9.0.3
* [PyVista](https://docs.pyvista.org/getting-started/why.html) 0.31.3

> Install all the requirements for this project:
```bash
pip3 install -r requirements.txt
```
> Installing torch==1.9.0 
```bash
pip3 install torch
``` 
> Check the requirements:
```bash
pip3 freeze
```

## Settings
The configuration files are in the **/config_files** folder and the name is **local_config.ini**.
```bash
cat /settings/config_files/local_config.ini
```
* The default values are indicated by 'DEFAULT'. Example:
```
[DEFAULT]
img_path = /Datasets/img
mesh_path = /Datasets/mesh
index = 0
```

# Concepts

## Watertight mesh
In computer graphics, a watertight mesh refers to a triangular mesh consisting of one closed surface. This loose definition rules out the following two cases which frequently cause problems when voxelizing meshes into occupancy grids or signed distance functions: 

* The mesh has holes which prevent a clear definition of inside and outside; 
* or the mesh contains structure within the main surface, hindering signed distance function computation.

However, automatically obtaining watertight meshes is difficult. First of all, it is non-trivial to fix larger holes; and second, irrelevant inner structures are difficult to identify.

[More info](https://www.sculpteo.com/en/3d-learning-hub/create-3d-file/fix-non-manifold-geometry/)