
# Description
This repository contains two general functions where is possible to get:

| Function |  Input |  Result |
|:-----------|:------------|:--------------|
| A projection of mesh to equirectangular image (JPG file): [projection_mesh_to_image](https://github.com/EvaAlmansa/ColorizedMesh/blob/master/mesh/manager.py#L188) | 360 image (1024x512 px) ![equi_img](/docs/living_room4_1.jpg) Boundary Mesh ![mesh](/docs/living_room4_1_mesh.png)<!-- .element height="50%" width="50%" -->  | Projected Mesh to 360º image ![project_mesh2equi](/docs/living_room4_1_projected.jpg) |
| A occluded point map from mesh and furniture (PLY file): [occluded_point_map](https://github.com/EvaAlmansa/ColorizedMesh/blob/master/colorized_mesh/colorized_mesh.py#140) | Boundary Mesh + Furniture![compl_mesh](/docs/living_room4_1_furniture_mesh.png) | Occluded Point Map ![occluded_point_map](/docs/living_room4_1_occl_map.jpg) Statistics about number of occluded points (CSV file) [statistics.csv](https://github.com/EvaAlmansa/ColorizedMesh/blob/master/result/statistics.csv) |


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
├── docs
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

## Triangle mesh 
[Reference](http://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Mesh-properties)

A triangle mesh has several properties that can be tested:
* the manifold property, where we can test the triangle mesh if it is edge manifold is_edge_manifold and if it is is_vertex_manifold. 
* A triangle mesh is edge manifold, if each edge is bounding either one or two triangles. 
* Triangle mesh is vertex manifold if the star of the vertex is edge-manifold and edge-connected, e.g., two or more faces connected only by a vertex and not by an edge. Another property is the test of self-intersection. 
* A watertight mesh can be defined as a mesh that is edge manifold, vertex manifold and not self intersecting. 
* if it is orientable, i.e. the triangles can be oriented in such a way that all normals point towards the outside. 

[More info](https://www.sculpteo.com/en/3d-learning-hub/create-3d-file/fix-non-manifold-geometry/)

There is a function to check those properties in this project [here](https://github.com/EvaAlmansa/ColorizedMesh/blob/master/mesh/manager.py#L82).