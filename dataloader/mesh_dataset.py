import os
import numpy as np

import torch
import torch.utils.data as data
import trimesh

from PIL import Image


class MeshDataset(data.Dataset):   

    def __init__(self, args):
        
        self.img_path = args.img_path
        self.mesh_path = args.mesh_path
        self.furniture_path = args.furniture_path
        self.mesh_name = args.mesh_name

        self.v_pad_max = args.v_pad_max
        self.f_pad_max = args.f_pad_max

        self.return_name = args.return_name
        self.return_valid = args.return_valid

        self._read_name_data()
        
        self._check_dataset()
        
    def __getitem__(self, idx):      
        img_path = os.path.join(self.img_path,
                                self.img_fnames[idx])

        mesh_path = os.path.join(self.mesh_path,
                                self.mesh_fnames[idx])

        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.

        # Convert all data to tensor
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        out_lst = [x]
        
        mesh = trimesh.load_mesh(mesh_path)
        V = mesh.vertices
        F = mesh.faces
        valid_mesh = mesh.is_volume
                       
        # pad V and F
        Vt = torch.FloatTensor(V)                                       
        Ft = torch.FloatTensor(F)

        if (self.v_pad_max is not None) and (self.f_pad_max is not None):
            verts_padded = list_to_padded(Vt, self.v_pad_max, pad_value=0.0)
            faces_padded = list_to_padded(Ft, self.f_pad_max, pad_value=-1.0)
                                             
            out_lst.append(verts_padded)
            out_lst.append(faces_padded)
        else:
            out_lst.append(Vt)
            out_lst.append(Ft)

        if self.return_name:
            out_lst.append(self.mesh_fnames[idx])
            out_lst.append(img_path)

        if self.return_valid:
            is_valid = ( (len(Vt) > 0) and (len(Ft) > 0) and torch.isfinite(Vt).all() and torch.isfinite(Ft).all() and valid_mesh)
            out_lst.append(is_valid)
        
        return out_lst

    def __len__(self):
        return len(self.img_fnames)

    def _check_dataset(self):
        for fname in self.mesh_fnames:
            assert os.path.isfile(os.path.join(self.mesh_path, fname)),\
                '%s not found' % os.path.join(self.mesh_path, fname)
            if self.furniture_path:
                assert os.path.isfile(os.path.join(self.furniture_path, fname)),\
                '%s not found' % os.path.join(self.furniture_path, fname)

    def _read_name_data(self):

        if not self.mesh_name:
            self.img_fnames = sorted([
                fname for fname in os.listdir(self.img_path)
            ])
        else:
            self.img_fnames = [fname for fname in os.listdir(self.img_path) if self.mesh_name[:-4] == fname[:-4]]

        self.mesh_fnames = ['%sobj' % fname[:-3] for fname in self.img_fnames] 