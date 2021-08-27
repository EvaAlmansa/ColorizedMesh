import torch
import torch.nn as nn
import numpy as np


def xy2uv(xyz, eps = 0.001):
    x, y, z = torch.unbind(xyz, dim=2)

    x = x+eps
    y = y+eps
    z = z+eps
            
    u = torch.atan2(x, -y)
    v = - torch.atan(z / torch.sqrt(x**2 + y**2)) ###  (default: - for z neg (under horizon) - grid sample instead expects -1,-1 top-left
        
    pi = float(np.pi)##torch.acos(torch.zeros(1)).item() * 2

    u = u / pi
    v = (2.0 * v) / pi 

    u = torch.clamp(u, min=-1, max=1)
    v = torch.clamp(v, min=-1, max=1)
        
    ###output: [batch_size x num_points x 2]##range -1,+1

    output = torch.stack([u, v], dim=-1) 
                        
    return output

def xyz2uvd(xyz, eps = 0.001):
    x, y, z = torch.unbind(xyz, dim=2)
        
    x = x+eps
    y = y+eps
    z = z+eps
    
    u = torch.atan2(x, -y)
    v = - torch.atan(z / torch.sqrt(x**2 + y**2))

    pi = float(np.pi)##torch.acos(torch.zeros(1)).item() * 2

    u = u / pi
    v = (2.0 * v) / pi

    xx = x * x
    yy = y * y
    zz = z * z

    d = torch.sqrt(xx+yy+zz)

    u = torch.clamp(u, min=-1, max=1)
    v = torch.clamp(v, min=-1, max=1)
    d = torch.clamp(d, min=0, max=16000)
        
    ###output: [batch_size x num_points x 3]

    output = torch.stack([u, v, d], dim=-1)
                
    return output

def uvd2xyz(uvd):
    u, v, d = torch.unbind(uvd, dim=2)

    pi = float(np.pi)##torch.acos(torch.zeros(1)).item() * 2

    theta = (pi * u) - np.pi/2.0
    phi   = (pi/2.0) * -v #####invert v axis to recover angles 

    x = d * torch.cos(phi) * torch.cos(theta)
    y = d * torch.cos(phi) * torch.sin(theta)
    z = d * torch.sin(phi)

    output = torch.stack([x, y, z], dim=-1)
                
    return output