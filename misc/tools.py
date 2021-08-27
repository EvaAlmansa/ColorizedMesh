import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import os
import json
from collections import OrderedDict

##from misc import config, post_proc, sobel

import torch.nn.functional as F

import math

PI = float(np.pi)

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI

def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, m_ratio = 1.0):
    '''
    coor: N x 2, index of array in (col, row) format eg. 1024x2
    m_ratio: pixel/cm ratio for tensor fitting
    '''           
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)       
    v = np_coory2v(coor[:, 1], coorH)
                  
    c = z / np.tan(v)
    x = m_ratio * c * np.sin(u) + floorW / 2 - 0.5
    y = -m_ratio *c * np.cos(u) + floorH / 2 - 0.5
    
    return np.hstack([x[:, None], y[:, None]])


def resize_crop(img, scale, size):
    
    re_size = int(img.shape[0]*scale)

    if(re_size>0):
        img = cv2.resize(img, (re_size, re_size), cv2.INTER_AREA)

    if size <= re_size:
        pd = int((re_size-size)/2)
        img = img[pd:pd+size,pd:pd+size]
    else:
        new = np.zeros((size,size))
        pd = int((size-re_size)/2)
        new[pd:pd+re_size,pd:pd+re_size] = img[:,:]
        img = new

    return img

def resize(img, scale):
    
    re_size = int(img.shape[0]*scale)
    if(re_size>0):
        img = cv2.resize(img, (re_size, re_size), cv2.INTER_CUBIC)
        
    return img

def var2np(data_lst):

    def trans(data):
        if data.shape[1] == 1:
            return data[0, 0].data.cpu().numpy()
        elif data.shape[1] == 3: 
            return data[0, :, :, :].permute(1, 2, 0).data.cpu().numpy()

    if isinstance(data_lst, list):
        np_lst = []
        for data in data_lst:
            np_lst.append(trans(data))
        return np_lst
    else:
        return trans(data_lst)

def save_map(img,name):
    
    vis = Image.fromarray(np.uint8(img* 255))
    vis.save(name)

def approx_shape(data, fp_threshold=0.5, epsilon_b=0.005, rel_threshold=0.5, return_reliability=False):
    data_c = data.copy()
    ret, data_thresh = cv2.threshold(data_c, fp_threshold, 1, 0)
    data_thresh = np.uint8(data_thresh)
    
    data_cnt, data_heri = cv2.findContours(data_thresh, 1, 2)##CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE
    ##data_cnt, data_heri = cv2.findContours(data_thresh, 0, 2)##CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE
        
    reliability = 0.0
    approx = np.empty([1, 1, 2])
    
    if(len(data_cnt)>0):    
        # Find the the largest connected component and its bounding box
        data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)
        
        area0 = cv2.contourArea(data_cnt[0])

        if(len(data_cnt)>1 and (area0 > 0)):
            area1 = cv2.contourArea(data_cnt[1])
            reliability = 1.0 - (area1/area0)
        else:
            reliability = 1.0
     
    if(reliability<rel_threshold and len(data_cnt)>1):
        mergedlist = np.concatenate((data_cnt[0], data_cnt[1]), axis=0)
        approx = cv2.convexHull(mergedlist)

    else:
        epsilon = epsilon_b*cv2.arcLength(data_cnt[0], True)
        approx = cv2.approxPolyDP(data_cnt[0], epsilon,True)

    if return_reliability:
        ap_area = cv2.contourArea(approx)
        return approx,reliability,ap_area
    else:
        return approx

def metric_scale(height, camera_h, fp_meter, fp_size):
        
    scale = 100 * ((height - camera_h) / camera_h) * (fp_meter / fp_size)
        
    return scale

def x2image(x):
    img = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)

    return img

def recover_h_value(mask):
    return np.amax(mask.numpy())


def group_weight(module):
    # Group module parameters into two group
    # One need weight_decay and the other doesn't
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]

def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        scale_running_lr = max((1. - frac), 0.) ** args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr

def save_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.backbone,
            'full_size': net.full_size,
            'decoder_type': net.decoder_type,
            },
        'state_dict': net.state_dict(),
    })
    torch.save(state_dict, path)

def save_parallel_model(net, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.module.backbone,
            'full_size': net.module.full_size,
            'decoder_type': net.module.decoder_type,
            },
        'state_dict': net.module.state_dict()
    })
    torch.save(state_dict, path)

def save_parallel_checkpoint(net, optimizer, epoch, path, args):
    state_dict = OrderedDict({
        'args': args.__dict__,
        'kwargs': {
            'backbone': net.module.backbone,
            'full_size': net.module.full_size,
            },
        'state_dict': net.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1
    })
    torch.save(state_dict, path)

def load_trained_model(Net, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    return net

def load_checkpoint(Net, optimizer, path):
    state_dict = torch.load(path, map_location='cpu')
    net = Net(**state_dict['kwargs'])
    net.load_state_dict(state_dict['state_dict'])
    ###to do pass model parameters to optimizer
    optimizer.load_state_dict(state_dict['optimizer'])
    return net,optimizer,state_dict['epoch']

def export2json(c_pts, W, H, fp_size, output_dir,def_img,k, z0, z1):
    c_pts = c_pts.squeeze(1)            
                       
                       
    c_cor = post_proc.np_xy2coor(np.array(c_pts), z0, W, H, fp_size, fp_size)
    f_cor = post_proc.np_xy2coor(np.array(c_pts), z1, W, H, fp_size, fp_size) ####based on the ceiling shape

    cor_count = len(c_cor)                                                        
            
    c_ind = np.lexsort((c_cor[:,1],c_cor[:,0])) 
    f_ind = np.lexsort((f_cor[:,1],f_cor[:,0]))
            
    ####sorted by theta (pixels coords)
    c_cor = c_cor[c_ind]
    f_cor = f_cor[f_ind]
                       
    cor_id = []

    for j in range(len(c_cor)):
        cor_id.append(c_cor[j])
        cor_id.append(f_cor[j])
   
    cor_id = np.array(cor_id)
              
                                    
    # Normalized to [0, 1]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H
                        
    # Output result
    with open(os.path.join(output_dir, k + '.json'), 'w') as f:
        json.dump({
        'z0': float(z0),
        'z1': float(z1),
        'uv': [[float(u), float(v)] for u, v in cor_id],
        }, f)
            
    ##store json full path name for 3D visualization - NB only uv and z are stored
    json_name = os.path.join(output_dir, k + '.json')

    return json_name

def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi

def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi

def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs)

    return np.stack([coorxs, coorys], axis=-1)    

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys

def createPointCloud(color, depth, ply_file):
    color = color.permute(1, 2, 0)
    print(color.shape)
    print(depth.shape)
    ### color:np.array (h, w)
    ### depth: np.array (h, w)

    pcSampleStride = 30

    heightScale = float(color.shape[0]) / depth.shape[0]
    widthScale = float(color.shape[1]) / depth.shape[1]

    points = []
    for i in range(color.shape[0]):
        if not i % pcSampleStride == 0:
            continue
        for j in range(color.shape[1]):
            if not j % pcSampleStride == 0:
                continue

            rgb = (color[i][j][0], color[i][j][1], color[i][j][2])

            d = depth[ int(i/heightScale) ][ int(j/widthScale) ]
            if d <= 0:
                continue

            coordsX = float(j) / color.shape[1]
            coordsY = float(i) / color.shape[0]

            xyz = coords2xyz((coordsX, coordsY) ,d)

            ##point = (xyz, rgb)
            ##pointCloud.append(point)

            points.append("%f %f %f %d %d %d 0\n"%(xyz[0],xyz[1],xyz[2],rgb[0],rgb[1],rgb[2]))

        file = open(ply_file,"w")
        file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        '''%(len(points),"".join(points)))
        file.close()

        
        #if i % int(color.shape[0]/10) == 0:
        #    print("PC generating {0}%".format(i/color.shape[0]*100))
    
    return points

def coords2xyz(coords, N):

    uv = coords2uv(coords)
    xyz = uv2xyz(uv, N)
    
    return xyz

def coords2uv(coords):  
    #coords: 0.0 - 1.0
    coords = (coords[0] - 0.5, coords[1] - 0.5)

    uv = (coords[0] * 2 * math.pi,
            -coords[1] * math.pi)

    return uv

def uv2xyz(uv, N):

    x = math.cos(uv[1]) * math.sin(uv[0])
    y = math.sin(uv[1])
    z = math.cos(uv[1]) * math.cos(uv[0])
    ##Flip Z　axis
    xyz = (N * x, N * y, -N * z)

    return xyz

def SphereGrid(equ_h, equ_w):
    cen_x = (equ_w - 1) / 2.0
    cen_y = (equ_h - 1) / 2.0
    theta = (2 * (np.arange(equ_w) - cen_x) / equ_w) * np.pi
    phi = (2 * (np.arange(equ_h) - cen_y) / equ_h) * (np.pi / 2)
    theta = np.tile(theta[None, :], [equ_h, 1])
    phi = np.tile(phi[None, :], [equ_w, 1]).T

    x = (np.cos(phi) * np.sin(theta)).reshape([equ_h, equ_w, 1])
    y = (np.sin(phi)).reshape([equ_h, equ_w, 1])
    z = (np.cos(phi) * np.cos(theta)).reshape([equ_h, equ_w, 1])
    xyz = np.concatenate([x, y, z], axis=-1)

    return xyz

def depth2pts(depth):
    grid = SphereGrid(*depth.shape) ### (h,w)
    pts = depth[..., None] * grid

    return pts

def image_depth_to_world(d):
    ##print('tools',d.shape)
    P = np.zeros(shape =(d.shape[0],d.shape[1],3),dtype=float)
    for i in range(d.shape[0]):
        theta = -np.pi * (float(i)/float(d.shape[0]-1)-0.5)
        for j in range(d.shape[1]):
            # check if there is replication
            phi = np.pi * (2.0*float(j)/float(d.shape[1]-1)-1.0) 
                                  
            P[i,j,0] = d[i,j]*math.cos(phi)*math.cos(theta)
            P[i,j,1] = d[i,j]*math.sin(theta)
            P[i,j,2] = d[i,j]*math.sin(phi)*math.cos(theta)
            
    return P

def depth_pixel_to_world(d, i, j):
    P = np.zeros(shape =(1, 1, 3),dtype=float)
       
    theta = -np.pi * (float(i)/float(d.shape[0]-1)-0.5)
    phi = np.pi * (2.0*float(j)/float(d.shape[1]-1)-1.0) 
    
    P[i,j,0] = d[i,j]*math.cos(phi)*math.cos(theta)
    P[i,j,1] = d[i,j]*math.sin(theta)
    P[i,j,2] = d[i,j]*math.sin(phi)*math.cos(theta)
            
    return P


def export_obj(outfile, P, rgb):
    rgb = 255.0 * rgb
    #P = 65535.0 * P 
    f = open(outfile, "w")
    f.write('# obj point cloud')
    for i in range (P.shape[0]):
        for j in range (P.shape[1]):  
            d  = P[i,j,0]**2 + P[i,j,1]**2 + P[i,j,2]**2
            if d > 1.0e-6:
                f.write('v %f %f %f %f %f %f \n'%(P[i,j,0],P[i,j,1],P[i,j,2],rgb[i,j,0],rgb[i,j,1],rgb[i,j,2]))
    f.close()

def export_model(img_depth,img_rgb,out_file):
    print('rgb',img_rgb.shape)
    print('depth',img_depth.shape)
    P = image_depth_to_world(img_depth)
    export_obj(out_file,P,img_rgb)

def export_from_batch(xyz_batch,img_rgb,out_file):
    print('rgb',img_rgb.shape)
    print('xyz', xyz_batch.shape)
    
    P = xyz_batch.squeeze(0)
    P = P.permute(1,2,0)

    export_obj(out_file, P, img_rgb)

def imgrad(img):
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x

def imgrad_yx(img):
    ##Sobel implementation for C multi-channel images (eg. RGB-D)
    ### unsqueeze to put 1 channel C
    img = img.unsqueeze(1)
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
                
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def depth2normals(depth):
    ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float()
    
    get_gradient = sobel.Sobel()

    depth_grad = get_gradient(depth)
       
    depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
    depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
    
    depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
    depth_normal = F.normalize(depth_normal, p=2, dim=1)
    
    return depth_normal





