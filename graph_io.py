import numpy as np
import torch
from torch_geometric.nn import knn
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, knn_graph

import graph_helpers as gh
import sphere_helpers as sh
import mesh_helpers as mh
import clusters as cl
import utils

from torch_scatter import scatter

import math
from math import pi, sqrt

from warnings import warn

def image2Graph(data, gt = None, mask = None, depth = 1, x_only = False, device = 'cpu'):
    
    _,ch,rows,cols = data.shape
    
    x = torch.reshape(data,(ch,rows*cols)).permute((1,0)).to(device)
    
    if mask is not None:
        # Mask out nodes
        node_mask = torch.where(mask.flatten())
        x = x[node_mask]
    
    if gt is not None:
        y = gt.flatten().to(device)
        if mask is not None:
            y = y[node_mask]
    
    if x_only:
        if gt is not None:
            return x,y
        else:
            return x
    
    im_pos = gh.getImPos(rows,cols)
    
    if mask is not None:
        im_pos = im_pos[node_mask]
    
    # Make "point cloud" for clustering
    pos2D = gh.convertImPos(im_pos,flip_y=False)
    
    # Generate initial graph
    edge_index = gh.grid2Edges(pos2D)
    directions = pos2D[edge_index[1]] - pos2D[edge_index[0]]
    selections = gh.edges2Selections(edge_index,directions,interpolated=False,y_down=True)
    
    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list = cl.makeImageClusters(pos2D,cols,rows,edge_index,selections,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=None)
    metadata = Data(original=data,im_pos=im_pos.long(),rows=rows,cols=cols,ch=ch)
    
    if gt is not None:
        graph.y = y
    
    return graph,metadata

def graph2Image(result,metadata,canvas=None):
    
    x = utils.toNumpy(result,permute=False)
    im_pos = utils.toNumpy(metadata.im_pos,permute=False)
    if canvas is None:
        canvas = utils.makeCanvas(x,metadata.original)
    
    # Paint over the original image (neccesary for masked images)
    canvas[im_pos[:,0],im_pos[:,1]] = x
    
    return canvas

### Begin Interpolated Methods ###

def sphere2Graph(data, structure="layering", cluster_method="layering", scale=1.0, stride=2, interpolation_mode = "angle", gt = None, mask = None, depth = 1, x_only = False, device = 'cpu'):
    
    _,ch,rows,cols = data.shape
    
    if structure == "equirec":
        # Use the original data to start with
        cartesian, spherical = sh.sampleSphere_Equirec(scale*rows,scale*cols)
    elif structure == "layering":
        cartesian, spherical = sh.sampleSphere_Layering(scale*rows)
    elif structure == "spiral":
        cartesian, spherical = sh.sampleSphere_Spiral(scale*rows,scale*cols)
    elif structure == "icosphere":
        cartesian, spherical = sh.sampleSphere_Icosphere(scale*rows)
    elif structure == "random":
        cartesian, spherical = sh.sampleSphere_Random(scale*rows,scale*cols)
    else:
        raise ValueError("Sphere structure unknown")
    
    if interpolation_mode == "bary":            
        bary_d = pi/(scale*rows)
    else:
        bary_d = None
    
    # Get the landing point for each node
    sample_x, sample_y = sh.spherical2equirec(spherical[:,0],spherical[:,1],rows,cols)
    
    if mask is not None:

        node_mask = gh.maskPoints(mask,sample_x,sample_y)
        sample_x = sample_x[node_mask]
        sample_y = sample_y[node_mask]
        spherical = spherical[node_mask]
        cartesian = cartesian[node_mask]
    
    features = utils.bilinear_interpolate(data, sample_x, sample_y).to(device)
    
    if gt is not None:
        features_y = utils.bilinear_interpolate(gt.unsqueeze(0), sample_x, sample_y).to(device)
    
    if x_only:
        if gt is not None:
            return features,features_y
        else:
            return features
        
    # Build initial graph
    edge_index,directions = gh.surface2Edges(cartesian,cartesian)
    edge_index,selections,interps = gh.edges2Selections(edge_index,directions,interpolated=True,bary_d=bary_d)
    
    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list, interps_list = cl.makeSphereClusters(cartesian,edge_index,selections,interps,rows*scale,cols*scale,cluster_method,stride=stride,bary_d=bary_d,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=features,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=interps_list)
    metadata = Data(original=data,pos3D=cartesian,mask=mask,rows=rows,cols=cols,ch=ch)

    if gt is not None:
        graph.y = features_y   
    
    return graph, metadata
    
def graph2Sphere(features,metadata):
    
    # Generate equirectangular points and their 3D locations
    theta, phi = sh.equirec2spherical(metadata.rows, metadata.cols)
    x,y,z = sh.spherical2xyz(theta,phi)
    
    v = torch.stack((x,y,z),dim=1)
    
    # Find closest 3D point to each equirectangular point
    nearest = torch.reshape(knn(metadata.pos3D,v,3)[1],(len(v),3))
    
    #Interpolate based on proximty to each node
    w0 = 1/torch.linalg.norm((v - metadata.pos3D[nearest[:,0]]),dim=1, keepdim=True).to(features.device)
    w1 = 1/torch.linalg.norm((v - metadata.pos3D[nearest[:,1]]),dim=1, keepdim=True).to(features.device)
    w2 = 1/torch.linalg.norm((v - metadata.pos3D[nearest[:,2]]),dim=1, keepdim=True).to(features.device)
    
    w0 = torch.nan_to_num(w0, nan=1e6)
    w1 = torch.nan_to_num(w1, nan=1e6)
    w2 = torch.nan_to_num(w2, nan=1e6)
    
    w0 = torch.clamp(w0,0,1e6)
    w1 = torch.clamp(w1,0,1e6)
    w2 = torch.clamp(w2,0,1e6)
    
    total = w0 + w1 + w2
    
    #w0,w1,w2 = mh.getBarycentricWeights(v,metadata.pos3D[nearest[:,0]],metadata.pos3D[nearest[:,1]],metadata.pos3D[nearest[:,2]])
    
    #w0 = w0.unsqueeze(1).to(features.device)
    #w1 = w1.unsqueeze(1).to(features.device)
    #w2 = w2.unsqueeze(1).to(features.device)
        
    result = (w0*features[nearest[:,0]] + w1*features[nearest[:,1]] + w2*features[nearest[:,2]])/total
    
    #result = result.clamp(0,1)

    if hasattr(metadata,"mask"):
        mask = utils.toNumpy(metadata.mask.squeeze(),permute=False)
        canvas = utils.makeCanvas(result,metadata.original)
        result = np.reshape(result.data.cpu().numpy(),(metadata.rows,metadata.cols,features.shape[1]))
        canvas[np.where(mask)] = result[np.where(mask)]
        return canvas
    else:
        return np.reshape(result.data.cpu().numpy(),(metadata.rows,metadata.cols,features.shape[1]))



def splat2Graph(data, mesh, up_vector = None, N = 100000, ratio=.25, depth = 1, device = 'cpu'):
    """ Sample mesh faces to determine graph """

    if up_vector == None:
        up_vector = torch.tensor([[1,1,1]],dtype=torch.float)
        #up_vector = 2*torch.rand((1,3))-1
        up_vector = up_vector/torch.linalg.norm(up_vector,dim=1)


    #position, normal vector, uv coordinates in the texture map, x is color
    pos3D, normals = mh.sampleSurface(mesh,N)
        
    # Build initial graph
    #edge_index are neighbors of a point, directions are the directions from that point
    edge_index,directions = gh.surface2Edges(pos3D,normals,up_vector,k_neighbors=16)
    #directions need to be turned into selections "W sub n" from the star-like coordinate system from Dr. Hart's github interpolated-selectionconv
    edge_index,selections,interps = gh.edges2Selections(edge_index,directions,interpolated=True)

    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list, interps_list = cl.makeSurfaceClusters(pos3D,normals,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)
    #clusters, edge_indexes, selections_list, interps_list = cl.makeMeshClusters(pos3D,mesh,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=interps_list)
    metadata = Data(original=data,pos3D=pos3D,mesh=mesh)
    
    return graph,metadata

def mesh2Graph(data, mesh, up_vector = None, N = 100000, ratio=.25, mask = None, depth = 1, x_only = False, device = 'cpu'):
    """ Sample mesh faces to determine graph """

    if up_vector == None:
        up_vector = torch.tensor([[1,1,1]],dtype=torch.float)
        #up_vector = 2*torch.rand((1,3))-1
        up_vector = up_vector/torch.linalg.norm(up_vector,dim=1)

    if mask is not None:
        warn("Masks are not currently implemented for mesh graphs")      
    
    #position, normal vector, uv coordinates in the texture map, x is color
    pos3D, normals, uvs, x = mh.sampleSurface(mesh,N,return_x=True)
    
    x = x.to(device)
    
    if x_only:
        warn("x_only returns randomly selected points for mesh2Graph. Do not use with previous graph structures")
        return x
    
    # Build initial graph
    #edge_index are neighbors of a point, directions are the directions from that point
    edge_index,directions = gh.surface2Edges(pos3D,normals,up_vector,k_neighbors=16)
    #directions need to be turned into selections "W sub n" from the star-like coordinate system from Dr. Hart's github interpolated-selectionconv
    edge_index,selections,interps = gh.edges2Selections(edge_index,directions,interpolated=True)

    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list, interps_list = cl.makeSurfaceClusters(pos3D,normals,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)
    #clusters, edge_indexes, selections_list, interps_list = cl.makeMeshClusters(pos3D,mesh,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=interps_list)
    metadata = Data(original=data,pos3D=pos3D,uvs=uvs,mesh=mesh)
    
    return graph,metadata

def graph2Splat(features,metadata,view3D=False):

    features = features.cpu().numpy()
    
    canvas = utils.toNumpy(metadata.original)
    rows,cols,ch = canvas.shape
    
    # Get 2D positions by scaling uv
    pos2D = metadata.uvs.cpu().numpy()
    pos2D[:,0] = pos2D[:,0]*cols
    pos2D[:,1] = 1-pos2D[:,1] # UV puts y=0 at the bottom
    pos2D[:,1] = pos2D[:,1]*rows
    
    # Generate desired points
    row_space = np.arange(rows)
    col_space = np.arange(cols)
    col_image,row_image = np.meshgrid(col_space,row_space)
    
    canvas = utils.interpolatePointCloud2D(pos2D,features,col_image,row_image)
    canvas = np.clip(canvas,0,1)

    if view3D:
        mesh = mh.setTexture(metadata.mesh,canvas)
        mesh.show()
    
    return canvas


def graph2Mesh(features,metadata,view3D=False):
    
    features = features.cpu().numpy()
    
    canvas = utils.toNumpy(metadata.original)
    rows,cols,ch = canvas.shape
    
    # Get 2D positions by scaling uv
    pos2D = metadata.uvs.cpu().numpy()
    pos2D[:,0] = pos2D[:,0]*cols
    pos2D[:,1] = 1-pos2D[:,1] # UV puts y=0 at the bottom
    pos2D[:,1] = pos2D[:,1]*rows
    
    # Generate desired points
    row_space = np.arange(rows)
    col_space = np.arange(cols)
    col_image,row_image = np.meshgrid(col_space,row_space)
    
    canvas = utils.interpolatePointCloud2D(pos2D,features,col_image,row_image)
    canvas = np.clip(canvas,0,1)

    if view3D:
        mesh = mh.setTexture(metadata.mesh,canvas)
        mesh.show()
    
    return canvas
