import torch
import pyntcloud_io as plyio
from torch_geometric.nn import knn
import numpy as np
from torch_scatter import scatter_mean
import plyio as splatio
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
    
def splat_update_and_save(rawdata, results, fileName):


    #df1[['f_dc_0','f_dc_1','f_dc_2']] = results
    #plyNumpy = df1.to_numpy()
    #plyNumpy.tofile('Scaniverse_chair_testOut2.ply')

    #plyNumpy = df1.to_numpy()
    #np.savetxt('Scaniverse_chair_test.csv', plyNumpy, delimiter=',')

    #SH_C0 = 0.28209479177387814
    #normedResults = 0.5 - results/SH_C0

    #df1[['f_dc_0','f_dc_1','f_dc_2']] = normedResults


    #plyNumpy = df1.to_numpy()
    #np.savetxt('Scaniverse_chair_testOut.csv', plyNumpy, delimiter=',')
    #plyNumpy = df1.to_numpy()
    #plyNumpy.tofile('Scaniverse_chair_testOutNormed.ply')


    outFileName='Scaniverse_chair_outputFloat.ply'
    df1 = pd.DataFrame(rawdata, columns=['x','y','z',
                                         'nx','ny','nz',
                                         'f_dc_0','f_dc_1','f_dc_2',
                                         'opacity',
                                         'scale_0','scale_1','scale_2',
                                         'rot_0','rot_1','rot_2','rot_3'
                                         ])

    plyio.write_ply_float(filename=outFileName, points= df1, mesh=None, as_text=False)

    
    outFilePath='C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Scaniverse_out.splat'
    opacity = rawdata[:,-1].reshape(224713, 1)
    positions = rawdata[:,0:3]
    scales = rawdata[:,3:6]
    rots = rawdata[:,6:10]
    colors = results

    splatio.numpy_to_splat(positions, scales, rots, colors, opacity, outFilePath)

    #save as csv
    normalized = False
    #normalize the color values if needed
    if (normalized):
        SH_C0 = 0.28209479177387814
        colors = colors/SH_C0-0.5
        opacity = -np.log(1/(opacity-1))
        
    colors = np.concatenate((colors, opacity), axis=1)
    splat = np.concatenate((positions, scales, rots, colors), axis=1)
    #np.savetxt('C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Scaniverse_out.csv', splat, delimiter=',', header=splat.dtype.names)
    np.savetxt('C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Scaniverse_out.csv', splat, delimiter=',')

    return 

    
def splat_save(positions, scales, rots, colors, output_path):



    splatio.numpy_to_splat(positions, scales, rots, colors, output_path)

    return 






def splat_unpacker(neighbors, fileName, removeTopQ = 0):

    '''
    #Get the raw PLY data into a tensor
    plyout = plyio.read_ply('Scaniverse_chair_StyleOutput.ply')
    df1out = plyout.get('points')

    #Get the raw PLY data into a tensor
    ply = plyio.read_ply(fileName)
    df1 = ply.get('points')
    f1 = df1.to_numpy()
    f2 = df1out.to_numpy()
    '''

    positions, scales, rots, colors = splatio.ply_to_numpy(fileName)
    
    
    #Get the raw PLY data into a tensor
    torchPoints = torch.Tensor(positions)
    samples = torchPoints.size()[0]
    
    y = torchPoints
    x = torchPoints
    assign_index = knn(x, y, neighbors)

    indexSrc = assign_index[0:1][0]
    indexSrcTrn = indexSrc.reshape(-1,1)
    index = indexSrcTrn.expand(indexSrc.size()[0],3)

    src = x[assign_index[1:2]][0]
    out = src.new_zeros(src.size())
    out = scatter_mean(src, index, 0, out=out)

    diffvector = y - out[0:samples]

    diffvectorNum =diffvector.numpy()
    
    normals = torch.from_numpy(diffvectorNum)
    
    pos3D = torch.from_numpy(positions)

    torchColors = torch.Tensor(colors)
    torchColors.clamp(0,1)
    
    return pos3D, normals, colors, scales, rots



def splat_downsampler(pos3D, colors, scaleV):

    pointsNP = pos3D.numpy()
    colorsNP = colors.numpy()

    #cluster
    kmeans = KMeans(n_clusters=scaleV*256, random_state=0, n_init="auto").fit(pointsNP)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_


    x = np.indices(pointsNP.shape)[0]
    y = np.indices(pointsNP.shape)[1]
    z = np.indices(pointsNP.shape)[2]
    col = pointsNP.flatten()

    # 3D Plot
    fig = plt.figure()
    ax3D = fig.add_subplot(projection='3d')
    cm = plt.colormaps['brg']
    p3d = ax3D.scatter(x, y, z, c=col, cmap=cm)
    plt.colorbar(p3d)
    plt.colorbar(p3d)



    return pos3D, colors



def splat_unpacker_threshold(neighbors, fileName, threshold):

    #check if .ply or .splat
    if fileName[-3:] == 'ply':
        positionsFile, scalesFile, rotsFile, colorsFile = splatio.ply_to_numpy(fileName)
        fileType = 'ply'
    else:
        positionsFile, scalesFile, rotsFile, colorsFile = splatio.splat_to_numpy(fileName)
        fileType = 'splat'


    
    positionsNP = positionsFile.copy()
    scalesNP = scalesFile.copy()    
    rotsNP = rotsFile.copy()    
    colorsNP = colorsFile.copy()
    

    #Get the raw PLY data into tensors
    pos3D = torch.from_numpy(positionsNP)
    colors = torch.from_numpy(colorsNP)
    colors.clamp(0,1)
    rots = torch.from_numpy(rotsNP)
    scales = torch.from_numpy(scalesNP)
    points = torch.from_numpy(positionsNP)

    samples = points.size()[0]
    y = points
    x = points
    
    #knn find midpoint
    assign_index = knn(x, y, neighbors)
    indexSrc = assign_index[0:1][0]
    indexSrcTrn = indexSrc.reshape(-1,1)
    index = indexSrcTrn.expand(indexSrc.size()[0],3)

    src = x[assign_index[1:2]][0]
    out = src.new_zeros(src.size())
    out = scatter_mean(src, index, 0, out=out)

    #calculate normals
    normals = y - out[0:samples]
    

    #use Euclidian distances to create a mask
    #distances = torch.sqrt((out[0:samples][:,0])**2 +(out[0:samples][:,1])**2+(out[0:samples][:,2])**2 )
    distances = torch.sqrt((y[:,0] - out[0:samples][:,0])**2 +(y[:,1] - out[0:samples][:,1])**2+(y[:,2] - out[0:samples][:,2])**2 )
    threshold = np.clip(threshold, 0, 100)
    boundry = np.percentile(distances, threshold)
    mask = distances < boundry

    
    pos3DFiltered = pos3D[mask]
    normalsFiltered = normals[mask]    
    colorsFiltered = colors[mask]     
    scalesFiltered = scales[mask]
    rotsFiltered = rots[mask]

    return pos3DFiltered, normalsFiltered, colorsFiltered[:,0:3], colorsFiltered[:,3], scalesFiltered, rotsFiltered, fileType


def splat_unpacker_threshold_graph_normals(neighbors, fileName, threshold):

    coneSize = 1

    positionsNP, scalesNP, rotsNP, colorsNP = splatio.ply_to_numpy(fileName)
    
    #Get the raw PLY data into tensors
    pos3D = torch.from_numpy(positionsNP)
    colors = torch.from_numpy(colorsNP)
    colors.clamp(0,1)
    rots = torch.from_numpy(rotsNP)
    scales = torch.from_numpy(scalesNP)
    points = torch.from_numpy(positionsNP)

    samples = points.size()[0]
    y = points
    x = points
    
    #knn find midpoint
    assign_index = knn(x, y, neighbors)
    indexSrc = assign_index[0:1][0]
    indexSrcTrn = indexSrc.reshape(-1,1)
    index = indexSrcTrn.expand(indexSrc.size()[0],3)

    src = x[assign_index[1:2]][0]
    out = src.new_zeros(src.size())
    out = scatter_mean(src, index, 0, out=out)

    #calculate normals
    normals = y - out[0:samples]
    

    #use Euclidian distances to create a mask
    #distances = torch.sqrt((out[0:samples][:,0])**2 +(out[0:samples][:,1])**2+(out[0:samples][:,2])**2 )
    distances = torch.sqrt((y[:,0] - out[0:samples][:,0])**2 +(y[:,1] - out[0:samples][:,1])**2+(y[:,2] - out[0:samples][:,2])**2 )
    threshold = np.clip(threshold, 0, 100)
    boundry = np.percentile(distances, threshold)
    mask = distances < boundry

    
    pos3DFiltered = pos3D[mask]
    normalsFiltered = normals[mask]    
    colorsFiltered = colors[mask]     
    scalesFiltered = scales[mask]
    rotsFiltered = rots[mask]


    
    diffvector = y - out[0:samples]

    diffvectorNum =diffvector.numpy()
    diffNum = distances.numpy()
    resultNum =  out[0:samples].numpy()
    # Normalised [0,1]
    diffNumNorm = (diffNum - np.min(diffNum))/np.ptp(diffNum)

    '''
    #point cloud
    marker_data = go.Scatter3d(
        x=points[:, 0], 
        y=points[:, 2], 
        z=-points[:, 1], 
        marker=go.scatter3d.Marker(size=3, color= diffNumNorm), 
        opacity=0.8, 
        mode='markers'
    )
    fig=go.Figure(data=marker_data)
    fig.show()
    '''
    #normals
    fig = go.Figure(data=go.Cone(
        x=points[:, 0],
        y=points[:, 2],
        z=-points[:, 1],
        u=diffvectorNum[:, 0],
        v=diffvectorNum[:, 2],
        w=-diffvectorNum[:, 1],
        sizemode="absolute",
        sizeref=coneSize,
        anchor="tail"))

    fig.update_layout(
        scene=dict(domain_x=[0, 1],
                    camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

    fig.show()


    return pos3DFiltered, normalsFiltered, colorsFiltered, scalesFiltered, rotsFiltered