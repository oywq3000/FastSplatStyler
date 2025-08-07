import torch
import torch.utils
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

    
def splat_save(positions, scales, rots, colors, output_path, fileType):

    splatio.numpy_to_splat(positions, scales, rots, colors, output_path, fileType)

    return 



def splat_randomsampler(pos3D):

    pointsNP = pos3D.numpy().copy()
    np.random.shuffle(pointsNP)
    scaler = int(pointsNP.shape[0]/2)
    valuesNP = pointsNP[:scaler]
    values64 = torch.from_numpy(valuesNP)
    values = values64.to(torch.float32)
    return values


def splat_GaussianSuperSampler(pos3D_fr, colors_fr, opacity_fr, scales_fr, rots_fr, GaussianSamples):

    pos3D = pos3D_fr.clone()
    colors = colors_fr.clone()
    opacity = opacity_fr.clone()
    scales = scales_fr.clone()
    rots = rots_fr.clone()

    #get all the splats' relative sizes/densities and opacities
    importance = (torch.absolute(scales[:,0])+torch.absolute(scales[:,1])+torch.absolute(scales[:,2]))/scales.max()+torch.absolute(opacity)/opacity.max()
    importanceConstant = importance.sum().to(torch.int32)
    
    increaseNormalizer = GaussianSamples/importanceConstant.item()
    copiesToMakeQuantity = (importance*increaseNormalizer).to(torch.int32)
    #create duplicates of all the larger Gaussians, create matched tensors for all the other attributes too
    duplicatePoints = torch.from_numpy(np.repeat(pos3D.numpy(), copiesToMakeQuantity.numpy(), axis=0))
    duplicateColors = torch.from_numpy(np.repeat(colors.numpy(), copiesToMakeQuantity.numpy(), axis=0))
    duplicateRots = torch.from_numpy(np.repeat(rots.numpy(), copiesToMakeQuantity.numpy(), axis=0))
    duplicateScales = torch.from_numpy(np.repeat(scales.numpy(), copiesToMakeQuantity.numpy(), axis=0))

    #create a copy for saving the rotation results
    duplicateScalesRotated = duplicateScales.clone()
    
    q0 = duplicateRots[:,0]
    q1 = duplicateRots[:,1]
    q2 = duplicateRots[:,2]
    q3 = duplicateRots[:,3]
    x00 = q0**2 + q1**2 - q2**2 - q3**2
    x01 = 2*q1*q2 - 2*q0*q3
    x02 = 2*q1*q3 + 2*q0*q2
    x10 = 2*q1*q2 + 2*q0*q3
    x11 = q0**2 - q1**2 + q2**2 - q3**2
    x12 = 2*q2*q3 - 2*q0*q1
    x20 = 2*q1*q3 - 2*q0*q2
    x21 = 2*q2*q3 + 2*q0*q1
    x22 = q0**2 - q1**2 - q2**2 + q3**2

    X0 = torch.stack((x00, x01, x02), dim=1)
    X1 = torch.stack((x10, x11, x12), dim=1)
    X2 = torch.stack((x20, x21, x22), dim=1)

    
    R_matrix = torch.stack((X0, X1, X2), dim=1)
    duplicateScalesRotated = torch.bmm(R_matrix, duplicateScales.unsqueeze(2)).squeeze(2)
    #torch.normal can't handle negative numbers, store the signs and add them back afterwards
    duplicateScalesRotatedSigns = duplicateScalesRotated.sign()
    duplicateScalesRotatedABS = torch.absolute(duplicateScalesRotated)

    GaussianPointsNoSignScale = torch.normal(duplicatePoints, duplicateScalesRotatedABS)
    #correct for signs
    GaussianPoints = duplicatePoints + (GaussianPointsNoSignScale-duplicatePoints)*duplicateScalesRotatedSigns

    pos3D = torch.cat((pos3D_fr, GaussianPoints
                        ), dim=0)
    colors = torch.cat((colors_fr,duplicateColors
                        ), dim=0)
    return pos3D, colors.to(torch.float32)
    



def splat_unpacker_with_threshold(neighbors, fileName, threshold):

    #check if .ply or .splat or gen
    if fileName[-3:] == 'ply':
        positionsFile, scalesFile, rotsFile, colorsFile = splatio.ply_to_numpy(fileName)
        fileType = 'ply'
    elif fileName[-3:] == 'gen':
        resolution = 100*5
        radius = 1
        indices = np.arange(0, resolution, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/resolution)
        theta = np.pi * (1 + 5**0.5) * indices

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        pos3DFiltered = torch.from_numpy(np.transpose(np.stack([x,y,z]).astype(np.float32)))
        normalsFiltered = torch.full_like(pos3DFiltered, 0.0)
        opacity = torch.full_like(pos3DFiltered, 0.5)
        #colorsFiltered = torch.full_like(pos3DFiltered, 0.5)
        scalesFiltered = torch.full_like(pos3DFiltered, 0.015)
        rotsFiltered = torch.full((resolution, 4), 0.0)
        rotsFiltered[:,0] = 1.0
        fileType = 'splat'

        #make is stripy
        c1 = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32)
        c2 = torch.tensor([0.9, 0.0, 0.0], dtype=torch.float32)
        c3 = torch.tensor([0.0, 0.9, 0.0], dtype=torch.float32)
        c4 = torch.tensor([0.0, 0.0, 0.9], dtype=torch.float32)
        c5 = torch.tensor([0.9, 0.9, 0.0], dtype=torch.float32)

        C1 = torch.from_numpy(np.tile(c1.numpy(), (int(resolution/5),1)))
        C2 = torch.from_numpy(np.tile(c2.numpy(), (int(resolution/5),1)))
        C3 = torch.from_numpy(np.tile(c3.numpy(), (int(resolution/5),1)))
        C4 = torch.from_numpy(np.tile(c4.numpy(), (int(resolution/5),1)))
        C5 = torch.from_numpy(np.tile(c5.numpy(), (int(resolution/5),1)))

        colorsFiltered = torch.cat((C1, C2, C3, C4, C5), dim=0)

        addnoise = True
        if (addnoise):
            noisePoints = torch.from_numpy(np.random.uniform(-1.5, 1.5, (50,3)).astype(np.float32))
            pos3DFiltered = torch.cat((noisePoints, pos3DFiltered
                                ), dim=0)
            
            noiseColors = torch.from_numpy(np.random.uniform(0, 9.99, (50,3)).astype(np.float32))
            colorsFiltered = torch.cat((noiseColors, colorsFiltered
                                ), dim=0)
            
            noiseNormals = torch.full_like(noisePoints, 0.0)
            normalsFiltered = torch.cat((noiseNormals, normalsFiltered
                                ), dim=0)
            
            noiseScales = torch.full_like(noisePoints, 0.015)
            scalesFiltered = torch.cat((noiseScales, scalesFiltered
                                ), dim=0)
            
            noiseopacity = torch.full_like(noisePoints, 0.5)
            opacity = torch.cat((noiseopacity, opacity
                                ), dim=0)
            
            noiseScales = torch.full((50, 4), 0.0)
            rotsFiltered = torch.cat((noiseScales, rotsFiltered
                                ), dim=0)
            rotsFiltered[:,0] = 1.0
            

        return  pos3DFiltered, normalsFiltered, colorsFiltered, opacity[:,0], scalesFiltered, rotsFiltered, fileType
    elif fileName[-4:] == 'gen2':
        
        
        z = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9], dtype=torch.float32)
        y = torch.tensor([1.7, 1.4, 1.0, 0.5, 0.0], dtype=torch.float32)
        x = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float32)


        pos3DFiltered = torch.from_numpy(np.transpose(np.stack([x,y,z]).astype(np.float32)))
        normalsFiltered = torch.full_like(pos3DFiltered, 0.0)
        opacity = torch.full_like(pos3DFiltered, 0.99)
        #colorsFiltered = torch.full_like(pos3DFiltered, 0.5)
        #scalesFiltered = torch.full_like(pos3DFiltered, 0.02)
        rotsFiltered = torch.full((5, 4), 0.0)
        rotsFiltered[:,0] = 1.0
        fileType = 'splat'

        scalesFiltered = torch.tensor([[0.03, 0.03, 0.01],
                                       [0.03, 0.03, 0.05],
                                       [0.07, 0.07, 0.03],
                                       [0.05, 0.05, 0.09],
                                       [0.09, 0.12, 0.12,]
                                       ], dtype= torch.float32)

        
        #each Gaussian has its own color
        c1 = torch.tensor([0.1, 0.2, 0.5, 0.7, 0.9], dtype=torch.float32)
        c2 =  torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
        c3 =  torch.tensor([0.9, 0.7, 0.5, 0.2, 0.1], dtype=torch.float32)


        colorsFiltered =  torch.from_numpy(np.transpose(np.stack([c1,c2,c3]).astype(np.float32)))



        return  pos3DFiltered, normalsFiltered, colorsFiltered, opacity[:,0], scalesFiltered, rotsFiltered, fileType
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
        
    if threshold == 100:
    
        pos3DFiltered = pos3D
        normalsFiltered = normals  
        colorsFiltered = colors   
        scalesFiltered = scales
        rotsFiltered = rots

    else:

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

        
        displayNormals = False
        if (displayNormals):
                
            coneSize = 0.1
            diffvectorNum =normals.numpy()
            
            showBoth = False
            if (showBoth):    


                diffBoth = torch.cat((normals, normalsFiltered), dim=0).numpy().astype(np.float16)
                pos3dShifted = pos3DFiltered
                pos3dShifted[:,2] = pos3dShifted[:,2] + 1
                posBoth = torch.cat((pos3D, pos3dShifted), dim=0).numpy().astype(np.float16)

                distancesMasked = distances[mask]
                diffBothNum =  torch.cat((distances, distancesMasked), dim=0).numpy().astype(np.float16)
                
                # Normalised [0,1]
                diffNumNorm = (diffBothNum - np.min(diffBothNum))/np.ptp(diffBothNum)
            
                #point cloud
                marker_data = go.Scatter3d(
                    x=posBoth[:, 0], 
                    y=posBoth[:, 2], 
                    z=-posBoth[:, 1], 
                    marker=go.scatter3d.Marker(size=1, color= diffNumNorm), 
                    opacity=0.8, 
                    mode='markers'
                )
                fig=go.Figure(data=marker_data)
                fig.show()



                '''
                #Too high memory usage :(
                diffBoth = torch.cat((normals, normalsFiltered), dim=0).numpy().astype(np.float16)
                pos3dShifted = pos3DFiltered
                pos3dShifted[:,2] = pos3dShifted[:,2] + 1
                posBoth = torch.cat((pos3D, pos3dShifted), dim=0).numpy().astype(np.float16)


                #normals
                fig = go.Figure(data=go.Cone(
                    x=posBoth[:, 0],
                    y=posBoth[:, 2],
                    z=-posBoth[:, 1],
                    u=diffBoth[:, 0],
                    v=diffBoth[:, 2],
                    w=-diffBoth[:, 1],
                    sizemode="raw",
                    sizeref=coneSize,
                    anchor="tail"))

                fig.update_layout(
                    scene=dict(domain_x=[0, 1],
                                camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

                fig.show()

                exit()
                '''
                

            else:
                            
                diffNum = distances.numpy()
                # Normalised [0,1]
                diffNumNorm = (diffNum - np.min(diffNum))/np.ptp(diffNum)
            
                #point cloud
                marker_data = go.Scatter3d(
                    x=pos3DFiltered[:, 0], 
                    y=pos3DFiltered[:, 2], 
                    z=-pos3DFiltered[:, 1], 
                    marker=go.scatter3d.Marker(size=1, color= diffNumNorm), 
                    opacity=0.8, 
                    mode='markers'
                )
                fig=go.Figure(data=marker_data)
                fig.show()


                    
                #normals
                fig = go.Figure(data=go.Cone(
                    x=pos3D[:, 0],
                    y=pos3D[:, 2],
                    z=-pos3D[:, 1],
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


    return pos3DFiltered, normalsFiltered, colorsFiltered[:,0:3], colorsFiltered[:,3], scalesFiltered, rotsFiltered, fileType



def generate_with_noise_ablation(neighbors, fileName, threshold):

    resolution = 100*5
    radius = 1
    indices = np.arange(0, resolution, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/resolution)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    pos3D = torch.from_numpy(np.transpose(np.stack([x,y,z]).astype(np.float32)))
    normalsFiltered = torch.full_like(pos3D, 0.0)
    opacity = torch.full_like(pos3D, 0.5)
    #colorsFiltered = torch.full_like(pos3DFiltered, 0.5)
    scales = torch.full_like(pos3D, 0.015)
    rots = torch.full((resolution, 4), 0.0)
    rots[:,0] = 1.0
    fileType = 'splat'

    #make is stripy
    c1 = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float32)
    c2 = torch.tensor([0.9, 0.0, 0.0], dtype=torch.float32)
    c3 = torch.tensor([0.0, 0.9, 0.0], dtype=torch.float32)
    c4 = torch.tensor([0.0, 0.0, 0.9], dtype=torch.float32)
    c5 = torch.tensor([0.9, 0.9, 0.0], dtype=torch.float32)

    C1 = torch.from_numpy(np.tile(c1.numpy(), (int(resolution/5),1)))
    C2 = torch.from_numpy(np.tile(c2.numpy(), (int(resolution/5),1)))
    C3 = torch.from_numpy(np.tile(c3.numpy(), (int(resolution/5),1)))
    C4 = torch.from_numpy(np.tile(c4.numpy(), (int(resolution/5),1)))
    C5 = torch.from_numpy(np.tile(c5.numpy(), (int(resolution/5),1)))

    colors = torch.cat((C1, C2, C3, C4, C5), dim=0)

    addnoise = True
    if (addnoise):
        noisePoints = torch.from_numpy(np.random.uniform(-1.01, 1.01, (50,3)).astype(np.float32))
        pos3D = torch.cat((noisePoints, pos3D
                            ), dim=0)
        
        noiseColors = torch.from_numpy(np.random.uniform(0, 9.99, (50,3)).astype(np.float32))
        colors = torch.cat((noiseColors, colors
                            ), dim=0)
                
        noiseScales = torch.full_like(noisePoints, 0.015)
        scales = torch.cat((noiseScales, scales
                            ), dim=0)
        
        noiseopacity = torch.full_like(noisePoints, 0.5)
        opacity = torch.cat((noiseopacity, opacity
                            ), dim=0)
        
        noiseRots = torch.full((50, 4), 0.0)
        rots = torch.cat((noiseRots, rots
                            ), dim=0)
        rots[:,0] = 1.0
        
    points = pos3D

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
    
    if threshold == 100:
    
        pos3DFiltered = pos3D
        normalsFiltered = normals  
        colorsFiltered = colors   
        scalesFiltered = scales
        rotsFiltered = rots
        opacity = opacity

    else:

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
        opacity = opacity[mask]

        displayNormals = True
        if (displayNormals):
                
            coneSize = 1
            diffvectorNum = normals.numpy()
            diffvectorfilteredNum = normalsFiltered.numpy()
            diffNum = distances.numpy()
            
            # Normalised [0,1]
            diffNumNorm = (diffNum - np.min(diffNum))/np.ptp(diffNum)
        
            #point cloud
            marker_data = go.Scatter3d(
                x=pos3D[:, 0], 
                y=pos3D[:, 2], 
                z=-pos3D[:, 1], 
                marker=go.scatter3d.Marker(size=5, color= diffNumNorm), 
                opacity=0.8, 
                mode='markers'
            )
            fig=go.Figure(data=marker_data)
            fig.show()

            #normals
            fig2 = go.Figure(data=go.Cone(
                x=pos3D[:, 0],
                y=pos3D[:, 2],
                z=-pos3D[:, 1],
                u=diffvectorNum[:, 0],
                v=diffvectorNum[:, 2],
                w=-diffvectorNum[:, 1],
                sizemode="raw",
                sizeref=coneSize,
                anchor="tail"))

            fig2.update_layout(
                scene=dict(domain_x=[0, 1],
                            camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

            fig2.show()

            
            #normals Filtered
            fig3 = go.Figure(data=go.Cone(
                x=pos3DFiltered[:, 0],
                y=pos3DFiltered[:, 2],
                z=-pos3DFiltered[:, 1],
                u=diffvectorfilteredNum[:, 0],
                v=diffvectorfilteredNum[:, 2],
                w=-diffvectorfilteredNum[:, 1],
                sizemode="raw",
                sizeref=coneSize,
                anchor="tail"))

            fig3.update_layout(
                scene=dict(domain_x=[0, 1],
                            camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

            fig3.show()


    return  pos3DFiltered, normalsFiltered, colorsFiltered, opacity[:,0], scalesFiltered, rotsFiltered, fileType




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