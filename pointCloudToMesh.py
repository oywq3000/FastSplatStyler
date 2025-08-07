import numpy as np
import open3d as o3d
from skimage import measure
from scipy.spatial import cKDTree
import splat_helpers as splt

# based on code from https://towardsdatascience.com/transform-point-clouds-into-3d-meshes-a-python-guide-8b0407a780e6
# credit Florent Poux
# Towards Data Science (2024)

def MarchingCubes_from_ply(dataset, voxel_size, iso_level_percentile):

    pcd = o3d.io.read_point_cloud(dataset)

    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points)
    # Compute the bounds of the point cloud
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    # Create a 3D grid
    x = np.arange(mins[0], maxs[0], voxel_size)
    y = np.arange(mins[1], maxs[1], voxel_size)
    z = np.arange(mins[2], maxs[2], voxel_size)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Compute the scalar field (distance to nearest point)
    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    distances, _ = tree.query(grid_points)
    scalar_field = distances.reshape(x.shape)

    # Determine iso-level based on percentile of distances
    iso_level = np.percentile(distances, iso_level_percentile)

    # Apply Marching Cubes
    verts, faces, _, _ = measure.marching_cubes(scalar_field, level=iso_level)

    # Scale and translate vertices back to original coordinate system
    verts = verts * voxel_size + mins
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # Compute vertex normals
    mesh.compute_vertex_normals()
    # Visualize the result
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)



def MarchingCubes_with_filtering(dataset, voxel_size, iso_level_percentile, threshold=99, out_file = 'out.obj'):

    pos3D, _, _, _, _, _, _ = splt.splat_unpacker_threshold(25, dataset, threshold)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos3D.numpy())

    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points)
    # Compute the bounds of the point cloud
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    # Create a 3D grid
    x = np.arange(mins[0], maxs[0], voxel_size)
    y = np.arange(mins[1], maxs[1], voxel_size)
    z = np.arange(mins[2], maxs[2], voxel_size)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Compute the scalar field (distance to nearest point)
    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    distances, _ = tree.query(grid_points)
    scalar_field = distances.reshape(x.shape)

    # Determine iso-level based on percentile of distances
    iso_level = np.percentile(distances, iso_level_percentile)

    # Apply Marching Cubes
    verts, faces, _, _ = measure.marching_cubes(scalar_field, level=iso_level)

    # Scale and translate vertices back to original coordinate system
    verts = verts * voxel_size + mins
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # Compute vertex normals
    mesh.compute_vertex_normals()
    # Save the result
    o3d.io.write_triangle_mesh(out_file, mesh)
    # Visualize the result
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd])
    

    
def MarchingCubes_return_vertices(dataset, visualize = False):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dataset.numpy())
    voxel_size_tensor = (abs(dataset.min()) + abs(dataset.max()))/100
    voxel_size = voxel_size_tensor.item()
    iso_level_percentile = 5

    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points)
    # Compute the bounds of the point cloud
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    # Create a 3D grid
    x = np.arange(mins[0], maxs[0], voxel_size)
    y = np.arange(mins[1], maxs[1], voxel_size)
    z = np.arange(mins[2], maxs[2], voxel_size)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Compute the scalar field (distance to nearest point)
    grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    distances, _ = tree.query(grid_points)
    scalar_field = distances.reshape(x.shape)

    # Determine iso-level based on percentile of distances
    iso_level = np.percentile(distances, iso_level_percentile)

    # Apply Marching Cubes
    verts, faces, _, _ = measure.marching_cubes(scalar_field, level=iso_level)

    # Scale and translate vertices back to original coordinate system
    verts = verts * voxel_size + mins
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # Compute vertex normals
    mesh.compute_vertex_normals()
    # Save the result
    #o3d.io.write_triangle_mesh(out_file, mesh)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd2.estimate_normals()
    if visualize == True:
        # Visualize the result
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        o3d.visualization.draw_geometries([pcd2])

    vertices = np.asarray(pcd2.points, dtype= np.float32)
    return vertices
    
def graph_Points(points, colors):
    import matplotlib.pyplot as plt

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Scatter plot with colors
    ax.scatter(x, y, z, c=colors)

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()

    return 

def Estimate_Normals(points, n = 25, threshold=75):

    #pos3D, _, _, _, _ = splt.splat_unpacker_threshold(n, dataset, threshold)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())

    pcd.estimate_normals()
    pcd.normalize_normals()
    normals = np.asarray(pcd.normals)

    #scaleSize = normals.max()
    #normalsBad = np.random.rand(normals.shape[0],normals.shape[1])
    #normalsBad = normalsBad * scaleSize
    #return normalsBad

    return normals


    