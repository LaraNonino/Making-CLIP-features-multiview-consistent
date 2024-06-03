import os

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

import json

def load_scannet_scene(scan_path, scan):
    scene_mesh_path = os.path.join(scan_path, scan + "_vh_clean_2.ply")
    scene_mesh = o3d.io.read_triangle_mesh(scene_mesh_path)
    segmentation_mesh_path = os.path.join(scan_path, scan + "_vh_clean_2.labels.ply")
    segmentation_mesh = o3d.io.read_triangle_mesh(segmentation_mesh_path)

    with open(os.path.join(scan_path, scan + '_vh_clean.aggregation.json')) as f:
        aggregation_data = json.load(f)
    
    return scene_mesh, segmentation_mesh, aggregation_data

def get_mesh_object(aggregation_data, object_id, scan_path, scan, scene_mesh):
    for seg_group in aggregation_data['segGroups']:
        if seg_group['objectId'] == object_id:
            selected_instance_segments = seg_group['segments']

    # Load all vertices
    with open(os.path.join(scan_path, scan + '_vh_clean_2.0.010000.segs.json')) as f:
        segmentation_data = json.load(f)

    # Load instance vertices
        all_vertices = np.array(segmentation_data['segIndices'])

    # Get indices of instance vertices
    instance_vertices_mask = np.isin(all_vertices, selected_instance_segments)
    instance_vertex_indices = np.where(instance_vertices_mask)[0]

    # Filter faces of the mesh: included if all its vertices are part of the instance
    faces = np.asarray(scene_mesh.triangles)
    face_mask = np.all(np.isin(faces, instance_vertex_indices), axis=1)
    instance_faces = faces[face_mask] # Triplets of vertex indices forming each triangle (the indices refer to scene vertices - all of them)
    instance_vertices = np.asarray(scene_mesh.vertices)[instance_vertex_indices] # Coordinates of each vertex
    vertex_remap = {scene_idx: instance_idx for instance_idx, scene_idx in enumerate(instance_vertex_indices)}
    instance_faces = np.vectorize(vertex_remap.get)(instance_faces) # Triplets of vertex indices forming each triangle (the indices refer to instance vertices - masked)

    # Create the mesh for the selected instance
    instance_mesh = o3d.geometry.TriangleMesh()
    instance_mesh.vertices = o3d.utility.Vector3dVector(instance_vertices)
    instance_mesh.triangles = o3d.utility.Vector3iVector(instance_faces)

    return instance_mesh, instance_vertices

def load_camera_parameters(scan_path, scan, masked_scannet_directory):
    reader_directory = "/cluster/home/lnonino/Making-CLIP-features-multiview-consistent/scripts/SensReader"
    sens_file = os.path.join(scan_path, scan + ".sens")
    output_directory = os.path.join(masked_scannet_directory, "reader", scan)
    os.makedirs(output_directory, exist_ok=True)
    
    # Export data if not already exported
    if not os.listdir(output_directory):
        os.system(f"python {os.path.join(reader_directory, 'reader.py')} --filename {sens_file} --output_path {output_directory} --export_depth_images --export_color_images --export_poses --export_intrinsics --export_depth_images")

    pose_directory = os.path.join(output_directory, "pose")
    color_directory = os.path.join(output_directory, "color")
    depth_directory = os.path.join(output_directory, "depth")
    pose_files = [f for f in os.listdir(pose_directory) if f.endswith('.txt')]
    color_files = [f for f in os.listdir(color_directory) if f.endswith('.jpg')]
    frame_indices = [int(f.split('.')[0]) for f in color_files]
    intrinsics = np.loadtxt(os.path.join(output_directory, "intrinsic", "intrinsic_color.txt"))  # Camera intrinsics
    # extrinsics = np.loadtxt(os.path.join(output_directory, "intrinsic", "extrinsic_color.txt"))  # Camera estrinsic
    camera_intrinsics = intrinsics[:3, :3]

    return camera_intrinsics, frame_indices

def get_minimum_vertices(total_vertices, visible_object):
    minimum_vertices = int(visible_object * total_vertices)
    return minimum_vertices

def get_original_frame(reader_directory, frame_index):
    original_path = os.path.join(reader_directory, "color", f"{frame_index}.jpg")

    if not os.path.exists(original_path):
        print(f"Error: No file found at {original_path}")
    else:
        image = cv2.imread(original_path)
        if image is None:
            print(f"Error: Failed to read the image from {original_path}")
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
    return

def get_camera_extrinsics(frame_index, reader_directory):
    extrinsics_path = os.path.join(reader_directory, "pose", f"{frame_index}.txt")
    return  np.loadtxt(extrinsics_path)

def get_2d_homogeneous(points_3d, intrinsic, extrinsic):
    inv_extrinsic = np.linalg.inv(extrinsic)
    points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    points_camera = np.dot(inv_extrinsic, points_3d_homogeneous.T).T
    points_camera_xyz = points_camera[:, :3] # Extract only the x, y, z components because the intrinsic matrix is 3x3
    points_2d_homogeneous = np.dot(intrinsic, points_camera_xyz.T).T # the third value is the depth
    return points_2d_homogeneous

# Get projected points and z coordinate
def get_xyz(vertices, intrinsics, extrinsics):
    points_2d = get_2d_homogeneous(vertices, intrinsics, extrinsics)
    points_2d_xy = (points_2d[:, :2] / points_2d[:, 2].reshape(-1, 1)).astype(int)
    points_2d[:, :2] = points_2d_xy
    return points_2d

def get_camera_for_rendering(camera_extrinsics, camera_intrinsics, h, w):
    camera = o3d.camera.PinholeCameraParameters()
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=w, height=h, fx=camera_intrinsics[0][0], fy=camera_intrinsics[1][1], cx=camera_intrinsics[0][2], cy=camera_intrinsics[1][2])
    camera.intrinsic = intrinsic
    camera.extrinsic = np.array(np.linalg.inv(camera_extrinsics))
    return camera

def get_occluded_vertices_idx(depth, points, error):
    total_vertices = points.shape[0]
    image_width, image_height = depth.shape
    occluded_idx = []
    valid_vertices = 0
    for vertex in range(total_vertices):
        point = points[vertex]
        # Check if the vertex falls into the image
        if (point[0] >= 0)& (point[0] < image_width) & (point[1] >= 0) & (point[1] < image_height):
            # Check if the vertex is in front of the camera
            if point[2] > 0:
                valid_vertices += 1
                # Check if the vertex is occluded
                if depth[int(point[0]), int(point[1])] + error < point[2]:
                    occluded_idx.append(vertex)
    return valid_vertices, occluded_idx

def render_image(mesh, camera, w, h):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width = w, height = h)
    mesh.paint_uniform_color([0, 0, 0])  # Black mesh
    vis.add_geometry(mesh)
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera, True)
    vis.get_render_option().background_color = np.array([1, 1, 1])  # White background
    vis.get_render_option().light_on = False
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=False)
    vis.destroy_window()
    return image

def process_frame(reader_directory, frame_index, vis, camera_intrinsics, instance_mesh, error):
    original_frame = get_original_frame(reader_directory=reader_directory, frame_index=frame_index)
    if original_frame is not None:
        pass
    else:
        return
    image_height, image_width = original_frame.shape[:2]
    camera_extrinsics = get_camera_extrinsics(frame_index, reader_directory)
    points_2d = get_xyz(np.asarray(instance_mesh.vertices), camera_intrinsics, camera_extrinsics)
    camera = get_camera_for_rendering(camera_extrinsics, camera_intrinsics, image_height, image_width)
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera, True)
    depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))

    valid_vertices, vertices_to_remove = get_occluded_vertices_idx(depth, points_2d, error)
    
    if valid_vertices - len(vertices_to_remove) < minimum_vertices:
        return
    
    vertices_to_remove = set(vertices_to_remove)
    all_faces = np.asarray(instance_mesh.triangles)
    faces_to_keep = [face for face in all_faces if not any(vertex in vertices_to_remove for vertex in face)]

    # Re-index faces: create a mapping from old to new vertex indices
    remaining_vertices_indices = list(set(range(len(instance_mesh.vertices))) - vertices_to_remove)
    new_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_vertices_indices)}
    # Update face indices based on new vertex indices
    new_faces = [[new_index_map[vertex] for vertex in face] if all(vertex in new_index_map for vertex in face) else None for face in faces_to_keep]
    new_faces = [face for face in new_faces if face is not None]
    # Get new mesh
    new_vertices = np.asarray(instance_mesh.vertices)[remaining_vertices_indices]
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)

    # Rendering
    camera = get_camera_for_rendering(camera_extrinsics, camera_intrinsics, image_height, image_width)
    image = render_image(new_mesh, camera, image_width, image_height)
    image = np.asarray(image)

    # Masking
    mask = (image[:, :, 0] == 0)
    masked_frame_wrong_colors = np.zeros_like(original_frame)
    masked_frame_wrong_colors[mask] = original_frame[mask]
    masked_frame = cv2.cvtColor(masked_frame_wrong_colors, cv2.COLOR_RGB2BGR)

    return masked_frame

def get_masked_images(reader_directory, masked_object_directory, scene_mesh, frame_indices, camera_intrinsics, instance_mesh, error):
    if not os.path.exists(masked_object_directory):
        os.makedirs(masked_object_directory)
    original_frame = get_original_frame(reader_directory, 0)
    image_height, image_width = original_frame.shape[:2]
    scene_vis = o3d.visualization.Visualizer()
    scene_vis.create_window(visible=False, width = image_width, height = image_height)
    scene_vis.add_geometry(scene_mesh)

    for frame_index in sorted(frame_indices):
        masked_frame = process_frame(reader_directory=reader_directory, frame_index=frame_index, vis=scene_vis, camera_intrinsics=camera_intrinsics, instance_mesh=instance_mesh, error=error)
        if masked_frame is not None:
            output_path = os.path.join(masked_object_directory, f"{frame_index:05d}.jpg")
            cv2.imwrite(output_path, masked_frame)
    return
    

num_scans = 10
visible_object = 0.8 # percentage
error = 0.2 # Tolerance on occlusions in meters
# masked_scannet_directory = "/cluster/project/cvg/data/masked_scannet"
masked_scannet_directory = "/cluster/scratch"
scans_directory = "/cluster/project/cvg/data/scannet/scans"


scans = os.listdir(scans_directory)
scans_loaded = 0
current_directory = os.getcwd()

for scan in scans:
    if scans_loaded == 10:
        break
    if scan.endswith("_00"):
        scans_loaded += 1
        scan_path = os.path.join(scans_directory, scan)
        scene_mesh, segmentation_mesh, aggregation_data = load_scannet_scene(scan_path, scan)
        for seg_group in aggregation_data['segGroups']:
            object_id = seg_group['objectId']
            instance_mesh, instance_vertices = get_mesh_object(aggregation_data, object_id, scan_path, scan, scene_mesh)
            camera_intrinsics, frame_indices = load_camera_parameters(scan_path, scan, masked_scannet_directory)
            minimum_vertices = get_minimum_vertices(total_vertices=instance_vertices.shape[0], visible_object = visible_object)
            get_masked_images(reader_directory = os.path.join(masked_scannet_directory, "reader", scan), 
                              masked_object_directory=os.path.join(masked_scannet_directory, "masked_images/project_without_occlusion", scan, f"{object_id}"),
                              scene_mesh=scene_mesh,
                              instance_mesh = instance_mesh,
                              frame_indices=frame_indices,
                              camera_intrinsics=camera_intrinsics,
                              error=error)