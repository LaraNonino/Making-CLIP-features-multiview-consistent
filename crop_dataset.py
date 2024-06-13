import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import json
from tqdm import tqdm

class Object:
    def __init__(self, seg_group, scene):
        self.id = seg_group['objectId']
        self.seg_group = seg_group
        self.scene = scene
        self.mesh, self.vertices = self.get_mesh_object()

    def get_mesh_object(self):
        instance_segments = self.seg_group['segments']
        instance_vertices_mask = np.isin(scene.all_vertices, instance_segments)
        instance_vertex_indices = np.where(instance_vertices_mask)[0]

        faces = np.asarray(scene.mesh.triangles)
        face_mask = np.all(np.isin(faces, instance_vertex_indices), axis=1)
        instance_faces = faces[face_mask] # Triplets of vertex indices forming each triangle (the indices refer to scene vertices - all of them)
        instance_vertices = np.asarray(scene.mesh.vertices)[instance_vertex_indices] # Coordinates of each vertex
        vertex_remap = {scene_idx: instance_idx for instance_idx, scene_idx in enumerate(instance_vertex_indices)}
        instance_faces = np.vectorize(vertex_remap.get)(instance_faces) # Triplets of vertex indices forming each triangle (the indices refer to instance vertices - masked)

        # Create the mesh for the selected instance
        instance_mesh = o3d.geometry.TriangleMesh()
        instance_mesh.vertices = o3d.utility.Vector3dVector(instance_vertices)
        instance_mesh.triangles = o3d.utility.Vector3iVector(instance_faces)

        self.mesh = instance_mesh
        self.vertices = instance_vertices

        return instance_mesh, instance_vertices
    
    def minimum_vertices(self, visible_object):
        return int(self.vertices.shape[0] * visible_object)

class Scene:
    def __init__(self, scans_directory, scan):
        self.name = scan
        self.path = os.path.join(scans_directory, self.name)

        scene_mesh_path = os.path.join(self.path, self.name + "_vh_clean_2.ply")
        scene_mesh = o3d.io.read_triangle_mesh(scene_mesh_path)
        segmentation_mesh_path = os.path.join(self.path, self.name + "_vh_clean_2.labels.ply")
        segmentation_mesh = o3d.io.read_triangle_mesh(segmentation_mesh_path)

        with open(os.path.join(self.path, self.name + '_vh_clean.aggregation.json')) as f:
            aggregation_data = json.load(f)

        with open(os.path.join(self.path, self.name + '_vh_clean_2.0.010000.segs.json')) as f:
            segmentation_data = json.load(f)
        
        self.mesh = scene_mesh
        self.segmentation_mesh = segmentation_mesh
        self.aggregation_data = aggregation_data
        self.segmentation_data = segmentation_data
        self.all_vertices = np.array(self.segmentation_data['segIndices'])
        self.num_objects = len(self.aggregation_data['segGroups'])
        self.objects = {}
        self.scan = Scan(self)

    def populate(self, to_exlude):
        for seg_group in self.aggregation_data['segGroups']:
            if seg_group['label'] in to_exlude:
                continue
            object = Object(seg_group, self)
            self.objects[seg_group['objectId']] = object
        return
    
class Scan:
    def __init__(self, scene):
        self.scene = scene

    def load_camera_parameters(self, reader_script, unzipped_directory):
        self.unzipped_directory = unzipped_directory
        sens_file = os.path.join(self.scene.path, self.scene.name + ".sens")
        output_directory = os.path.join(unzipped_directory, self.scene.name)
        os.makedirs(output_directory, exist_ok=True)

        # Export data if not already exported
        if not os.listdir(output_directory):
            os.system(f"python {reader_script} --filename {sens_file} --output_path {output_directory} --export_depth_images --export_color_images --export_poses --export_intrinsics --export_depth_images")

        pose_directory = os.path.join(output_directory, "pose")
        color_directory = os.path.join(output_directory, "color")
        # depth_directory = os.path.join(output_directory, "depth")
        # pose_files = [f for f in os.listdir(pose_directory) if f.endswith('.txt')]
        color_files = [f for f in os.listdir(color_directory) if f.endswith('.jpg')]
        frame_indices = [int(f.split('.')[0]) for f in color_files]
        intrinsics = np.loadtxt(os.path.join(output_directory, "intrinsic", "intrinsic_color.txt"))  # Camera intrinsics
        # extrinsics = np.loadtxt(os.path.join(output_directory, "intrinsic", "extrinsic_color.txt"))  # Camera estrinsic
        camera_intrinsics = intrinsics[:3, :3]

        self.camera_intrinsics = camera_intrinsics
        self.frame_indices = frame_indices
        self.pose_directory = pose_directory
        self.color_directory = color_directory

        return
    
    def get_original_frame(self, frame_index):
        original_path = os.path.join(self.color_directory, f"{frame_index}.jpg")

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

    def get_camera_extrinsics(self, frame_index):
        extrinsics_path = os.path.join(self.pose_directory, f"{frame_index}.txt")
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

def process_object(object, visible_object, camera_intrinsics, camera_extrinsics, depth, camera, original_frame):
    points_2d = get_xyz(np.asarray(object.mesh.vertices), camera_intrinsics, camera_extrinsics)
    valid_vertices, vertices_to_remove = get_occluded_vertices_idx(depth, points_2d, error)
    if valid_vertices - len(vertices_to_remove) < object.minimum_vertices(visible_object):
        return
    vertices_to_remove = set(vertices_to_remove)
    all_faces = np.asarray(object.mesh.triangles)
    faces_to_keep = [face for face in all_faces if not any(vertex in vertices_to_remove for vertex in face)]

    # Re-index faces: create a mapping from old to new vertex indices
    remaining_vertices_indices = list(set(range(len(object.mesh.vertices))) - vertices_to_remove)
    new_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_vertices_indices)}
    # Update face indices based on new vertex indices
    new_faces = [[new_index_map[vertex] for vertex in face] if all(vertex in new_index_map for vertex in face) else None for face in faces_to_keep]
    new_faces = [face for face in new_faces if face is not None]
    # Get new mesh
    new_vertices = np.asarray(object.mesh.vertices)[remaining_vertices_indices]
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)

    # Rendering
    image_height, image_width = original_frame.shape[:2]
    projected_mesh = render_image(new_mesh, camera, image_width, image_height)
    projected_mesh = np.asarray(projected_mesh)

    # Masking
    mask = (projected_mesh[:, :, 0] == 0)
    masked_frame = np.zeros_like(original_frame)   # masked_frame is a black image with same shape as original_frame
    masked_frame[mask] = original_frame[mask]      # wrong_colors
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)    # correct_colors

    # Cropping
    y_indices, x_indices = np.where(mask)
    if y_indices.size > 0 and x_indices.size > 0:
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)    # correct_colors
        cropped_frame = original_frame[y_min:y_max+1, x_min:x_max+1]
    else:
        cropped_frame = None

    return cropped_frame

def process_frame(scene, scene_vis, camera_intrinsics, frame_index, error, visible_object):
    scan = scene.scan
    objects = scene.objects

    original_frame = scan.get_original_frame(frame_index=frame_index)
    if original_frame is None: return
    image_height, image_width = original_frame.shape[:2]
    camera_extrinsics = scan.get_camera_extrinsics(frame_index)

    camera = get_camera_for_rendering(camera_extrinsics, camera_intrinsics, image_height, image_width)
    scene_vis.get_view_control().convert_from_pinhole_camera_parameters(camera, True)
    depth = np.asarray(scene_vis.capture_depth_float_buffer(do_render=True))

    cropped_frames = {}
    for object_id in objects.keys():
        cropped_frames[object_id] = process_object( object = objects[object_id], visible_object = visible_object, 
                                        camera_intrinsics = camera_intrinsics, camera_extrinsics = camera_extrinsics,
                                        depth = depth, camera = camera, original_frame=original_frame)
    
    return cropped_frames

def get_masked_images(scene, visible_object, error, cropped_images_directory):
    if not os.path.exists(cropped_images_directory):
        os.makedirs(cropped_images_directory)
    original_frame = scene.scan.get_original_frame(0)
    image_height, image_width = original_frame.shape[:2]
    scene_vis = o3d.visualization.Visualizer()
    scene_vis.create_window(visible=False, width = image_width, height = image_height)
    scene_vis.add_geometry(scene.mesh)
    
    processed_frames = 0
    frame_indices = scene.scan.frame_indices
    for frame_index in tqdm(sorted(frame_indices), total=len(frame_indices), desc="Processing Frames"):
        cropped_frames = process_frame(scene = scene, 
                                     scene_vis=scene_vis,  
                                     camera_intrinsics=scene.scan.camera_intrinsics,
                                     frame_index=frame_index, 
                                     error=error,
                                     visible_object=visible_object)
        
        if cropped_frames is not None:
            for object_id in cropped_frames.keys():
                if cropped_frames[object_id] is None:
                    continue
                output_directory = os.path.join(cropped_images_directory, str(object_id))
                os.makedirs(output_directory, exist_ok=True)
                output_path = os.path.join(output_directory, f"{frame_index:05d}.jpg")
                cv2.imwrite(output_path, cropped_frames[object_id])
        processed_frames += 1
    return


num_scans = 10
visible_object = 0.8 # percentage
error = 1.5 # Tolerance on occlusions in meters
# masked_scannet_directory = "/cluster/project/cvg/data/masked_scannet"
# masked_scannet_directory = "/cluster/scratch"
masked_scannet_directory = "/Users/lara/Desktop/Making-CLIP-features-multiview-consistent/outputs"
unzipped_directory =  "/Users/lara/Desktop/Making-CLIP-features-multiview-consistent/unzipped"
reader_script = "/Users/lara/Desktop/Making-CLIP-features-multiview-consistent/scannet_scripts/SensReader/reader.py"
# scans_directory = "/cluster/project/cvg/data/scannet/scans"
scans_directory = "/Users/lara/Desktop/Making-CLIP-features-multiview-consistent/data/scans"



scans = os.listdir(scans_directory)
scans.sort()
scans_loaded = 0
current_directory = os.getcwd()

objects_to_exlude = {"wall", "floor", "closet floor", "door wall", "window", "doorframe", "mirror", "curtain", "ceiling", }

for scan_name in scans:
    if scan_name == "scene0000_00" or scan_name == "scene0001_00" or scan_name == "scene0002_00" or scan_name == "scene0003_00" or scan_name == "scene0004_00" or scan_name == "scene0005_00" or scan_name == "scene0006_00" or scan_name == "scene0006_00" or scan_name == "scene0007_00" or scan_name == "scene0008_00" or scan_name == "scene0009_00":
        continue
    if scans_loaded == 20:
        break
    if scan_name.endswith("_00"):
        scene = Scene(scans_directory, scan_name)
        scene.populate(to_exlude=objects_to_exlude)
        scene.scan.load_camera_parameters(reader_script=reader_script, unzipped_directory=unzipped_directory)
        cropped_images_directory = os.path.join(masked_scannet_directory, "cropped_images/project_without_occlusion", scan_name)
        get_masked_images(scene = scene, visible_object = visible_object, error = error, cropped_images_directory=cropped_images_directory)
        scans_loaded += 1