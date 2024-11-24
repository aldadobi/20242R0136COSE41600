# 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# PCD 파일 불러오기
file_path = "test_data/1727320101-665925967.pcd"
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling
voxel_size = 0.2
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR)
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용한 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = ror_pcd.select_by_index(inliers, invert=True)

# DBSCAN 클러스터링
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=8, print_progress=True))

# 포인트 클라우드 색상 지정
colors = np.zeros((len(labels), 3))
colors[labels >= 0] = [0, 0, 1]  # 파란색
final_point.colors = o3d.utility.Vector3dVector(colors)

# 필터링 기준 설정
min_points_in_cluster = 5
max_points_in_cluster = 40
min_height = 0.25
max_height = 2.0
max_distance = 30.0
min_aspect_ratio = 1.0
max_aspect_ratio = 3.0
road_tolerance = 0.7

# Bounding Box의 하단 중심점과 도로 평면 간 거리 필터링 함수
def filter_bboxes_by_road_proximity(bboxes, road_plane, road_tolerance):
    filtered_bboxes = []
    a, b, c, d = road_plane
    for bbox in bboxes:
        min_bound = bbox.get_min_bound()
        base_center = np.array([min_bound[0] + bbox.get_extent()[0] / 2, 
                                 min_bound[1] + bbox.get_extent()[1] / 2, 
                                 min_bound[2]])
        distance_to_road = np.abs(a * base_center[0] + b * base_center[1] + c * base_center[2] + d) / np.sqrt(a**2 + b**2 + c**2)
        if distance_to_road < road_tolerance:
            filtered_bboxes.append(bbox)
    return filtered_bboxes

# Bounding Box 생성
bboxes = []
#road_z_value = -plane_model[3] / plane_model[2]

for i in range(labels.max() + 1):
    cluster_indices = np.where(labels == i)[0]
    if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
        cluster_pcd = final_point.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        z_values = points[:, 2]
        z_min = z_values.min()
        z_max = z_values.max()
#        if road_z_value - road_tolerance <= z_min and z_max <= road_z_value + road_tolerance:
        height_diff = z_max - z_min
        if min_height <= height_diff <= max_height:
            distances = np.linalg.norm(points, axis=1)
            if distances.max() <= max_distance:
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                bbox.color = (1, 0, 0)
                bboxes.append(bbox)

# 도로 평면과의 거리로 Bounding Box 필터링
bboxes_proximity_filtered = filter_bboxes_by_road_proximity(bboxes, plane_model, road_tolerance)

# 형상 분석 필터링
bboxes_shape_filtered = []
for bbox in bboxes_proximity_filtered:
    extent = bbox.get_extent()
    height = extent[2]
    max_width = max(extent[0], extent[1])
    aspect_ratio = height / max_width
    if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
        bboxes_shape_filtered.append(bbox)

# Bounding Box 및 포인트 클라우드 시각화 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 최종 시각화
visualize_with_bounding_boxes(final_point, bboxes_shape_filtered, point_size=2.0)

# 평면 시각화 함수
def visualize_plane(plane_model, point_cloud, plane_size=10.0):
    a, b, c, d = plane_model
    x = np.linspace(-plane_size, plane_size, 10)
    y = np.linspace(-plane_size, plane_size, 10)
    X, Y = np.meshgrid(x, y)
    Z = (-d - a * X - b * Y) / c
    plane_points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    plane_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_points))
    plane_pcd.paint_uniform_color([1, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RANSAC Plane Visualization")
    vis.add_geometry(point_cloud)
    vis.add_geometry(plane_pcd)
    vis.run()
    vis.destroy_window()

# 평면 시각화
#visualize_plane(plane_model, ror_pcd)
