import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# PCD 파일 불러오기
#file_path = "test_data/1727320101-665925967.pcd"  # 데이터 파일 경로
file_path = "test_data/1727320101-665925967.pcd"
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.2
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# 다중 평면 추정 함수
def multi_plane_segmentation(pcd, distance_threshold=0.1, num_iterations=2000, max_planes=5):
    remaining_pcd = pcd
    planes = []
    inliers_list = []
    for _ in range(max_planes):
        if len(remaining_pcd.points) < 10:
            break
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=num_iterations
        )
        planes.append(plane_model)
        inliers_list.append(inliers)
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
    return planes, inliers_list

# 클러스터와 평면 간의 관계를 분석하여 필터링
def filter_clusters_by_plane_relation(clusters, planes, tolerance=1.0):
    filtered_clusters = []
    for cluster in clusters:
        points = np.asarray(cluster.points)
        keep_cluster = False
        for plane_model in planes:
            a, b, c, d = plane_model
            distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
            if np.mean(distances) < tolerance:  # 클러스터가 평면에 가까운 경우
                keep_cluster = True
                break
        if keep_cluster:
            filtered_clusters.append(cluster)
    return filtered_clusters

# 떠 있는 객체 제외를 위한 설정
def filter_clusters_by_road_tolerance(clusters, road_plane, road_tolerance):
    filtered_clusters = []
    a, b, c, d = road_plane
    for cluster in clusters:
        points = np.asarray(cluster.points)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        if np.mean(distances) < road_tolerance:  # 도로 평면에 가까운 경우
            filtered_clusters.append(cluster)
    return filtered_clusters

# 다중 평면 추정
planes, inliers_list = multi_plane_segmentation(ror_pcd)

# 도로 평면 선택 (가장 큰 inliers를 가진 평면으로 가정)
road_plane_index = np.argmax([len(inliers) for inliers in inliers_list])
road_plane = planes[road_plane_index]

# 도로에 속하지 않는 포인트 추출
final_point = ror_pcd.select_by_index(np.hstack(inliers_list), invert=True)

# DBSCAN 클러스터링 적용
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))

# 클러스터 추출
clusters = [final_point.select_by_index(np.where(labels == i)[0]) for i in range(labels.max() + 1)]

# 클러스터와 평면 간 관계로 필터링
filtered_clusters = filter_clusters_by_plane_relation(clusters, planes)

# 도로 평면 기반 추가 필터링
road_tolerance = 2.0
road_filtered_clusters = filter_clusters_by_road_tolerance(filtered_clusters, road_plane, road_tolerance)

# 필터링 기준 설정
min_points_in_cluster = 5
max_points_in_cluster = 40
min_height = 0.5
max_height = 2.0
max_distance = 30.0

#형상 분석을 위한 조건 설정
min_aspect_ratio = 1.00
max_aspect_ratio = 2.50

# Bounding Box 생성 및 필터링
bboxes_road_filtered = []
for cluster in road_filtered_clusters:
    points = np.asarray(cluster.points)
    z_values = points[:, 2]
    z_min = z_values.min()
    z_max = z_values.max()
    height_diff = z_max - z_min
    if min_points_in_cluster <= len(points) <= max_points_in_cluster:
        if min_height <= height_diff <= max_height:
            distances = np.linalg.norm(points, axis=1)
            if distances.max() <= max_distance:
                bbox = cluster.get_axis_aligned_bounding_box()
                bbox.color = (1, 0, 0)  # 빨간색으로 설정
                bboxes_road_filtered.append(bbox)

# 형상 분석 필터링
bboxes_shape_filtered = []
for bbox in bboxes_road_filtered:
    extent = bbox.get_extent()
    height = extent[2]
    max_width = max(extent[0], extent[1])
    aspect_ratio = height / max_width
    if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
        bboxes_shape_filtered.append(bbox)

# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
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

# 평면 시각화
def visualize_planes(planes, point_cloud, plane_size=10.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RANSAC Planes Visualization")
    vis.add_geometry(point_cloud)
    for plane_model in planes:
        a, b, c, d = plane_model
        x = np.linspace(-plane_size, plane_size, 10)
        y = np.linspace(-plane_size, plane_size, 10)
        X, Y = np.meshgrid(x, y)
        Z = (-d - a * X - b * Y) / c
        plane_points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
        plane_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_points))
        plane_pcd.paint_uniform_color([1, 0, 0])  # 빨간색으로 설정
        vis.add_geometry(plane_pcd)
    vis.run()
    vis.destroy_window()

# 평면 시각화
#visualize_planes(planes, ror_pcd)
