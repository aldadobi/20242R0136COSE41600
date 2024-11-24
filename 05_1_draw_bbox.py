# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# pcd 파일 불러오기, 필요에 맞게 경로 수정
#file_path = "test_data/1727320101-665925967.pcd" # 1
#file_path = "test_data/1727320101-961578277.pcd" # 2
file_path = "test_data/1727320102-53276943.pcd" # 3

# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

# Radius Outlier Removal (ROR) 적용
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
ror_pcd = downsample_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=3,
                                             num_iterations=2000)

# 도로에 속하지 않는 포인트 (outliers) 추출
final_point = ror_pcd.select_by_index(inliers, invert=True)

# DBSCAN 클러스터링 적용
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))

# 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정

final_point.colors = o3d.utility.Vector3dVector(colors)

# 필터링 기준 설정
min_points_in_cluster = 5   # 클러스터 내 최소 포인트 수
max_points_in_cluster = 40  # 클러스터 내 최대 포인트 수
min_z_value = -1.0          # 클러스터 내 최소 Z값
max_z_value = 2.5           # 클러스터 내 최대 Z값
min_height = 0.5            # Z값 차이의 최소값
max_height = 2.0            # Z값 차이의 최대값
max_distance = 30.0         # 원점으로부터의 최대 거리

# 떠있는 객체 제외를 위한 설정
road_z_value = -plane_model[3] / plane_model[2]
road_tolerance = 2.0
# 형상 분석을 위한 조건 설정
min_aspect_ratio = 1.25  # 세로:가로 비율의 최소값, 앉아있거나
max_aspect_ratio = 2 # 세로:가로 비율의 최대값





# 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
bboxes_1234 = []
for i in range(labels.max() + 1):
    cluster_indices = np.where(labels == i)[0]
    if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
        cluster_pcd = final_point.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        z_values = points[:, 2]
        z_min = z_values.min()
        z_max = z_values.max()
        #if min_z_value <= z_min and z_max <= max_z_value:
        if road_z_value - road_tolerance <= z_min and z_max <= road_z_value + road_tolerance:   
            height_diff = z_max - z_min
            if min_height <= height_diff <= max_height:
                distances = np.linalg.norm(points, axis=1)
                if distances.max() <= max_distance:
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0) 
                    bboxes_1234.append(bbox)

bboxes_shape_filtered = []
for bbox in bboxes_1234:
    extent = bbox.get_extent()  # Bounding Box 크기 (Width, Depth, Height)
    height = extent[2]  # Z축 길이
    max_width = max(extent[0], extent[1])  # X, Y 축 중 더 큰 값을 가로로 간주
    aspect_ratio = height / max_width  # 세로:가로 비율 계산

    # 형상 조건을 만족하는 경우만 필터링
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

# 시각화 (포인트 크기를 원하는 크기로 조절 가능)
#visualize_with_bounding_boxes(final_point, bboxes_1234, point_size=2.0)
visualize_with_bounding_boxes(final_point, bboxes_shape_filtered, point_size=2.0)




###
def visualize_plane(plane_model, point_cloud, plane_size=10.0):

    a, b, c, d = plane_model
    
    x = np.linspace(-plane_size, plane_size, 10)
    y = np.linspace(-plane_size, plane_size, 10)
    X, Y = np.meshgrid(x, y)
    Z = (-d - a * X - b * Y) / c  # z = (-d - ax - by) / c

    # 평면 포인트 클라우드 생성
    plane_points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    plane_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(plane_points))
    plane_pcd.paint_uniform_color([1, 0, 0])  # 평면 색상을 빨간색으로 설정

    # 포인트 클라우드와 평면 시각화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RANSAC Plane Visualization")
    vis.add_geometry(point_cloud)  
    vis.add_geometry(plane_pcd)  
    vis.destroy_window()

# 평면 시각화
visualize_plane(plane_model, ror_pcd)

