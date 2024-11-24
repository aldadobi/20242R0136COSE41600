import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

# PCD 파일이 저장된 디렉토리 경로
pcd_directory = "/home/youngeon/바탕화면/ad/COSE416_HW1_tutorial/COSE416_HW1_tutorial/data/01_straight_walk/pcd"
output_directory = "./output_frames"

# 출력 디렉토리 생성
os.makedirs(output_directory, exist_ok=True)

# Bounding Box 및 포인트 클라우드 시각화 함수 (이미지 저장 포함)
def visualize_and_save(pcd, bounding_boxes, output_path, point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.poll_events()
    vis.update_renderer()
    # 이미지 저장
    vis.capture_screen_image(output_path)
    vis.destroy_window()

# PCD 파일 처리
pcd_files = sorted([f for f in os.listdir(pcd_directory) if f.endswith('.pcd')])

for idx, file_name in enumerate(pcd_files):
    print(f"Processing {file_name} ({idx + 1}/{len(pcd_files)})")
    file_path = os.path.join(pcd_directory, file_name)

    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)

    # Voxel Downsampling
    voxel_size = 0.2
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Radius Outlier Removal (ROR)
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    # RANSAC 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)

    # 도로에 속하지 않는 포인트 추출
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    # DBSCAN 클러스터링
    labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=8, print_progress=False))

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

    # Bounding Box 생성
    bboxes = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            height_diff = z_max - z_min
            if min_height <= height_diff <= max_height:
                distances = np.linalg.norm(points, axis=1)
                if distances.max() <= max_distance:
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0)  # 빨간색
                    bboxes.append(bbox)

    # 도로 평면과의 거리로 Bounding Box 필터링
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

    # 시각화 및 저장
    output_path = os.path.join(output_directory, f"frame_{idx:04d}.png")
    visualize_and_save(final_point, bboxes_shape_filtered, output_path, point_size=2.0)

