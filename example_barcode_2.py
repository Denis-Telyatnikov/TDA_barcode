import TDA_barcode.barcode as bc

# ----------------------------------------------------------
# Пример построения баркодов H0 и H1 для облака из 1000 точек,
# расположенных около 5 окружностей.
# Фильтрация по подвыборке из 100 точек-ориентиров.
# ----------------------------------------------------------

# Облако из 1000 точек, расположенных около 5 окружностей
point_cloud = [bc.circle_points(0, 0, 8, 2) for _ in range(400)]
point_cloud.extend([bc.circle_points(2.3, 2, 2.8, 1) for _ in range(100)])
point_cloud.extend([bc.circle_points(-7.5, -15, 6, 2) for _ in range(200)])
point_cloud.extend([bc.circle_points(-12, -3, 4.5, 2) for _ in range(100)])
point_cloud.extend([bc.circle_points(7, -17, 6, 2) for _ in range(200)])

share = 10

# landmarks = bc.get_landmarks_random(point_cloud, share)
landmarks = bc.get_landmarks_maxmin(point_cloud, share)

detailed_filtering = bc.initial_filtering(landmarks)
matrix = bc.get_distance_matrix(landmarks)

bc.visualization_landmarks_complex(detailed_filtering, point_cloud, landmarks, 0)

max_radius = 7.5
step = 0.5
bc.filtering(landmarks, matrix, detailed_filtering,
            max_radius, step, witnesses=point_cloud, visualization_landmarks=True)
barcode_H0 = bc.build_barcode_H0(detailed_filtering, max_radius)
barcode_H1 = bc.build_barcode_H1(detailed_filtering)