import TDA_barcode.barcode as bc

# ----------------------------------------------------------
# Пример построения баркодов H0 и H1 для облака из 50 точек,
# расположенных около двух окружностей
# ----------------------------------------------------------

point_cloud = [bc.circle_points(1, 2, 2, 0.5) for _ in range(25)]
point_cloud.extend([bc.circle_points(4.5, 2, 1.5, 0.5) for _ in range(25)])

detailed_filtering = bc.initial_filtering(point_cloud)
matrix = bc.get_distance_matrix(point_cloud)
bc.visualization_complex(detailed_filtering, point_cloud, 0)

max_radius = 3
step = 0.5
bc.filtering(point_cloud, matrix, detailed_filtering,
             max_radius, step, visualization_VietorisRips=True)

barcode_H0 = bc.build_barcode_H0(detailed_filtering, max_radius)
barcode_H1 = bc.build_barcode_H1(detailed_filtering)
