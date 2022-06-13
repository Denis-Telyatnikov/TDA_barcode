import matplotlib.pyplot as plt
import matplotlib.patches
import random
from math import pi, cos, sin
import numpy as np

random.seed(10)

# Облако точек, расположенных около окружности
# h, k - координаты центра, r - радиус, noise - шум
def circle_points(h, k, r, noise):
    noise_x = random.uniform(-noise, noise)
    noise_y = random.uniform(-noise, noise)
    theta = random.random() * 2 * pi
    return [(h + cos(theta) * r) + noise_x, (k + sin(theta) * r) + noise_y]

# Облако точек, расположенных около сферы
# x0, y0, z0 - координаты центра, r - радиус, noise - шум
def sphere_points(x0, y0, z0, r, noise):
    noise_x = random.uniform(-noise, noise)
    noise_y = random.uniform(-noise, noise)
    noise_z = random.uniform(-noise, noise)
    theta = random.random() * 2 * pi
    phi = random.random() * 2 * pi
    return [(x0 + sin(theta) * cos(phi) * r) + noise_x,
            (y0 + sin(theta) * sin(phi) * r) + noise_y,
            (z0 + cos(theta) * r) + noise_z]

# Матрица расстояний для облака точек
def get_distance_matrix(point_cloud):
    n = len(point_cloud)
    distance_matrix = []

    for i in range(n):
        lst = []
        for j in range(n):
            d = 0
            for coord in range(len(point_cloud[i])):
                d += (point_cloud[i][coord] - point_cloud[j][coord]) ** 2
            d = d ** (1/2)
            lst.append(d)
        distance_matrix.append(lst)
    return distance_matrix


# Матрица смежности для построения треугольников
def get_adjacency_matrix(e, n):
    v = [i for i in range(n)]
    m = []
    for i in range(len(v)):
        lst = []
        for j in range(len(v)):
            if i == j:
                lst.append(0)
            elif [v[i], v[j]] in e or [v[j], v[i]] in e:
                lst.append(1)
            else:
                lst.append(0)
        m.append(lst)
    return m


def intersection_list(list1, list2):
   list3 = [value for value in list1 if value in list2]
   return list3


def construction_simplicial_complex(point_cloud, distance_matrix, filtering, r):
    n = len(point_cloud)
    k = 1
    edges = []
    for i in range(n - 1):
        for j in range(k, n):
            if distance_matrix[i][j] <= r and [i, j] not in filtering[0]:
                filtering[0].append([i, j])
                filtering[1].append(r)
        k += 1

    for lst in filtering[0]:
        if len(lst) == 2:
            edges.append(lst)


    matrix = get_adjacency_matrix(edges, n)
    h = len(matrix)
    for a in range(0, h):
        for b in range(a + 1, h):
            if not matrix[a][b]:
                continue
            for c in range(b + 1, h):
                if matrix[b][c] and matrix[a][c] and [a, b, c] not in filtering[0]:
                    filtering[0].append([a, b, c])
                    filtering[1].append(r)

    # ----
    # triangles = []
    # for sim in filtering[0]:
    #     if len(sim) == 3:
    #         triangles.append(sim)
    #
    # matrix_triangles = []
    # for i in range(len(triangles)):
    #     lst = []
    #     for j in range(len(triangles)):
    #         if i == j:
    #             lst.append(0)
    #         elif len(intersection_list(triangles[i], triangles[j])) == 2:
    #             lst.append(1)
    #         else:
    #             lst.append(0)
    #     matrix_triangles.append(lst)
    # triangle_indexes = []
    # h = len(matrix_triangles)
    # for a in range(0, h):
    #     for b in range(a + 1, h):
    #         if not matrix_triangles[a][b]:
    #             continue
    #         for c in range(b + 1, h):
    #             if not matrix_triangles[b][c] or not matrix_triangles[a][c]:
    #                 continue
    #             for d in range(c + 1, h):
    #                 if matrix_triangles[a][d] and matrix_triangles[b][d] \
    #                         and matrix_triangles[c][d] and [a, b, c, d] not in triangle_indexes:
    #                     triangle_indexes.append([a, b, c, d])
    #
    # for h in triangle_indexes:
    #     tetrahedron = []
    #     for j in h:
    #         for l in triangles[j]:
    #             if l not in tetrahedron:
    #                 tetrahedron.append(l)
    #     if tetrahedron not in filtering[0]:
    #         filtering[0].append(sorted(tetrahedron))
    #         filtering[1].append(r)

    return filtering


def visualization_complex(filtered_complex, point_array, t):
    x0, y0, x1, y1, x2, y2 = [], [], [], [], [], []
    fig, ax = plt.subplots()
    for simpl in filtered_complex[0]:
        if len(simpl) == 1:
            point = point_array[simpl[0]]
            x0.append(point[0])
            y0.append(point[1])
        elif len(simpl) == 2:
            point1 = point_array[simpl[0]]
            point2 = point_array[simpl[1]]
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='#191970')
        elif len(simpl) == 3:
            point1 = point_array[simpl[0]]
            point2 = point_array[simpl[1]]
            point3 = point_array[simpl[2]]
            polygon = matplotlib.patches.Polygon([(point1[0], point1[1]),
                                                  (point2[0], point2[1]),
                                                  (point3[0], point3[1])], color='#66CDAA')
            ax.add_patch(polygon)
    ax.scatter(x=x0, y=y0, s=20, c='#006400')
    plt.suptitle(f"r = {t}")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def visualization_landmarks_complex(filtered_complex, point_array, landmarks, t):
    x0, y0, x1, y1, x2, y2 = [], [], [], [], [], []
    x, y = [], []
    fig, ax = plt.subplots()
    for i in point_array:
        x.append(i[0])
        y.append(i[1])
    for simpl in filtered_complex[0]:
        if len(simpl) == 1:
            point = landmarks[simpl[0]]
            x0.append(point[0])
            y0.append(point[1])
        elif len(simpl) == 2:
            point1 = landmarks[simpl[0]]
            point2 = landmarks[simpl[1]]
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color='#191970')
        elif len(simpl) == 3:
            point1 = landmarks[simpl[0]]
            point2 = landmarks[simpl[1]]
            point3 = landmarks[simpl[2]]
            polygon = matplotlib.patches.Polygon([(point1[0], point1[1]),
                                                  (point2[0], point2[1]),
                                                  (point3[0], point3[1])], color='#66CDAA')
            ax.add_patch(polygon)
    ax.scatter(x=x, y=y, s=20, c='#006400')
    ax.scatter(x=x0, y=y0, s=30, c='#8B0000', marker='s', label='ориентиры')
    plt.suptitle(f"r = {t}")
    plt.xticks([])
    plt.yticks([])
    plt.show()


# Первоначальная фильтрация(r=0), состоящая из 0-мерных симплексов
def initial_filtering(point_cloud):
    n = len(point_cloud)
    x, y = [], []
    for points in point_cloud:
        x.append(points[0])
        y.append(points[1])
    # plt.scatter(x, y, s=20, c='g')
    # plt.show()
    detailed_filtering = [[], []]
    for i in range(n):
        lst = [i]
        detailed_filtering[0].append(lst)
        detailed_filtering[1].append(0)
    return detailed_filtering


# Фильтрация
def filtering(point_cloud, distance_matrix, detailed_filtering,
                           max_radius, step, visualization_VietorisRips=False,
                           visualization_landmarks=False, witnesses=[]):
    # dim = len(point_cloud[0])
    for r in np.arange(0 + step, max_radius + step, step):
        construction_simplicial_complex(point_cloud, distance_matrix, detailed_filtering, r)
        if visualization_VietorisRips:
            visualization_complex(detailed_filtering, point_cloud, r)
        elif visualization_landmarks:
            visualization_landmarks_complex(detailed_filtering, witnesses, point_cloud,  r)


# Конфликтуют ли столбцы граничной матрицы
def conflicting_columns(lst):
    for t1 in range(1, len(lst)):
        for t2 in range(0, t1):
            if lst[t1] == lst[t2] and lst[t1] != -1:
                # print(t2, "конфликтует с", t1)
                return t2, t1
    return 0, 0


def build_full_barcode(detailed_filtering, max_radius, visualization=True):
    s = len(detailed_filtering[0])
    boundary_matrix = [[0 for _ in range(s)] for _ in range(s)]
    for i in range(s - 1):
        for j in range(i + 1, s):
            if len(detailed_filtering[0][i]) == 1 and len(detailed_filtering[0][j]) == 2 and (
                    detailed_filtering[0][i][0] in detailed_filtering[0][j]):
                boundary_matrix[i][j] = 1
            elif len(detailed_filtering[0][i]) == 2 and len(detailed_filtering[0][j]) == 3 and (
                    detailed_filtering[0][i][0] in detailed_filtering[0][j]) and (
                    detailed_filtering[0][i][1] in detailed_filtering[0][j]):
                boundary_matrix[i][j] = 1
    L = [-1 for _ in range(s)]
    for j in range(s):
        for i in range(s):
            if boundary_matrix[s - i - 1][j] == 1:
                L[j] = s - i - 1
                break
    left_j, right_j = conflicting_columns(L)

    while left_j != 0 and right_j != 0:
        for i in range(s):
            boundary_matrix[i][right_j] = (boundary_matrix[i][right_j] + boundary_matrix[i][left_j]) % 2

        L = [-1 for _ in range(s)]
        for j in range(s):
            for i in range(s):
                if boundary_matrix[s - i - 1][j] == 1:
                    L[j] = s - i - 1
                    break
        left_j, right_j = conflicting_columns(L)

    pair_list = []
    for i in range(len(L) - 1):
        if L[i] == -1:
            pair = 0
            for j in range(i + 1, len(L)):
                if L[j] == i:
                    pair = 1
                    # print(detailed_filtering[0][i], "--", detailed_filtering[0][j])
                    pair_list.append([detailed_filtering[0][i], detailed_filtering[0][j]])
                    break
            if pair == 0:
                # print(detailed_filtering[0][i], "-- infinity")
                pair_list.append([detailed_filtering[0][i], ['infinity']])

    barcode_R2 = [[], []]
    y_bar = 0
    for p in pair_list:
        if len(p[0]) == 1 and ['infinity'] in p:
            x1 = detailed_filtering[1][detailed_filtering[0].index(p[0])]
            x2 = max_radius + 2
            # print(p, x1, x2)
            y_bar += 0.5
            plt.plot([x1, x2], [y_bar, y_bar], c='#228B22', linewidth=3)
            plt.scatter(x2, y_bar, s=90, c='#228B22', marker='+')
            barcode_R2[0].append([x1, 'infinity'])
        if len(p[1]) == 2:
            x1 = detailed_filtering[1][detailed_filtering[0].index(p[0])]
            x2 = detailed_filtering[1][detailed_filtering[0].index(p[1])]
            # print(p, x1, x2)
            y_bar += 0.5
            plt.plot([x1, x2], [y_bar, y_bar], c='#228B22', linewidth=3)
            barcode_R2[0].append([x1, x2])
    if visualization:
        plt.yticks([])
        plt.show()

    y_bar = 0
    for p in pair_list:
        if len(p[0]) == 2 and len(p[1]) == 3:
            x1 = detailed_filtering[1][detailed_filtering[0].index(p[0])]
            x2 = detailed_filtering[1][detailed_filtering[0].index(p[1])]
            if x1 - x2 != 0:
                y_bar += 0.5
                plt.plot([x1, x2], [y_bar, y_bar], c='#1E90FF', linewidth=3)
                barcode_R2[1].append([x1, x2])
    if visualization:
        plt.yticks([])
        plt.show()
    return barcode_R2


def get_landmarks_maxmin(point_cloud, share):
    value = int(len(point_cloud) / share)
    landmarks = []
    landmarks.append(random.choice(point_cloud))
    matrix = []
    for i in point_cloud:
        d = [((i[0] - landmarks[0][0]) ** 2 + (i[1] - landmarks[0][1]) ** 2) ** (1 / 2)]
        matrix.append(d)
    while len(landmarks) < value:
        min_values = []
        for i in matrix:
            min_values.append(min(i))
        t = min_values.index(max(min_values))
        landmarks.append(point_cloud[t])
        for i in range(len(point_cloud)):
            d = ((point_cloud[i][0] - point_cloud[t][0]) ** 2 +
                 (point_cloud[i][1] - point_cloud[t][1]) ** 2) ** (1 / 2)
            matrix[i].append(d)
    return landmarks


def get_landmarks_random(point_cloud, share):
    value = int(len(point_cloud) / share)
    lst = [i for i in range(0, len(point_cloud))]
    landmarks_ind = np.random.choice(lst, value, replace=False)
    landmarks = []
    for i in landmarks_ind:
        landmarks.append(point_cloud[i])
    return landmarks



def build_barcode_H0(detailed_filtering, max_radius, visualization=True):
    detailed_filtering_H0 = [[], []]
    for i in range(len(detailed_filtering[0])):
        if len(detailed_filtering[0][i]) == 1 or len(detailed_filtering[0][i]) == 2:
            detailed_filtering_H0[0].append(detailed_filtering[0][i])
            detailed_filtering_H0[1].append(detailed_filtering[1][i])
    s = len(detailed_filtering_H0[0])
    boundary_matrix = [[0 for _ in range(s)] for _ in range(s)]
    for i in range(s - 1):
        for j in range(i + 1, s):
            if len(detailed_filtering_H0[0][i]) == 1 and len(detailed_filtering_H0[0][j]) == 2 and (
                    detailed_filtering_H0[0][i][0] in detailed_filtering_H0[0][j]):
                boundary_matrix[i][j] = 1
    L = [-1 for _ in range(s)]
    for j in range(s):
        for i in range(s):
            if boundary_matrix[s - i - 1][j] == 1:
                L[j] = s - i - 1
                break
    left_j, right_j = conflicting_columns(L)
    it = 1
    while left_j != 0 and right_j != 0:
        for i in range(s):
            boundary_matrix[i][right_j] = (boundary_matrix[i][right_j] + boundary_matrix[i][left_j]) % 2

        L = [-1 for _ in range(s)]
        for j in range(s):
            for i in range(s):
                if boundary_matrix[s - i - 1][j] == 1:
                    L[j] = s - i - 1
                    break
        left_j, right_j = conflicting_columns(L)
        print(it)
        it += 1
    pair_list = []
    for i in range(len(L) - 1):
        if L[i] == -1:
            pair = 0
            for j in range(i + 1, len(L)):
                if L[j] == i:
                    pair = 1
                    # print(detailed_filtering[0][i], "--", detailed_filtering[0][j])
                    pair_list.append([detailed_filtering_H0[0][i], detailed_filtering_H0[0][j]])
                    break
            if pair == 0:
                # print(detailed_filtering[0][i], "-- infinity")
                pair_list.append([detailed_filtering_H0[0][i], ['infinity']])

    barcode_H0 = []
    y_bar = 0
    for p in pair_list:
        if len(p[0]) == 1 and ['infinity'] in p:
            x1 = detailed_filtering_H0[1][detailed_filtering_H0[0].index(p[0])]
            x2 = max_radius + 2
            # print(p, x1, x2)
            y_bar += 0.5
            plt.plot([x1, x2], [y_bar, y_bar], c='#228B22', linewidth=3)
            plt.scatter(x2, y_bar, s=90, c='#228B22', marker='+')
            barcode_H0.append([x1, 'infinity'])
        if len(p[1]) == 2:
            x1 = detailed_filtering_H0[1][detailed_filtering_H0[0].index(p[0])]
            x2 = detailed_filtering_H0[1][detailed_filtering_H0[0].index(p[1])]
            # print(p, x1, x2)
            y_bar += 0.5
            plt.plot([x1, x2], [y_bar, y_bar], c='#228B22', linewidth=3)
            barcode_H0.append([x1, x2])
    if visualization:
        plt.yticks([])
        plt.show()
    return barcode_H0


def build_barcode_H1(detailed_filtering, visualization=True):
    detailed_filtering_H1 = [[], []]
    for i in range(len(detailed_filtering[0])):
        if len(detailed_filtering[0][i]) == 2 or len(detailed_filtering[0][i]) == 3:
            detailed_filtering_H1[0].append(detailed_filtering[0][i])
            detailed_filtering_H1[1].append(detailed_filtering[1][i])
    s = len(detailed_filtering_H1[0])
    print(s)
    boundary_matrix = [[0 for _ in range(s)] for _ in range(s)]
    for i in range(s - 1):
        for j in range(i + 1, s):
            if len(detailed_filtering_H1[0][i]) == 2 and len(detailed_filtering_H1[0][j]) == 3 and (
                    detailed_filtering_H1[0][i][0] in detailed_filtering_H1[0][j]) and (
                    detailed_filtering_H1[0][i][1] in detailed_filtering_H1[0][j]):
                boundary_matrix[i][j] = 1
    L = [-1 for _ in range(s)]
    for j in range(s):
        for i in range(s):
            if boundary_matrix[s - i - 1][j] == 1:
                L[j] = s - i - 1
                break
    left_j, right_j = conflicting_columns(L)
    it = 1
    while left_j != 0 and right_j != 0:
        for i in range(s):
            boundary_matrix[i][right_j] = (boundary_matrix[i][right_j] + boundary_matrix[i][left_j]) % 2

        L = [-1 for _ in range(s)]
        for j in range(s):
            for i in range(s):
                if boundary_matrix[s - i - 1][j] == 1:
                    L[j] = s - i - 1
                    break
        left_j, right_j = conflicting_columns(L)
        print(it)
        it += 1
    pair_list = []
    for i in range(len(L) - 1):
        if L[i] == -1:
            pair = 0
            for j in range(i + 1, len(L)):
                if L[j] == i:
                    pair = 1
                    # print(detailed_filtering[0][i], "--", detailed_filtering[0][j])
                    pair_list.append([detailed_filtering_H1[0][i], detailed_filtering_H1[0][j]])
                    break
            if pair == 0:
                # print(detailed_filtering[0][i], "-- infinity")
                pair_list.append([detailed_filtering_H1[0][i], ['infinity']])
    barcode_H1 = []
    y_bar = 0
    for p in pair_list:
        if len(p[0]) == 2 and len(p[1]) == 3:
            x1 = detailed_filtering_H1[1][detailed_filtering_H1[0].index(p[0])]
            x2 = detailed_filtering_H1[1][detailed_filtering_H1[0].index(p[1])]
            if x1 - x2 != 0:
                y_bar += 0.5
                plt.plot([x1, x2], [y_bar, y_bar], c='#1E90FF', linewidth=3)
                barcode_H1.append([x1, x2])
    if visualization:
        plt.yticks([])
        plt.show()
    return barcode_H1


def build_barcode_H2(detailed_filtering, visualization=True):
    detailed_filtering_H2 = [[], []]
    for i in range(len(detailed_filtering[0])):
        if len(detailed_filtering[0][i]) == 3 or len(detailed_filtering[0][i]) == 4:
            detailed_filtering_H2[0].append(detailed_filtering[0][i])
            detailed_filtering_H2[1].append(detailed_filtering[1][i])
    s = len(detailed_filtering_H2[0])
    print(s)
    boundary_matrix = [[0 for _ in range(s)] for _ in range(s)]
    for i in range(s - 1):
        for j in range(i + 1, s):
            if len(detailed_filtering_H2[0][i]) == 3 and len(detailed_filtering_H2[0][j]) == 4 and (
                    detailed_filtering_H2[0][i][0] in detailed_filtering_H2[0][j]) and (
                    detailed_filtering_H2[0][i][1] in detailed_filtering_H2[0][j]):
                boundary_matrix[i][j] = 1
    L = [-1 for _ in range(s)]
    for j in range(s):
        for i in range(s):
            if boundary_matrix[s - i - 1][j] == 1:
                L[j] = s - i - 1
                break
    left_j, right_j = conflicting_columns(L)
    it = 1
    while left_j != 0 and right_j != 0:
        for i in range(s):
            boundary_matrix[i][right_j] = (boundary_matrix[i][right_j] + boundary_matrix[i][left_j]) % 2

        L = [-1 for _ in range(s)]
        for j in range(s):
            for i in range(s):
                if boundary_matrix[s - i - 1][j] == 1:
                    L[j] = s - i - 1
                    break
        left_j, right_j = conflicting_columns(L)
        print(it)
        it += 1
    pair_list = []
    for i in range(len(L) - 1):
        if L[i] == -1:
            pair = 0
            for j in range(i + 1, len(L)):
                if L[j] == i:
                    pair = 1
                    # print(detailed_filtering[0][i], "--", detailed_filtering[0][j])
                    pair_list.append([detailed_filtering_H2[0][i], detailed_filtering_H2[0][j]])
                    break
            if pair == 0:
                # print(detailed_filtering[0][i], "-- infinity")
                pair_list.append([detailed_filtering_H2[0][i], ['infinity']])
    barcode_H2 = []
    y_bar = 0
    for p in pair_list:
        if len(p[0]) == 3 and len(p[1]) == 4:
            x1 = detailed_filtering_H2[1][detailed_filtering_H2[0].index(p[0])]
            x2 = detailed_filtering_H2[1][detailed_filtering_H2[0].index(p[1])]
            if x1 - x2 != 0:
                y_bar += 0.5
                plt.plot([x1, x2], [y_bar, y_bar], c='#8B008B', linewidth=3)
                barcode_H2.append([x1, x2])
    if visualization:
        plt.yticks([])
        plt.show()
    return barcode_H2


# Визуализация баркода
def visualize_barcode(barcode):
    colors = ['#1E90FF', '#8B008B', '#B8860B', '#B22222']
    for bar in range(len(barcode)):
        y_bar = 0
        if bar == 0:
            for i in barcode[bar]:
                if i[1] != 'infinity':
                    y_bar += 0.5
                    plt.plot([i[0], i[1]], [y_bar, y_bar], c='#228B22', linewidth=3)
                else:
                    y_bar += 0.5
                    x = i[0] + 14
                    plt.plot([i[0], x], [y_bar, y_bar], c='#228B22', linewidth=3)
                    plt.scatter(x, y_bar, s=90, c='#228B22', marker='+')
            plt.suptitle(f'H{bar}')
            plt.yticks([])
            plt.show()
        else:
            for i in barcode[bar]:
                y_bar += 0.5
                plt.plot([i[0], i[1]], [y_bar, y_bar], c=colors[(bar - 1) % 4], linewidth=3)
            plt.suptitle(f'H{bar}')
            plt.yticks([])
            plt.show()


def two_barcodes_on_diagram(barcode1, barcode2):
    plt.plot([0, 12], [0, 12], "--", c='#BC8F8F', linewidth=1)
    colors = ['#1E90FF', '#8B008B', '#B8860B', '#B22222']
    for bar in range(len(barcode1)):
        if bar == 0:
            continue
        else:
            for i in barcode1[bar]:
                plt.scatter(i[0], i[1], s=50, c=colors[(bar - 1) % 4], marker='o')
    for bar in range(len(barcode2)):
        if bar == 0:
            continue
        else:
            for i in barcode2[bar]:
                plt.scatter(i[0], i[1], s=50, c=colors[(bar - 1) % 4],
                            marker='s', edgecolor='#FF4500')
    plt.xlabel("birth")
    plt.ylabel("death")
    plt.show()

# Сравнение баркодов
def compare_two_barcodes(barcode1, barcode2, insignificant):
    if len(barcode1) != len(barcode2):
        print("Баркоды не являются сравнимыми!")
    else:
        print('Расстояние между баркодами группы:')
        for bar in range(len(barcode1)):
            if bar == 0:
                max_d1, max_d2 = 0, 0
                for seg in barcode1[bar]:
                    if seg[1] != 'infinity' and seg[1] > max_d1:
                        max_d1 = seg[1]
                for seg in barcode2[bar]:
                    if seg[1] != 'infinity' and seg[1] > max_d2:
                        max_d2 = seg[1]
                distH0 = abs(max_d1 - max_d2)
                print('H0: ', distH0)
            else:
                print(f'H{bar}:  ', end="")
                bc1, bc2 = [], []
                for seg in barcode1[bar]:
                    if seg[1] - seg[0] >= insignificant:
                        bc1.append(seg)
                for seg in barcode2[bar]:
                    if seg[1] - seg[0] >= insignificant:
                        bc2.append(seg)
                # print(bc1)
                # print(bc2)
                if len(bc1) >= len(bc2):
                    max_dist = 0
                    for p1 in bc2:
                        dist = []
                        for p2 in bc1:
                            dist.append(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2))
                        if max_dist < min(dist):
                            max_dist = min(dist)
                        bc1.pop(dist.index(min(dist)))
                    for p in bc1:
                        dist_diagonal = abs(p[0] - p[1]) / (2 ** 0.5)
                        if dist_diagonal > max_dist:
                            max_dist = dist_diagonal
                    print(max_dist)

                elif len(bc1) < len(bc2):
                    max_dist = 0
                    for p1 in bc1:
                        dist = []
                        for p2 in bc2:
                            dist.append(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2))
                        if max_dist < min(dist):
                            max_dist = min(dist)
                        bc2.pop(dist.index(min(dist)))
                    for p in bc2:
                        dist_diagonal = abs(p[0] - p[1]) / (2 ** 0.5)
                        if dist_diagonal > max_dist:
                            max_dist = dist_diagonal
                    print(max_dist)