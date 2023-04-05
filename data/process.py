import math
import cv2
import numpy as np

## calculate x, y after rotation
def rotate_point(x, y, cx, cy, angle):
    new_x = (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle) + cx
    new_y = (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy
    return new_x, new_y

## calculate bounding box after rotation
def get_bounding_box(x, y, scale, rotate):
    length = scale * 50
    x_l = x - length / 2
    y_t = y - length / 2
    x_r = x + length / 2
    y_b = y + length / 2

    x1, y1 = rotate_point(x_l, y_t, x, y, rotate)
    x2, y2 = rotate_point(x_r, y_t, x, y, rotate)
    x3, y3 = rotate_point(x_r, y_b, x, y, rotate)
    x4, y4 = rotate_point(x_l, y_b, x, y, rotate)

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

def draw_square_by_points(image, points, **params):
    last_x, last_y = points[-1]
    
    if "color" not in params:
        params["color"] = (255, 0, 0)
    if "thickness" not in params:
        params["thickness"] = 2

    for x, y in points:
        image = cv2.line(image, (int(last_x), int(last_y)), (int(x), int(y)), params["color"], params["thickness"])
        last_x, last_y = x, y
    return image

## draw bounding box on image
def draw_square_by_label(image, x, y, scale, rotate, **params):
    points = get_bounding_box(x, y, scale, rotate)
    return draw_square_by_points(image, points, **params)

## get center point of points
def get_center_point(points):
    x = 0
    y = 0
    for point in points:
        x += point[0]
        y += point[1]
    return x / len(points), y / len(points)

## get area by points
def get_area_by_points(points):
    last_x, last_y = points[-1]
    area = 0
    for x, y in points:
        area += (x - last_x) * (y + last_y) / 2
        last_x, last_y = x, y
    return abs(area)

## calculate area of bounding box
def get_area(square):
    x1, y1 = square[0]
    x2, y2 = square[1]
    return (y2 - y1) ** 2 + (x2 - x1) ** 2

def get_line_length(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

## The very rough estimation of union area
def get_union_area(squareA, squareB):
    points = np.array(squareA + squareB)
    x = max(points[:, 0]) - min(points[:, 0])
    y = max(points[:, 1]) - min(points[:, 1])
    
    squareA = np.array(squareA, dtype=np.int32)
    squareB = np.array(squareB, dtype=np.int32)
    squareA[:,0] -= int(min(points[:, 0]))
    squareA[:,1] -= int(min(points[:, 1]))
    squareB[:,0] -= int(min(points[:, 0]))
    squareB[:,1] -= int(min(points[:, 1]))

    canvas = np.zeros((int(y), int(x)), dtype=np.uint8)

    canvas = cv2.fillPoly(canvas, [squareA], 255, lineType=cv2.LINE_4)
    canvas = cv2.fillPoly(canvas, [squareB], 255, lineType=cv2.LINE_4)
    cv2.imwrite("union.jpg", canvas)
    return cv2.countNonZero(canvas)


###########################################################################################################
## precise calculation for union area
##########################################################################################################
def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


## calculate intersection point of two lines
def get_intersection_point(point_1, point_2, point_3, point_4):
    x1, y1 = point_1
    x2, y2 = point_2
    x3, y3 = point_3
    x4, y4 = point_4
    try:
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        if intersect(point_1, point_2, point_3, point_4):
            return x, y
        return None
    except:
        return None

## check if a point in a bounding box
def is_point_in_bounding_box(point, points):
    x, y = point
    
    tl, tr, br, bl = points

    if (intersect((x, y), (x, min(tl[1], tr[1])-10), tl, tr) or intersect((x, y), (x, min(tr[1], br[1])-10), tr, br) or intersect((x, y), (x, min(br[1], bl[1])-10), br, bl) or intersect((x, y), (x, min(bl[1], tl[1])-10), bl, tl)) and \
        (intersect((x, y), (x, max(tl[1], tr[1])+10), tl, tr) or intersect((x, y), (x, max(tr[1], br[1])+10), tr, br) or intersect((x, y), (x, max(br[1], bl[1])+10), br, bl) or intersect((x, y), (x, max(bl[1], tl[1])+10), bl, tl)) and \
        (intersect((x, y), (max(tl[0], tr[0])+10, y), tl, tr) or intersect((x, y), (max(tr[0], br[0])+10, y), tr, br) or intersect((x, y), (max(br[0], bl[0])+10, y), br, bl) or intersect((x, y), (max(bl[0], tl[0])+10, y), bl, tl)) and \
        (intersect((x, y), (min(tl[0], tr[0])-10, y), tl, tr) or intersect((x, y), (min(tr[0], br[0])-10, y), tr, br) or intersect((x, y), (min(br[0], bl[0])-10, y), br, bl) or intersect((x, y), (min(bl[0], tl[0])-10, y), bl, tl)):
        return True
    
    return False

def get_intersection_area_points(square_A, square_B):
    ## check if one square is in another square
    A_in_B = []
    for point in square_A:
        if is_point_in_bounding_box(point, square_B):
            A_in_B.append(point)
    
    B_in_A = []
    for point in square_B:
        if is_point_in_bounding_box(point, square_A):
            B_in_A.append(point)
    
    if len(A_in_B) == 4 and len(B_in_A) == 0:
        return square_A
    elif len(B_in_A) == 4 and len(A_in_B) == 0:
        return square_B
    elif len(A_in_B) == 0 and len(B_in_A) == 0:
        return []
    
    drop_points = set(A_in_B + B_in_A)

    ## get connections between droped points
    lines = {}
    for points in drop_points:
        lines[points] = set()

    last = square_A[-1]
    for point in square_A:
        if (last in drop_points) and (point in drop_points):
            lines[last].add(point)
            lines[point].add(last)
        last = point

    last = square_B[-1]
    for point in square_B:
        if (last in drop_points) and (point in drop_points):
            lines[last].add(point)
            lines[point].add(last)
        last = point

    bound_points = drop_points.copy()

    ## get connections between intersection points and intersection points/droped points
    for first, second in [(square_A, square_B), (square_B, square_A)]:
        last_f = first[-1]
        for point_f in first:
            last_s = second[-1]
            last_intersection = None

            for point_s in second:
                intersection = get_intersection_point(last_f, point_f, last_s, point_s)
                if intersection:
                    if intersection not in lines:
                        lines[intersection] = set()
                    
                    bound_points.add(intersection)

                    if last_intersection:
                        lines[intersection].add(last_intersection)
                        lines[last_intersection].add(intersection)
                    for point in [last_f, point_f, last_s, point_s]:
                        if point in drop_points:
                            lines[intersection].add(point)
                            lines[point].add(intersection)
                    last_intersection = intersection
                last_s = point_s
            last_f = point_f

    inter_points = []

    point = bound_points.pop()
    inter_points.append(point)
    while len(bound_points) > 0:
        for next_point in lines[point]:
            if next_point in bound_points:
                bound_points.remove(next_point)
                inter_points.append(next_point)
                point = next_point
                break

    return inter_points

def get_intersection_area(square_A, square_B):
    intersection_points = get_intersection_area_points(square_A, square_B)
    if len(intersection_points) < 3:
        return 0
    return get_area_by_points(intersection_points)

## calculate iou of two bounding boxes
def get_iou(square_A, square_B):
    total_area = int(get_area(square_A) + get_area(square_B))
    inter_area = int(get_intersection_area(square_A, square_B))
    union_area = total_area - inter_area
    if inter_area > union_area:
        raise Exception(f"Intersection area is greater than Union area. inter {inter_area}, union {union_area}")        
    return abs(inter_area / union_area)

