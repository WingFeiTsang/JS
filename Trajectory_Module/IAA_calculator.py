def get_intersection(p1,p2,p3,p4):
    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    def subtract_vectors(v1, v2):
        return (v1[0] - v2[0], v1[1] - v2[1])

    def is_point_on_segment(point, segment_start, segment_end): # 判断点是否在线段上
        v1 = subtract_vectors(segment_end, segment_start)
        v2 = subtract_vectors(point, segment_start)
        cross = cross_product(v1, v2)

        if cross != 0:
            return False

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        if dot < 0 or dot > v1[0] * v1[0] + v1[1] * v1[1]:
            return False

        return True

    # 寻找交点
    def find_intersection(segment1_start, segment1_end, segment2_start, segment2_end): # 寻找交点
        v1 = subtract_vectors(segment1_end, segment1_start)
        v2 = subtract_vectors(segment2_end, segment2_start)
        v3 = subtract_vectors(segment2_start, segment1_start)

        cross1 = cross_product(v1, v3)
        cross2 = cross_product(v2, v3)

        if cross1 == 0 and cross2 == 0:
            if is_point_on_segment(segment2_start, segment1_start, segment1_end):
                return segment2_start
            if is_point_on_segment(segment2_end, segment1_start, segment1_end):
                return segment2_end
            if is_point_on_segment(segment1_start, segment2_start, segment2_end):
                return segment1_start
            if is_point_on_segment(segment1_end, segment2_start, segment2_end):
                return segment1_end

        denominator = cross_product(v1, v2)
        if denominator == 0:
            return None

        t1 = cross_product(v3, v2) / denominator
        t2 = cross_product(v3, v1) / denominator

        if t1 >= 0 and t1 <= 1 and t2 >= 0 and t2 <= 1:
            intersection_x = segment1_start[0] + t1 * v1[0]
            intersection_y = segment1_start[1] + t1 * v1[1]
            return intersection_x, intersection_y
        return None
    return find_intersection(p1, p2, p3, p4)

def get_trangle_area(p1,p2,p3): # 计算三角形面积
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)

def get_area(line1,line2): # 计算两条线段的面积
    area = 0
    i,j = len(line1)-2,len(line2)-2
    # plt.figure()
    while i >= 0 and j >= 0:
        intersection = get_intersection(line1[i],line1[i+1],line2[j],line2[j+1])
        # 有交点
        if not intersection is None:
            area += get_trangle_area(line1[i],line2[j],intersection)
            area += get_trangle_area(line1[i+1],line2[j+1],intersection)
        # 无交点
        else:
            area += get_trangle_area(line1[i],line2[j],line1[i+1])
            area += get_trangle_area(line1[i+1],line2[j+1],line2[j])
        i -= 1
        j -= 1
    return area