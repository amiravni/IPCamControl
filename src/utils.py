from config import *
import datetime

def time2string(input_time):
    return datetime.datetime.fromtimestamp(input_time).strftime('%Y%m%d_%H%M%S')

def is_intersect(veci, vecj, margin):
    xi0 = veci[0]
    yi0 = veci[1]
    xi1 = veci[0] + veci[2]
    yi1 = veci[1] + veci[3]
    xj0 = vecj[0] - margin
    yj0 = vecj[1] - margin
    xj1 = vecj[0] + vecj[2] + margin
    yj1 = vecj[1] + vecj[3] + margin
    if (xi0 >= xj0 and xi0 <= xj1 and yi0 >= yj0 and yi0 <= yj1) or \
            (xi1 >= xj0 and xi1 <= xj1 and yi0 >= yj0 and yi0 <= yj1) or \
            (xi0 >= xj0 and xi0 <= xj1 and yi1 >= yj0 and yi1 <= yj1) or \
            (xi1 >= xj0 and xi1 <= xj1 and yi1 >= yj0 and yi1 <= yj1):
        return True
    else:
        return False


def is_in_bb(center_of_mass, img_shape, box_wh=BOX_WIDTH_HEIGHT):
    height, width, layers = img_shape
    vbox = (center_of_mass[0], center_of_mass[1], 1, 1)
    for i, box in enumerate(box_wh):
        bbox = (int(width * box[0]), int(height * box[2]), int(width * box[1] - width * box[0]),
                int(height * box[3] - height * box[2]))
        if is_intersect(vbox, bbox, 0):
            return i
    return 0


def get_connected_components(bb_final_graph, seen, idx, cnt, bb_graph):
    if len(bb_graph[idx]) == 0:
      return bb_final_graph , seen
    for j,bbj in enumerate(bb_graph[idx]):
       if bbj not in seen:
           bb_final_graph[cnt].append(bbj)
           seen.append(bbj)
           bb_final_graph , seen = get_connected_components(bb_final_graph, seen, bbj, cnt, bb_graph)
    return bb_final_graph, seen


def get_relevant_bb(bb_final_list, img_shape):
      rel_bb = []
      for bbox in bb_final_list:
            COM = (bbox[0]+int(bbox[2]/2), bbox[1]+int(bbox[3]/2))
            boxNum = is_in_bb(COM, img_shape)
            if boxNum > 0:
               rel_bb.append((boxNum, bbox[0], bbox[1], bbox[2], bbox[3]) )
      return rel_bb

