from cfg import *
from utils import utils
import cv2

#import pyprofilers as pp
#@pp.profile_by_line(exit=1)

class DiffHandler:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.bb_final_list = []
        self.kernel = np.ones((3, 3), np.uint8)

    def run(self, img_curr):

        fgmask = self.fgbg.apply(img_curr)
        fgmask = cv2.erode(fgmask, self.kernel, iterations=1)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        height, width, layers = img_curr.shape
        cnt = 0
        bb_list = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min(DIFF['min_bb_area']):
                cnt = cnt + 1
                # cv2.drawContours(self.img_curr_2show, contour, -1, (0, 0, 255), 3)
                x, y, w, h = cv2.boundingRect(contour)
                COM = (x + int(w / 2), y + int(h / 2))
                pct_min_bb_area = ((COM[1] / height)) * (max(DIFF['min_bb_area']) - min(DIFF['min_bb_area'])) + min(DIFF['min_bb_area'])
                # print(pctDIFF['min_bb_area'],area,COM)
                boxNum = utils.is_in_bb(COM, img_curr.shape)
                if (boxNum > 0 and area > DIFF['min_bb_area'][boxNum]) or (area > pct_min_bb_area):
                    bb_list.append((x, y, w, h))
                # if (cnt > 0 ):
                #    cv2.rectangle(self.img_curr_2show, (x,y), (x+w,y+h), (0, 0, 255), 4)
                #    cv2.putText(self.img_curr_2show,str(cnt-1),(x,y), font, 0.75,(255,0,0),1,cv2.LINE_AA)
                # print("------>",(x,y), (x+w,y+h))

        bb_mtx = np.zeros((len(bb_list), len(bb_list)))
        bb_pairs = []
        bb_graph = []
        bb_final_list = []
        bb_final_graph = []
        seen = []
        if len(bb_list) > 1:
            for i, bbi in enumerate(bb_list):
                for j, bbj in enumerate(bb_list):
                    if len(bb_graph) == i:
                        bb_graph.append([])
                    if i >= j:
                        if i == j:
                            bb_graph[i].append(j)
                        continue
                    veci = np.array(bbi)
                    vecj = np.array(bbj)
                    if utils.is_intersect(veci, vecj, 20) or utils.is_intersect(vecj, veci, 20):
                        bb_mtx[(i, j)] = 1
                        bb_mtx[(j, i)] = 1
                        bb_pairs.append((i, j))
                        bb_graph[i].append(j)
            # print("--->",bb_graph)

            cnt = 0
            for i, bbi in enumerate(bb_graph):
                if bbi[0] not in seen:
                    if len(bb_final_graph) == cnt:
                        bb_final_graph.append([])
                    bb_final_graph[cnt].append(bbi[0])
                    seen.append(bbi[0])
                    [bb_final_graph, seen] = utils.get_connected_components(bb_final_graph, seen, i, cnt, bb_graph)
                    cnt = cnt + 1

            # print(bb_final_graph)
            bb_list_np = np.array(bb_list)

            # print(bb_list_np)
            for i, bbi in enumerate(bb_final_graph):
                x = np.min(bb_list_np[bbi, 0])
                y = np.min(bb_list_np[bbi, 1])
                x1 = np.max(bb_list_np[bbi, 0] + bb_list_np[bbi, 2])
                y1 = np.max(bb_list_np[bbi, 1] + bb_list_np[bbi, 3])
                w = x1 - x
                h = y1 - y
                bb_final_list.append((x, y, w, h, w * h))
                # print("--x-->",x,y,x1,y1)

        return sorted(bb_final_list, key=lambda x: x[4], reverse=True)
