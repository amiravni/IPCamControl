from config import *
import time
from collections import deque
import utils


class FalseAlarmClassifier:
    def __init__(self):
        self.detectionData = deque(maxlen=DETECT_DATA_LEN_MAX)
        
    
    def run(self, bb_final_list, img_shape):
        if len(bb_final_list) == 0:
            self.detectionData.append((time.time(), 0, 0, 0, 0, 0, 0, []))
            return 0
        height, width, layers = img_shape
        maxImageArea = height * width
        sumAllAreas = sum([x[-1] for x in bb_final_list])
        sumNear = sum([x[-1] * ((x[1] + x[3] / 2) > height / 2) for x in bb_final_list])
        sumFar = sum([x[-1] * ((x[1] + x[3] / 2) < height / 2) for x in bb_final_list])
        maxSingleArea = bb_final_list[0][-1]
        xMin = min([x[0] for x in bb_final_list])
        yMin = min([x[1] for x in bb_final_list])
        xMax = max([x[0] + x[2] for x in bb_final_list])
        yMax = max([x[1] + x[3] for x in bb_final_list])
        maxAllAreas = (xMax - xMin) * (yMax - yMin)

        rel_bb = utils.get_relevant_bb(bb_final_list, img_shape)

        self.detectionData.append((
                                  time.time(),
                                  100.0 * sumAllAreas / maxImageArea,
                                  100.0 * sumNear / (maxImageArea / 2),
                                  100.0 * sumFar / (maxImageArea / 2),
                                  100.0 * maxSingleArea / maxImageArea,
                                  100.0 * maxAllAreas / maxImageArea,
                                  len(bb_final_list),
                                  rel_bb))
        # print(self.detectionData[-1])

        # print(len(self.detectionData))


        if len(self.detectionData) > DETECT_DATA_LEN_MIN:
            negCount = 0
            rel_bb_cnt = np.zeros(len(BOX_WIDTH_HEIGHT))
            FA_string = 'FA --> '
            for data in self.detectionData:
                if data[1] is not 0:
                    FA_string += '('
                    if data[1] > SUMALLAREAS_MAX:
                        negCount = negCount + 1
                        FA_string += '{}, '.format(str(1))
                        #LOGGER.info(("FA #1 - {}".format(str(data[1]))))
                    if data[2] > 0 and data[2] < SUMNEAR_MIN:
                        negCount = negCount + 1
                        FA_string += '{}, '.format(str(2))
                        #LOGGER.info(("FA #2 - {}".format(str(data[2]))))
                    if data[3] > SUMFAR_MAX:
                        negCount = negCount + 1
                        FA_string += '{}, '.format(str(3))
                        #LOGGER.info(("FA #3 - {}".format(str(data[3]))))
                    if data[4] > MAXSINGLEAREA_MAX:
                        negCount = negCount + 1
                        FA_string += '{}, '.format(str(4))
                        #LOGGER.info(("FA #4 - {}".format(str(data[4]))))
                    if data[5] > MAXALLAREA_MAX:
                        negCount = negCount + 1
                        FA_string += '{}, '.format(str(5))
                        #LOGGER.info(("FA #5 - {}".format(str(data[5]))))
                    if data[6] > LEN_BB_MAX:
                        negCount = negCount + 1
                        FA_string += '{}, '.format(str(6))
                        #LOGGER.info(("FA #6 - {}".format(str(data[6]))))

                    FA_string += ') '
                    FA_string = FA_string.replace('() ', '').replace(', )', ')')

                    used = []
                    for relData in data[7]:
                        if sum([x == relData[0] for x in used]) > 0:
                            continue
                        else:
                            rel_bb_cnt[relData[0]] = rel_bb_cnt[relData[0]] + 1
                            used.append(relData[0])

                if negCount > NEG_MAX:
                    #cv2.putText(self.img_curr_2show, "False Alarm", (int(width * 0.05), int(height * 0.05)),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    LOGGER.info(FA_string)
                    LOGGER.info("False Alarm")
                    return -1

            if negCount > 0:
                LOGGER.info(FA_string)

            if rel_bb_cnt[rel_bb_cnt > POS_MAX].any():
                #cv2.putText(self.img_curr_2show, "Positive Detection", (int(width * 0.05), int(height * 0.05)),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                LOGGER.info("Positive Detection")
                return 1

                # LOGGER.info("UNKNOWN")
        return 0
        # print(self.detectionData[-1] )