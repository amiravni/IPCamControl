
from cfg import *
import time
import datetime
from collections import deque
from utils import utils
from threading import Thread
import cv2
import os
os.environ["DARKNET_PATH"] = './darknet/'
from darknet import darknet
from utils.VideoHandler import VideoHandler, VideosListHandler
import traceback

class DarkNetClassifier:
    def __init__(self, config_file="cfg/yolov4-leaky-416.cfg",
                 weights="yolov4-leaky-416.weights",
                 data_file='./cfg/coco.data',
                 darknet_path=os.environ.get('DARKNET_PATH', '../../darknet/'),
                 thresh=0.4,
                 queueSize=128, debug=False):

        self.init_done = False
        self.stopped = True
        self.debug = debug
        self.network, self.class_names, self.class_colors = darknet.load_network(
            utils.join_strings_as_path([darknet_path, config_file]),
            utils.join_strings_as_path([darknet_path, data_file]),
            utils.join_strings_as_path([darknet_path, weights]),
            batch_size=1
        )

        self.search_box = DARKNET['search_box']
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)
        self.thresh = thresh
        self.Q = object_detection_queue

        self.is_night_vision = None
        self.org_frame = None
        self.resize_frame = None
        self.dirs = utils.DirsHandler(DIRS)
        self.small_imgs = []
        self.first_result = None
        self.init_done = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = False
        self.thread.start()
        return self

    def is_alive(self):
        if self.thread:
            return self.thread.is_alive()
        else:
            return False

    def draw_boxes(self, detections, sorted_idxs):
        cnt = 0
        tmp_img = self.org_frame.copy()
        for label, confidence, bbox in detections:
            left, top, right, bottom = darknet.bbox2points(bbox)
            org_left, org_top, org_right, org_bottom = utils.scale_bbox([left, top, right, bottom],
                                                                        self.width, self.height,
                                                                        self.org_frame.shape[1],
                                                                        self.org_frame.shape[0])

            small_image_data = {
                'img': tmp_img[org_top:org_bottom, org_left:org_right],
                'label': label,
                'confidence': confidence,
                'width': org_right - org_left,
                'height': org_bottom - org_top,
                'new_data': True
            }

            if sorted_idxs[cnt] is None or sorted_idxs[cnt] >= len(self.small_imgs):
                small_image_data['fixed_size'] = [2*small_image_data['height'], 2*small_image_data['width']]
                small_image_data['final_labels'] = set([small_image_data['label']])
                small_image_data['video_name'] = None
                self.small_imgs.append(small_image_data)
            else:
                small_image_data['fixed_size'] = self.small_imgs[sorted_idxs[cnt]]['fixed_size']
                small_image_data['final_labels'] = self.small_imgs[sorted_idxs[cnt]]['final_labels']
                small_image_data['final_labels'].add(small_image_data['label'])
                small_image_data['video_name'] = self.small_imgs[sorted_idxs[cnt]]['video_name']
                self.small_imgs[sorted_idxs[cnt]] = small_image_data

            cnt += 1

            cv2.rectangle(self.org_frame,
                          (org_left, org_top),
                          (org_right, org_bottom),
                          self.class_colors[label], 3)
            cv2.putText(self.org_frame, "{} [{:.2f}]".format(label, float(confidence)),
                        (org_left, org_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        self.class_colors[label], 2)


    def video_capture(self, cap):
        while cap.isOpened():
            ret, frame = cap.read()
            # vid_width = frame.shape[1]
            # vid_height = frame.shape[0]
            # box = DARKNET['search_box'][-1]
            # w1 = int(vid_width * box[0])
            # h1 = int(vid_height * box[2])
            # w2 = int(vid_width * box[1])
            # h2 = int(vid_height * box[3])
            # frame = frame[h1:h2, w1:w2, :]
            if not ret:
                break
            if self.is_night_vision is None:
                if utils.is_night_vision(frame):
                    self.is_night_vision = True
                else:
                    self.is_night_vision = False
            if self.is_night_vision:
                frame = cv2.blur(frame, DARKNET['night_vision_blur_kernel']) # img grainy - seems to make better results
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height),
                                       interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
            self.org_frame = frame
            self.resize_frame = frame_resized
            # cv2.imshow('test', frame)
            # if cv2.waitKey(1) == 27:
            #      return True
            return True
        cap.release()
        return False

    def get_relevant_detections(self, detections):
        rel_detections = []
        tmp_detections = \
            [detection for detection in detections if detection[0] in DARKNET['keep_categories']]
        tmp_detections2 = \
            [detection for detection in tmp_detections
             if utils.is_in_bb(detection[2][0:2], self.resize_frame.shape, box_wh=self.search_box) >= 0]
        for detection in tmp_detections2:
            skip_detection = False
            for known_fa in DARKNET['known_false_alarms']:
                score = np.abs(np.array(detection[2]) - known_fa['MEAN']) / known_fa['STD']
                if np.mean(score) < DARKNET['known_false_alarms_thresh']:
                    skip_detection = True
                    break
            if not skip_detection:
                rel_detections.append(detection)
        ### merge similar detections
        remove_idxs = []
        for iii, detection1 in enumerate(rel_detections):
            for jjj, detection2 in enumerate(rel_detections):
                if iii <= jjj:
                    continue
                score = np.abs(np.array(detection1[2]) - np.array(detection2[2]))
                #print(iii,jjj,score,np.mean(score))
                if np.mean(score) < DARKNET['merge_thresh']:
                    remove_idxs.append(iii)
        for index in sorted(np.unique(remove_idxs), reverse=True):
            try:
                del rel_detections[index]
            except:
                LOGGER.error('DARKNET: Problem!')
                LOGGER.error(str(traceback.format_exc()))

        return rel_detections

    def check_for_continuity(self, dets):
        if len(dets) > 0:
            for det in dets:
                if self.first_result is None:
                    self.first_result = det
                    return True
                else:
                    COM_1 = np.array([int(self.first_result[2][0]), int(self.first_result[2][1])])
                    COM_2 = np.array([int(det[2][0]), int(det[2][1])])
                    diag = np.sqrt(self.org_frame.shape[0]**2 + self.org_frame.shape[1]**2)
                    print(np.linalg.norm(COM_1 - COM_2))
                    if np.linalg.norm(COM_1 - COM_2) > diag * DARKNET['min_pct_from_first_res']:
                        return True
        return False


    def inference(self):
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=self.thresh)
        return detections

    def handle_videos(self, VLH, vid_path, vid_sim_fps):
        try:
            VLH.add_frame('main', self.org_frame)
            for iii, img_data in enumerate(self.small_imgs):
                name = 'small_{}'.format(str(iii))
                if not VLH.is_exist(name):
                    full_path_small, vid_name = self.get_full_video_path(vid_path, '_' + name)
                    VLH.add_video(name, full_path_small,
                                  img_data['fixed_size'][0],
                                  img_data['fixed_size'][1],
                                  debug=True,
                                  vid_fps=vid_sim_fps)
                    img_data['video_name'] = vid_name
                if img_data['new_data']:
                    img = utils.resize_keep_ratio(img_data['img'], img_data['fixed_size'])
                    VLH.add_frame(name, img)
                    img_data['new_data'] = False
        except:
            LOGGER.error('DARKNET: Videos Problem!')
            LOGGER.error(str(traceback.format_exc()))
        return VLH


    def get_full_video_path(self, vid_path, ext):
        vid_name = os.path.basename(vid_path).split('.')[0] + ext +'.mkv'
        return utils.join_strings_as_path(
            [self.dirs.all_dirs['final_detection'], vid_name]), vid_name

    def update(self):
        self.stopped = False
        while not self.init_done:
            time.sleep(1)
        LOGGER.info('DarkNet initialized and listening...')
        # keep looping infinitely
        while True:
            if self.Q.qsize() > 0:
                vid_path = self.read()
                LOGGER.info('DARKNET: Start working on {}'.format(vid_path))
                self.small_imgs = []
                try:
                    cap = cv2.VideoCapture(vid_path)
                    #cap.set(cv2.CAP_PROP_POS_FRAMES, 3750)
                    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    vid_sim_fps = int(cap.get(cv2.CAP_PROP_FPS))
                    full_path, tmp = self.get_full_video_path(vid_path, '_NN')
                    VLH = VideosListHandler()
                    VLH.add_video('main', full_path, vid_height, vid_width, debug=True, vid_fps=vid_sim_fps)
                    found_counter = 0
                    debug_det = []
                    self.first_result = None
                    self.is_night_vision = None
                    frame_counter = 0
                    last_detection_arranged = None
                    while True:
                        if self.video_capture(cap):
                            # if self.debug:
                            #       self.org_frame = self.resize_frame
                            #       vid_width = self.width
                            #       vid_height = self.height
                            #       self.org_frame = cv2.cvtColor(self.org_frame, cv2.COLOR_BGR2RGB)
                            frame_counter += 1
                            detections = self.inference()
                            rel_detections = self.get_relevant_detections(detections)
                            last_detection_arranged, sorted_idxs = utils.association(last_detection_arranged,
                                                                                     rel_detections.copy(),
                                                                                     frame_counter)
                            if self.resize_frame is not None and \
                                    (found_counter > 0 or
                                     len(rel_detections) > 0):

                                if self.check_for_continuity(rel_detections):
                                    found_counter += 1
                                self.draw_boxes(rel_detections, sorted_idxs)
                                VLH = self.handle_videos(VLH, vid_path, vid_sim_fps)

                                if self.debug:
                                    for iii, box in enumerate(self.search_box):
                                        w1 = int(vid_width * box[0])
                                        h1 = int(vid_height * box[2])
                                        w2 = int(vid_width * box[1])
                                        h2 = int(vid_height * box[3])
                                        cv2.rectangle(self.org_frame, (w1, h1), (w2, h2), (0, 0, 255), 2)
                                    # for det in rel_detections:
                                    #     COM = (int(det[2][0]), int(det[2][1]))
                                    #     cv2.circle(self.org_frame, COM, 1, (255,255,255), 1)

                                    for iii, img_data in enumerate(self.small_imgs):
                                        try:
                                            img = utils.resize_keep_ratio(img_data['img'], img_data['fixed_size'])
                                            cv2.imshow(str(iii), img)
                                        except:
                                            print('error')
                                    if cv2.waitKey(1) == 27:
                                        break
                                    if len(rel_detections) > 0:
                                        debug_det.append(rel_detections[0][2])
                                        #print(rel_detections)

                        else:
                            break
                    if found_counter >= DARKNET['min_detection_to_save_file']:
                        LOGGER.info('DARKNET: FOUND SOMETHING working on {}'.format(vid_path))
                        VLH.close_all_videos()
                        if not self.debug:
                            target_path = utils.copy_all_video_refrences(os.path.basename(vid_path).split('_mov')[0],
                                                           self.dirs, wait_ffmpeg=True)
                            for iii, img_data in enumerate(self.small_imgs):
                                final_path = utils.join_strings_as_path([target_path, img_data['video_name']])
                                date_time = img_data['video_name'].split('_mov')[0]
                                alert_queue.put({
                                    'type': 'video',
                                    'path': final_path,
                                    'labels': img_data['final_labels'],
                                    'date': str(datetime.datetime.strptime(date_time, '%Y%m%d_%H%M%S').date()),
                                    'time': str(datetime.datetime.strptime(date_time, '%Y%m%d_%H%M%S').time())
                                })
                        if self.debug:
                            print("MEAN:" + str(np.mean(np.array(debug_det), axis=0)))
                            print("STD:" + str(np.std(np.array(debug_det), axis=0)))
                    else:
                        LOGGER.info('DARKNET: FOUND NOTHING working on {}'.format(vid_path))
                        if not self.debug:
                            VLH.close_all_videos(move_to_path='delete')
                        else:
                            VLH.close_all_videos()
                    LOGGER.info('DARKNET: Done working on {}'.format(vid_path))
                except:
                    LOGGER.error('DARKNET: Problem!')
                    LOGGER.error(str(traceback.format_exc()))
                    continue
            else:
                time.sleep(5)

            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return


    def read(self, timeout=5):
        # return next frame in the queue
        return self.Q.get(timeout=timeout)

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class FalseAlarmClassifier:
    def __init__(self):
        self.detectionData = deque(maxlen=FALSE_ALARM['detect_data_len_max'])
        self.fa_cnt = 0

    def run(self, bb_final_list, img_shape):
        if self.fa_cnt >= FALSE_ALARM['min_fa_counter']:
            LOGGER.info('fa counter is --> {}'.format(str(self.fa_cnt)))
        #print(self.fa_cnt)
        if len(bb_final_list) == 0:
            self.detectionData.append((time.time(), 0, 0, 0, 0, 0, 0, [], -999))
            self.fa_cnt = max((0, int(self.fa_cnt*FALSE_ALARM['fa_counte_decay'])))
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

        self.detectionData.append([
                                  time.time(),
                                  100.0 * sumAllAreas / maxImageArea,
                                  100.0 * sumNear / (maxImageArea / 2),
                                  100.0 * sumFar / (maxImageArea / 2),
                                  100.0 * maxSingleArea / maxImageArea,
                                  100.0 * maxAllAreas / maxImageArea,
                                  len(bb_final_list),
                                  rel_bb,
                                  -999]) #<-- Final Tag (data[8])
        # print(self.detectionData[-1])

        # print(len(self.detectionData))

        #if len(rel_bb) == 0:
        #    self.fa_cnt = max((0, int(self.fa_cnt * FALSE_ALARM['fa_counte_decay'])))
        #    return 0

        if len(self.detectionData) > FALSE_ALARM['detect_data_len_min']:
            negCount = 0
            rel_bb_cnt = np.zeros(len(DIFF['box_wh']))
            FA_string = 'FA --> '
            for cnt, data in enumerate(self.detectionData):
                if data[1] is not 0:
                    tests = [
                        data[1] > FALSE_ALARM['sum_all_areas_max'],
                        data[2] > 0 and data[2] < FALSE_ALARM['sum_near_min'],
                        data[3] > FALSE_ALARM['sum_far_max'],
                        data[4] > FALSE_ALARM['max_single_area_max'],
                        data[5] > FALSE_ALARM['max_all_area_max'],
                        data[6] > FALSE_ALARM['len_bb_max']
                    ]
                    if np.any(tests):
                        idxs = np.where(tests)[0]
                        negCount = negCount + np.sum(tests)
                        FA_string += '({}), '.format(str(np.where(tests)[0]).replace('[', '').replace(']', '').replace(' ',','))
                        if self.detectionData[cnt][8] == -999:
                            self.detectionData[cnt][8] = 0

                    used = []
                    for relData in data[7]:
                        if sum([x == relData[0] for x in used]) > 0:
                            continue
                        else:
                            if self.detectionData[cnt][8] <= 0: # Do not use positives that where classified as FA
                                rel_bb_cnt[relData[0]] = rel_bb_cnt[relData[0]] + 1
                                used.append(relData[0])

                if negCount > FALSE_ALARM['neg_max']:
                    #cv2.putText(self.img_curr_2show, "False Alarm", (int(width * 0.05), int(height * 0.05)),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    LOGGER.info(FA_string.rstrip(','))
                    LOGGER.info("False Alarm")
                    self.fa_cnt += 1
                    for cnt_fa, data_fa in enumerate(self.detectionData):
                        if self.detectionData[cnt_fa][8] == 0:
                            self.detectionData[cnt_fa][8] = 1
                    return -1

            if negCount > 0:
                LOGGER.info(FA_string)

            if rel_bb_cnt[rel_bb_cnt > FALSE_ALARM['pos_max']].any() and self.fa_cnt < FALSE_ALARM['min_fa_counter']:
                #cv2.putText(self.img_curr_2show, "Positive Detection", (int(width * 0.05), int(height * 0.05)),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                LOGGER.info("Positive Detection")
                self.fa_cnt = max((0, int(self.fa_cnt*FALSE_ALARM['fa_counte_decay'])))
                return 1

                # LOGGER.info("UNKNOWN")
        self.fa_cnt = max((0, int(self.fa_cnt*FALSE_ALARM['fa_counte_decay'])))
        return 0
        # print(self.detectionData[-1] )


if __name__=='__main__':
    DNC = DarkNetClassifier(debug=True).start()
    #object_detection_queue.put('../recordings/final_detection/run_1week_oct2020/20201001_092305_3detections/20201001_092305_mov.mkv')
    object_detection_queue.put('../recordings/detection/before_NN/20201023_162532_mov.mkv')
    #DNC.Q.put('../recordings/detection/before_NN/20201016_104643_mov.mkv')
    #DNC.Q.put('../recordings/final_detection/run_1week_oct2020/20201001_092305_3detections/20201001_092305_mov.mkv')
    #DNC.Q.put('../recordings/final_detection/Check/20201006_154010_mov.mp4')
    #DNC.Q.put('../recordings/final_detection/20200930_181913/20200930_181913_mov.mkv')
    #DNC.Q.put('../recordings/detection/before_NN/20200930_103412_mov.mkv')
    #DNC.Q.put('../sim/person/20200916_080831_mov.mkv')

    while True:
        time.sleep(10000)
