from threading import Thread
from queue import Queue
import time
from cfg import *
from utils.CaptureHandler import CaptureHandler
from image_processing.DiffHandler import DiffHandler
from image_processing.ClassifiersHandler import FalseAlarmClassifier
from utils.VideoHandler import VideosListHandler
from utils import utils
from collections import deque


class ImageHandler:

    def __init__(self, debug=False):
        self.debug = debug
        self.debug_queue = Queue(maxsize=10)
        self.time_for_fps = None
        self.fps = None
        self.fps_counter = 0
        self.img_orig = None
        self.img_curr = None
        self.img_curr_debug = None
        self.bb_list = []
        self.fac_res = 0
        self.VLH = VideosListHandler()
        self.video_time = 0.0
        self.video_time_string = 0.0 #utils.time2string(time.time())
        self.found_movemonet = False
        self.mov_frames = []
        self.frames_counter = 0
        self.dirs = utils.DirsHandler(DIRS)
        self.histograms = []
        self.last_frames = deque(maxlen=MOV_FRAME_HANDLE['frames_to_expand'])
        self.last_fac_res = 0.0

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = False
        t.start()
        return self

    def count_fps(self):
        if self.time_for_fps is None:
            self.time_for_fps = time.time()
            return
        if time.time() - self.time_for_fps > 1:
            self.time_for_fps = time.time()
            self.fps = self.fps_counter
            self.fps_counter = 0


    def update_image(self, frame):
        ## resize frame with RESIZE_ORG (no interp) --> blur(2,2)
        self.img_orig = frame
        self.img_curr = cv2.blur(frame[0::RESIZE_ORG, 0::RESIZE_ORG, 0:], (2, 2))

    def update_image_debug(self):
        self.img_curr_debug = self.img_curr.copy()
        height, width, layers = self.img_curr_debug.shape
        self.histograms = []
        for iii, box in enumerate(BOX_WIDTH_HEIGHT):
            w1 = int(width * box[0])
            h1 = int(height * box[2])
            w2 = int(width * box[1])
            h2 = int(height * box[3])
            cv2.rectangle(self.img_curr_debug, (w1, h1), (w2, h2), (0, 0, 255), 2)
            cv2.putText(self.img_curr_debug, str(iii), (w1, h1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.histograms.append(cv2.calcHist([self.img_curr_debug[h1:h2, w1:w2]], [0], None, [256], [0, 256]))

        for i, bbi in enumerate(self.bb_list):
            x = bbi[0]
            y = bbi[1]
            x1 = bbi[0] + bbi[2]
            y1 = bbi[1] + bbi[3]
            cv2.rectangle(self.img_curr_debug, (x, y), (x1, y1), (0, 255, 0), 1)

        if self.fac_res == 1:
            cv2.putText(self.img_curr_debug, "Positive Detection", (int(width * 0.05), int(height * 0.05)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.fac_res == -1:
            cv2.putText(self.img_curr_debug, "False Alarm", (int(width * 0.05), int(height * 0.05)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.count_fps()
        if isinstance(self.fps, int):
            cv2.putText(self.img_curr_debug, str(self.fps), (int(width * 0.01), int(height * 0.04)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        #self.img_curr_debug = self.img_curr_debug[h1:h2, w1:w2]

    #def add_mov_info(self):
    #    if len(self.mov_frames) == 0 or self.frames_counter - self.mov_frames[-1][-1] > MOV_FRAME_HANDLE['min_gap_to_stop_capture']:
    #        self.mov_frames.append([self.frames_counter, self.frames_counter])
    #    if self.fac_res == 1:
    #        self.mov_frames[-1][-1] = self.frames_counter

    def init_stream(self):
        LOGGER.info('Init Stream')
        if STREAM_TYPE == 'rtsp':
            CH = CaptureHandler(IPCAM_CONFIG['rtspurl']).start()
        elif STREAM_TYPE == 'sim':
            CH = CaptureHandler(IPCAM_CONFIG['sim_path'],
                                stream_type=STREAM_TYPE, sim_start_frame=SIM_START_FRAME).start()
        time.sleep(0.5)
        return CH


    def handle_videos(self, sim_end=False):
        if time.time() - self.video_time < MAX_SINGLE_VIDEO_TIME and not sim_end:
            return

        if self.video_time > 0.0:
            if self.found_movemonet:
                LOGGER.info('Moving files to detected')
                self.VLH.close_all_videos(self.dirs.all_dirs['diff_detection'])
                #utils.save_mov_info(self.mov_frames,
                #                    self.dirs.all_dirs['diff_detection'],
                #                    self.video_time_string)
            else:
                LOGGER.info('Moving files to not_detected')
                self.VLH.close_video('mov', move_to_path='delete', remove_key=True)
                self.VLH.close_all_videos(self.dirs.all_dirs['no_diff_detection'])


        if sim_end:
            return

        self.found_movemonet = False
        self.frames_counter = 0
        self.video_time = time.time()
        self.video_time_string = utils.time2string(self.video_time)
        self.VLH.add_video('main', '{}/{}_main.mkv'.format(self.dirs.all_dirs['main'], self.video_time_string),
                           self.img_orig.shape[0], self.img_orig.shape[1], only_if_not_exist=True)
        self.VLH.add_video('mov', '{}/{}_mov.mkv'.format(self.dirs.all_dirs['main'], self.video_time_string),
                           self.img_orig.shape[0], self.img_orig.shape[1], only_if_not_exist=True)
        if self.debug:
            self.VLH.add_video('debug', '{}/{}_debug.mkv'.format(self.dirs.all_dirs['main'], self.video_time_string),
                               self.img_curr_debug.shape[0], self.img_curr_debug.shape[1], only_if_not_exist=True)


    def add_frame_to_video(self):
        self.VLH.add_frame('main', self.img_orig)
        self.last_frames.append(self.img_orig)
        if self.fac_res == 1 or (time.time() - self.last_fac_res) < MOV_FRAME_HANDLE['frames_to_expand']/VID_FPS:
            self.found_movemonet = True
            for frame in self.last_frames:
                self.VLH.add_frame('mov', frame)
            self.last_frames = deque(maxlen=MOV_FRAME_HANDLE['frames_to_expand'])
            if self.fac_res == 1:
                self.last_fac_res = time.time()
        if self.debug:
            self.VLH.add_frame('debug', self.img_curr_debug)

    def queues_handling(self):
        if self.debug and self.debug_queue.qsize() < self.debug_queue.maxsize:
            self.debug_queue.put(self.img_curr_debug)
            if self.debug_queue.qsize() > QUEUE_WARNING_LENGTH:
                LOGGER.warn('DEBUGIMAGE: Frame Queue size is {}'.format(str(self.debug_queue.qsize())))

    def update(self):
        CH = self.init_stream()
        DH = DiffHandler()
        FAC = FalseAlarmClassifier()

        frame_error_counter = 0
        # keep looping infinitely
        while True:
            if CH.is_alive():
                try:
                    frame = CH.read(timeout=2)
                    frame_error_counter = 0
                    self.fps_counter += 1
                    self.frames_counter += 1
                except:
                    LOGGER.warn('IMAGEHANDLER: didnt get frame for 2 seconds')
                    frame_error_counter += 1
                    if frame_error_counter > 4:
                        LOGGER.error('IMAGEHANDLER: Stream Issue: Init Stream')
                        CH = self.init_stream()
                    continue
                if CH.Q.qsize() > CH.Q.maxsize/2:
                    LOGGER.error('IMAGEHANDLER: Queue Capture Issue: Init Stream')
                    CH = self.init_stream()
                    continue
                self.update_image(frame)
                self.bb_list = DH.run(self.img_curr)
                self.fac_res = FAC.run(self.bb_list, self.img_curr.shape)
                #if self.fac_res == 1:
                #    self.found_movemonet = True
                #if self.found_movemonet:
                    #self.add_mov_info()
                if self.debug:
                    self.update_image_debug()
                self.handle_videos()
                self.add_frame_to_video()

            else:
                if STREAM_TYPE == 'sim':
                    LOGGER.info('Simulation done!')
                    self.handle_videos(sim_end=True)
                    return
                else:
                    LOGGER.error('Thread is dead baby.. Restarting')
                    CH = self.init_stream()

            self.queues_handling()

class ImageShow:
    def __init__(self, IH, debug=False):
        self.debug = debug
        self.IH = IH
        self.img_curr_debug = np.zeros((100, 100, 3))
        pass

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = False
        t.start()
        return self

    def update(self):
        #plt.ion()
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #line1, = ax.plot(np.linspace(0, 256, 256), np.linspace(0, 3500, 256), 'r-')
        while True:
            try:
                self.img_curr_debug = self.IH.debug_queue.get(timeout=1)
            except:
                pass
            cv2.imshow('cam', self.img_curr_debug)

            #if len(self.IH.histograms) > 3:
            #    line1.set_ydata(self.IH.histograms[0])
            #    fig.canvas.draw()
            #    fig.canvas.flush_events()
            if cv2.waitKey(1) == 999:
                exit(0)


if __name__ == '__main__':
    IH = ImageHandler(debug=True).start()
    IS = ImageShow(IH, debug=True).start()
    while True:
        time.sleep(10000)