import shutil
import time
from cfg import *
from threading import Thread
import os
import subprocess
from utils import utils
from image_processing.ClassifiersHandler import DarkNetClassifier
import traceback

class FilesHandlerRT():

    def __init__(self, basepath, substring='.', make_mov_vid=False, delete_org=True, debug=False):

        self.basepath = basepath
        self.substring = substring
        self.delete_org = delete_org
        self.make_mov_vid = make_mov_vid
        self.debug = debug
        self.thread_cancelled = False
        command_param = 'command_' + FFMPEG_PARAMS['use']
        self.ffmpeg_command = FFMPEG_PARAMS[command_param].copy()
        self.dirs = utils.DirsHandler(DIRS)
        LOGGER.info("FileHandler initialized at {} with substring {}".format(self.basepath, self.substring))

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.daemon = False
        self.thread.start()
        return self

    def is_alive(self):
        return self.thread.isAlive()

    def shut_down(self):
        self.thread_cancelled = True
        # block while waiting for thread to terminate
        while self.thread.isAlive():
            time.sleep(1)
        return True

    # def make_mov_vid_func(self, basepath, vid_name):
    #     if STREAM_TYPE == 'sim':
    #         start_frame = SIM_START_FRAME
    #     else:
    #         start_frame = 0
    #
    #     full_path = utils.join_strings_as_path([basepath, vid_name]).split('_main.mkv')[0]+'_mov.mkv'
    #
    #     CH = CaptureHandler(utils.join_strings_as_path([basepath, vid_name]),
    #                         stream_type='sim', sim_start_frame=SIM_START_FRAME, name='FH', fps=500)
    #     VH = VideoHandler(full_path, CH.height, CH.width, debug=True, vid_fps=CH.sim_fps )
    #     frames_file = utils.join_strings_as_path([basepath, vid_name.split('_main.mkv')[0]+'.txt'])
    #     with open(frames_file, 'r') as f:
    #         mov_frames = json.loads(f.read())
    #     CH.start()
    #     delay_time = time.time()
    #     frame_counter = SIM_START_FRAME
    #     while CH.is_alive() or time.time() - delay_time < 5.0: # 2sec to start stream
    #         if CH.is_alive():
    #             try:
    #                 frame = CH.read(timeout=2)
    #             except:
    #                 continue
    #             frame_counter += 1
    #             if any([frame_counter > mov_frame[0] and frame_counter < mov_frame[1]
    #                     for mov_frame in mov_frames]):
    #                 VH.add_frame(frame)
    #         else:
    #             time.sleep(1.0)
    #     VH.close_and_move(self.dirs.all_dirs['diff_detection_ready_for_NN'])
    #     try:
    #         shutil.move(frames_file,
    #                 utils.join_strings_as_path([basepath, REC_DIR_COMPRESSED]))
    #     except:
    #         print('didnt work')

    def update(self):

        command = self.ffmpeg_command
        basepath = self.basepath
        if self.substring == '_mov':
            DNC = DarkNetClassifier().start()
        while not self.thread_cancelled:
            encoding = False
            for fname in os.listdir(basepath):
                path = os.path.join(basepath, fname)
                if os.path.isdir(path) or self.substring not in fname:
                    # skip directories
                    continue
                encoding = True
                time.sleep(1) # finish moving file
                output = '{}.mp4'.format(''.join(fname.split('.mkv')[:-1]))
                input_full_path = utils.join_strings_as_path([basepath, fname])
                output_full_path = utils.join_strings_as_path([basepath, REC_DIR_COMPRESSED, output])
                command[FFMPEG_PARAMS['input_index']] = input_full_path
                command[FFMPEG_PARAMS['output_index']] = output_full_path
                LOGGER.info("START ENCODING: {} --> {}".format(fname, output))
                try:
                    if os.stat(input_full_path).st_size > 50000: #at least 50000 Bytes
                        subprocess.call(command)
                        #if self.make_mov_vid:
                        #    self.make_mov_vid_func(basepath, fname)
                    else:
                        LOGGER.info("FILE {} is almost empty".format(fname))
                    LOGGER.info("DONE ENCODING: {} --> {}".format(basepath, output))
                    if self.delete_org:
                        os.remove(input_full_path)
                    else:
                        mov_path = utils.join_strings_as_path([basepath, REC_DIR_COMPRESSED_FOR_NN, fname])
                        shutil.move(input_full_path, mov_path)
                        if self.substring == '_mov':
                            DNC.Q.put(mov_path)

                except:
                    LOGGER.error("ERROR ENCODING: {} --> {}".format(fname, output))

            if not encoding:
                time.sleep(10)

class FilesHandlerNonRT():
    def __init__(self, dir_key, max_history=FILES['max_history_detected'], debug=False):
        self.dirs = utils.DirsHandler(DIRS)
        self.dir_key = dir_key
        self.max_history = max_history
        self.main_dir = self.dirs.get_path_string(self.dir_key)
        self.debug = debug
        LOGGER.info("FileHandlerNonRT initialized with key {}".format(self.dir_key))

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.daemon = False
        self.thread_cancelled = False
        self.thread.start()
        return self

    def is_alive(self):
        return self.thread.isAlive()

    def shut_down(self):
        self.thread_cancelled = True
        # block while waiting for thread to terminate
        while self.thread.isAlive():
            time.sleep(1)
        return True

    def delete_old_files(self):
        for dirpath, dirnames, filenames in os.walk(self.main_dir):
            for file in filenames:
                curpath = os.path.join(dirpath, file)
                dt_sec = time.time() - os.path.getmtime(curpath)
                if dt_sec > self.max_history:
                    LOGGER.info("deleting: " + curpath)
                    os.remove(curpath)

    def update(self):
        while not self.thread_cancelled:
            try:
                self.delete_old_files()
            except:
                LOGGER.error('FileHandlerNonRT: Problem!')
                LOGGER.error(str(traceback.format_exc()))
            time.sleep(MAX_SINGLE_VIDEO_TIME + 10)




if __name__=='__main__':
    FHNRT = FilesHandlerNonRT(dir_key='diff_detection', max_history=86400, debug=True).start()
    while True:
        time.sleep(10000)