import shutil
import time
from cfg import *
from threading import Thread
import os
import subprocess
from utils import utils
from image_processing.ClassifiersHandler import DarkNetClassifier


class FilesHandler():

    def __init__(self, basepath, substring='.', make_mov_vid=False, delete_org=True, debug=False):

        self.basepath = basepath
        self.substring = substring
        self.delete_org = delete_org
        self.make_mov_vid = make_mov_vid
        self.debug = debug
        self.thread_cancelled = False
        self.ffmpeg_command = FFMPEG_COMMAND.copy()
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

    def DeleteOldFiles(self, dir_to_search):
        for dirpath, dirnames, filenames in os.walk(dir_to_search):
            for file in filenames:
                curpath = os.path.join(dirpath, file)
                deltaTimeSeconds = time.time() - os.path.getmtime(curpath)
                if deltaTimeSeconds > TIME2DELETE_SEC:
                    self.Print("deleting: " + curpath)
                    os.remove(curpath)


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
        lastDeleted = 0.0
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
                command[2] = utils.join_strings_as_path([basepath, fname])
                command[13] = utils.join_strings_as_path([basepath, REC_DIR_COMPRESSED, output])
                LOGGER.info("START ENCODING: {} --> {}".format(fname, output))
                try:
                    if os.stat(command[2]).st_size > 50000: #at least 50000 Bytes
                        subprocess.call(command)
                        #if self.make_mov_vid:
                        #    self.make_mov_vid_func(basepath, fname)
                    else:
                        LOGGER.info("FILE {} is almost empty".format(fname))
                    LOGGER.info("DONE ENCODING: {} --> {}".format(basepath, output))
                    if self.delete_org:
                        os.remove(command[2])
                    else:
                        mov_path = utils.join_strings_as_path([basepath, REC_DIR_COMPRESSED_FOR_NN, fname])
                        shutil.move(command[2], mov_path)
                        if self.substring == '_mov':
                            DNC.Q.put(mov_path)

                except:
                    LOGGER.error("ERROR ENCODING: {} --> {}".format(fname, output))

            if not encoding:
                time.sleep(10)


    def run(self):
        global files2Handle
        global threadsLock
        command = FFMPEG_COMMAND
        lastDeleted = 0
        cnt = 0
        while not self.thread_cancelled:
            threadsLock.acquire()
            try:
                if files2Handle.nLen > 0:
                    filename_local = files2Handle.name[0]
                    nLen_local = files2Handle.nLen
                else:
                    filename_local = ''
                    nLen_local = 0
            finally:
                threadsLock.release()
            if nLen_local > 0 and files2Handle.afterNN[
                0] is True:  ### <---- files2Handle.afterNN[0] is True: need to deal with list afterNN
                name = ''.join(filename_local.split('.mkv')[:-1])
                output = '{}.mp4'.format(name)
                command[2] = filename_local
                command[13] = output
                self.Print("START ENCODING: " +
                           filename_local[len(self.dirs.all_dirs['main']):] +
                           " --> " + output[len(self.dirs.all_dirs['main']):])
                try:
                    subprocess.call(command)
                    self.Print("DONE ENCODING: " +
                               filename_local[len(self.dirs.all_dirs['main']):] +
                               " --> " + output[len(self.dirs.all_dirs['main']):])
                    os.remove(filename_local)
                except:
                    self.Print("ERROR ENCODING: " +
                               filename_local[len(self.dirs.all_dirs['main']):] +
                               " --> " + output[len(self.dirs.all_dirs['main']):])
                threadsLock.acquire()
                try:
                    files2Handle.name.remove(files2Handle.name[0])
                    files2Handle.nLen = files2Handle.nLen - 1
                finally:
                    threadsLock.release()
            else:
                # print "Sleeping..."
                time.sleep(10)
                cnt += 1
                if cnt > 20:
                    print("FH --> checking for deleting files")
                    cnt = 0
                if time.time() - lastDeleted > TIME2ACTIVATE_DELETE_SEC:
                    self.DeleteOldFiles(self.dirs.all_dirs['main'])
                    self.DeleteOldFiles(self.dirs.all_dirs['diff_detection'])
                    self.DeleteOldFiles(self.dirs.all_dirs['no_diff_detection'])

