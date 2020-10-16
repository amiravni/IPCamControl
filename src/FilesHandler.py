import shutil
import time
from cfg import *
from threading import Thread
import os
import subprocess
from utils import utils
from image_processing.ClassifiersHandler import DarkNetClassifier
import traceback
from multiprocessing import Process, Queue, current_process


class FilesHandlerRT():
    def __init__(self, basepath, run_compression=[], run_NN=[], delete_org=[], debug=False):
        self.basepath = basepath
        self.delete_org = delete_org
        self.run_compression = run_compression
        self.run_NN = run_NN
        self.debug = debug
        self.thread_cancelled = False
        command_param = 'command_' + FFMPEG_PARAMS['use']
        self.ffmpeg_command = FFMPEG_PARAMS[command_param].copy()
        self.dirs = utils.DirsHandler(DIRS)
        LOGGER.info("FileHandler initialized at {} \n"
                    "Will compress -> {} \n"
                    "Will run object detection -> {}\n"
                    "Will delete original files -> {}".format(self.basepath,
                                                              str(self.run_compression),
                                                              str(self.run_NN),
                                                              str(self.delete_org)))

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

    def file_process(self, basepath, fname, command, dnc_proc_q, compress_file, run_nn, delete_file):
        try:
            input_full_path = utils.join_strings_as_path([basepath, fname])
            tmp_full_path = utils.join_strings_as_path([basepath, 'tmp', fname])
            shutil.move(input_full_path, tmp_full_path)

            if compress_file:
                output = '{}.mp4'.format(''.join(fname.split('.mkv')[:-1]))
                output_full_path = utils.join_strings_as_path([basepath, REC_DIR_COMPRESSED, output])
                command[FFMPEG_PARAMS['input_index']] = tmp_full_path
                command[FFMPEG_PARAMS['output_index']] = output_full_path
                LOGGER.info("START ENCODING: {} --> {}".format(fname, output))
                subprocess.call(command)
                LOGGER.info("DONE ENCODING: {} --> {}".format(fname, output))

            if run_nn:
                mov_path = utils.join_strings_as_path([self.basepath, REC_DIR_COMPRESSED_FOR_NN, fname])
                shutil.move(tmp_full_path, mov_path)
                dnc_proc_q.put(mov_path)
            elif delete_file:
                os.remove(tmp_full_path)
        except:
            LOGGER.error('FileHandlerRT: Process Problem!')
            LOGGER.error(str(traceback.format_exc()))

        return

    def update(self):
        utils.make_dir_if_not_exist(utils.join_strings_as_path([self.basepath, 'tmp']))
        dnc_proc_q = Queue(maxsize=128)
        if len(self.run_NN) > 0:
            DNC = DarkNetClassifier().start()
        else:
            DNC = []
        while not self.thread_cancelled:
            found_file = False
            for fname in os.listdir(self.basepath):
                input_full_path = utils.join_strings_as_path([self.basepath, fname])
                if os.path.isdir(input_full_path):
                    # skip directories
                    continue
                # try:
                compress_file = False
                run_nn = False
                delete_file = False
                time.sleep(1)  # finish moving file
                if os.stat(input_full_path).st_size > 50000: #at least 50000 Bytes
                    if utils.is_str_in_file(self.run_compression, fname):
                        compress_file = True
                    if utils.is_str_in_file(self.run_NN, fname):
                        run_nn = True
                else:
                    LOGGER.info("FILE {} is almost empty".format(fname))

                if utils.is_str_in_file(self.delete_org, fname):
                    delete_file = True

                if compress_file or run_nn or delete_file:
                    process = Process(name='ffmpeg_{}'.format('123'),
                                      target=self.file_process,
                                      args=(self.basepath, fname,
                                            self.ffmpeg_command, dnc_proc_q,
                                            compress_file, run_nn, delete_file))
                    process.daemon = False
                    process.start()
                    time.sleep(1)  # finish moving file
                    found_file = True

            while dnc_proc_q.qsize() > 0:
                filename = dnc_proc_q.get()
                LOGGER.info('NN: Put new file in queue -> {}'.format(filename))
                DNC.Q.put(filename)

            if not found_file:
                time.sleep(10)

# class FilesHandlerRT_():
#
#     def __init__(self, basepath, substring='.', make_mov_vid=False, delete_org=True, debug=False):
#
#         self.basepath = basepath
#         self.substring = substring
#         self.delete_org = delete_org
#         self.make_mov_vid = make_mov_vid
#         self.debug = debug
#         self.thread_cancelled = False
#         command_param = 'command_' + FFMPEG_PARAMS['use']
#         self.ffmpeg_command = FFMPEG_PARAMS[command_param].copy()
#         self.dirs = utils.DirsHandler(DIRS)
#         LOGGER.info("FileHandler initialized at {} with substring {}".format(self.basepath, self.substring))
#
#     def start(self):
#         self.thread = Thread(target=self.update)
#         self.thread.daemon = False
#         self.thread.start()
#         return self
#
#     def is_alive(self):
#         return self.thread.isAlive()
#
#     def shut_down(self):
#         self.thread_cancelled = True
#         # block while waiting for thread to terminate
#         while self.thread.isAlive():
#             time.sleep(1)
#         return True
#
#     def update(self):
#
#         command = self.ffmpeg_command
#         basepath = self.basepath
#         if self.substring == '_mov':
#             DNC = DarkNetClassifier().start()
#         while not self.thread_cancelled:
#             encoding = False
#             for fname in os.listdir(basepath):
#                 path = os.path.join(basepath, fname)
#                 if os.path.isdir(path) or self.substring not in fname:
#                     # skip directories
#                     continue
#                 encoding = True
#                 time.sleep(1) # finish moving file
#                 output = '{}.mp4'.format(''.join(fname.split('.mkv')[:-1]))
#                 input_full_path = utils.join_strings_as_path([basepath, fname])
#                 output_full_path = utils.join_strings_as_path([basepath, REC_DIR_COMPRESSED, output])
#                 command[FFMPEG_PARAMS['input_index']] = input_full_path
#                 command[FFMPEG_PARAMS['output_index']] = output_full_path
#                 LOGGER.info("START ENCODING: {} --> {}".format(fname, output))
#                 try:
#                     if os.stat(input_full_path).st_size > 50000: #at least 50000 Bytes
#                         subprocess.call(command)
#                         #if self.make_mov_vid:
#                         #    self.make_mov_vid_func(basepath, fname)
#                     else:
#                         LOGGER.info("FILE {} is almost empty".format(fname))
#                     LOGGER.info("DONE ENCODING: {} --> {}".format(basepath, output))
#                     if self.delete_org:
#                         os.remove(input_full_path)
#                     else:
#                         mov_path = utils.join_strings_as_path([basepath, REC_DIR_COMPRESSED_FOR_NN, fname])
#                         shutil.move(input_full_path, mov_path)
#                         if self.substring == '_mov':
#                             DNC.Q.put(mov_path)
#
#                 except:
#                     LOGGER.error("ERROR ENCODING: {} --> {}".format(fname, output))
#
#             if not encoding:
#                 time.sleep(10)

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