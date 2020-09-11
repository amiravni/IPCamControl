import shutil
import time
from config import *
from threading import Thread
import os
import subprocess


class FilesHandler():

    def __init__(self, basepath, substring='.', delete_org=True, debug=False):

        self.basepath = basepath
        self.substring = substring
        self.delete_org = delete_org
        self.debug = debug
        self.thread_cancelled = False
        self.ffmpeg_command = FFMPEG_COMMAND.copy()
        LOGGER.info("FileHandler initialized at {} with substring {}".format(self.basepath, self.substring))

    def start(self):
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
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

    def update(self):

        command = self.ffmpeg_command
        basepath = self.basepath
        lastDeleted = 0.0
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
                command[2] = self.basepath + '/' + fname
                command[13] = self.basepath + REC_DIR_COMPRESSED + '/' + output
                LOGGER.info("START ENCODING: {} --> {}".format(fname, output))
                try:
                    if os.stat(command[2]).st_size > 50000: #at least 50000 Bytes
                        subprocess.call(command)
                    else:
                        LOGGER.info("FILE {} is almost empty".format(fname))
                    LOGGER.info("DONE ENCODING: {} --> {}".format(fname, output))
                    if self.delete_org:
                        os.remove(command[2])
                    else:
                        shutil.move(command[2], basepath + REC_DIR_COMPRESSED_FOR_NN + '/' + fname)

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
                self.Print("START ENCODING: " + filename_local[len(MAIN_DIR):] + " --> " + output[len(MAIN_DIR):])
                try:
                    subprocess.call(command)
                    self.Print("DONE ENCODING: " + filename_local[len(MAIN_DIR):] + " --> " + output[len(MAIN_DIR):])
                    os.remove(filename_local)
                except:
                    self.Print("ERROR ENCODING: " + filename_local[len(MAIN_DIR):] + " --> " + output[len(MAIN_DIR):])
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
                    self.DeleteOldFiles(MAIN_DIR)
                    self.DeleteOldFiles(MAIN_DIR + REC_DIR)
                    self.DeleteOldFiles(MAIN_DIR + REC_DIR_2del)

