import time
from ImageHandler import ImageHandler, ImageShow
from FilesHandler import FilesHandler
from cfg import *
import os
from utils import utils


def make_dirs():
    def make_dir_if_not_exist(_dir):
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    DH = utils.DirsHandler(DIRS)
    DH.exec_func(make_dir_if_not_exist)

if __name__ == '__main__':
    make_dirs()
    if FH_ENABLE:
        FH = []
        DH = utils.DirsHandler(DIRS)
        FH.append(FilesHandler(DH.all_dirs['diff_detection'], substring='_mov', delete_org=False, debug=True).start())
        FH.append(FilesHandler(DH.all_dirs['diff_detection'], substring='_main', debug=True).start())
        FH.append(FilesHandler(DH.all_dirs['diff_detection'], substring='_debug', debug=True).start())
        FH.append(FilesHandler(DH.all_dirs['no_diff_detection'], substring='_main', debug=True).start()) #TODO: Disable in the future
        FH.append(FilesHandler(DH.all_dirs['no_diff_detection'], substring='_debug', debug=True).start()) #TODO: Disable in the futur

    IH = ImageHandler(debug=True).start()
    if SHOW_STREAM:
        IS = ImageShow(IH, debug=True).start()
    LOGGER.info('AMIR IPCAM VERSION {} IS READY'.format(VERSION))
    while True:
        time.sleep(10000)