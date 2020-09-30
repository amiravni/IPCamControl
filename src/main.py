import time
from ImageHandler import ImageHandler, ImageShow
from FilesHandler import FilesHandlerRT, FilesHandlerNonRT
from cfg import *
import os
from utils import utils

if __name__ == '__main__':
    DH = utils.DirsHandler(DIRS)
    DH.exec_func(utils.make_dir_if_not_exist)
    if FH_ENABLE:
        FH = []
        FHNRT = []
        DH = utils.DirsHandler(DIRS)
        FH.append(FilesHandlerRT(DH.all_dirs['diff_detection'], substring='_mov', delete_org=False).start())
        FH.append(FilesHandlerRT(DH.all_dirs['diff_detection'], substring='_main').start())
        FH.append(FilesHandlerRT(DH.all_dirs['diff_detection'], substring='_debug').start())
        FH.append(FilesHandlerRT(DH.all_dirs['no_diff_detection'], substring='_main').start())
        FH.append(FilesHandlerRT(DH.all_dirs['no_diff_detection'], substring='_debug').start())
        FHNRT.append(FilesHandlerNonRT(dir_key='diff_detection',
                                       max_history=FILES['max_history_detected']).start())
        FHNRT.append(FilesHandlerNonRT(dir_key='no_diff_detection',
                                       max_history=FILES['max_history_not_detected']).start())

    IH = ImageHandler(debug=True).start()
    if SHOW_STREAM:
        IS = ImageShow(IH, debug=True).start()
    LOGGER.info('AMIR IPCAM VERSION {} IS READY'.format(VERSION))
    while True:
        time.sleep(10000)