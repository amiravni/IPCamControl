import time
from ImageHandler import ImageHandler, ImageShow
from FilesHandler import FilesHandlerRT, FilesHandlerNonRT
from AlertsHandler import TelegramAlerts
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
        FH.append(FilesHandlerRT(DH.all_dirs['diff_detection'],
                                 run_compression=['_main', '_debug', 'mov'],
                                 run_NN=['_mov'],
                                 delete_org=['_main', '_debug', 'mov']
                                 ).start())
        FH.append(FilesHandlerRT(DH.all_dirs['no_diff_detection'],
                                 run_compression=['_main'],
                                 delete_org=['_main', '_debug']
                                 ).start())
        FHNRT.append(FilesHandlerNonRT(dir_key='diff_detection',
                                       max_history=FILES['max_history_detected']).start())
        FHNRT.append(FilesHandlerNonRT(dir_key='no_diff_detection',
                                       max_history=FILES['max_history_not_detected']).start())

    IH = ImageHandler(debug=True).start()
    if SHOW_STREAM:
        IS = ImageShow(IH, debug=True).start()

    TLGA = TelegramAlerts()

    LOGGER.info('AMIR IPCAM VERSION {} IS READY'.format(VERSION))
    while True:
        time.sleep(10000)