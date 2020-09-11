import time
from ImageHandler import ImageHandler, ImageShow
from FilesHandler import FilesHandler
from config import *
import os

def make_dirs():
    ALL_DIRS = [MAIN_DIR,
                MAIN_DIR + REC_DIR,
                MAIN_DIR + REC_DIR + REC_DIR_COMPRESSED,
                MAIN_DIR + REC_DIR + REC_DIR_COMPRESSED_WITH_NN,
                MAIN_DIR + REC_DIR + REC_DIR_COMPRESSED_FOR_NN,
                MAIN_DIR + REC_DIR_2del,
                MAIN_DIR + REC_DIR_2del + REC_DIR_COMPRESSED,
                MAIN_DIR + REC_DIR_2del + REC_DIR_COMPRESSED_WITH_NN,
                MAIN_DIR + REC_DIR_2del + REC_DIR_COMPRESSED_FOR_NN
                ]
    for _dir in ALL_DIRS:
        if not os.path.exists(_dir):
            os.makedirs(_dir)


if __name__ == '__main__':
    make_dirs()
    IH = ImageHandler(debug=True).start()
    if DEBUG:
        IS = ImageShow(IH, debug=True).start()
    video_types = ['_main', '_mov', '_debug']
    FH = []
    for v_type in video_types:
        if v_type == '_mov':
            FH.append(FilesHandler(MAIN_DIR + REC_DIR, substring=v_type, delete_org=False, debug=True).start())
        else:
            FH.append(FilesHandler(MAIN_DIR + REC_DIR, substring=v_type, debug=True).start())

        FH.append(FilesHandler(MAIN_DIR + REC_DIR_2del, substring=v_type, debug=True).start())
    while True:
        time.sleep(10000)