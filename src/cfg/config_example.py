import numpy as np
import cv2

import logging.config
import os.path
LOGGING_CONF = os.path.join(os.path.dirname(__file__), "logging.ini")
logging.config.fileConfig(LOGGING_CONF)
import logging
LOGGER = logging.getLogger('main')

IPCAM_CONFIG = {
'rtspurl': 'rtsp://UN:PW@IP:PORT/CAM_DIR/',
'sim_path': '../sim/20200911_181738_main.mp4'
}

STREAM_TYPE = 'rtsp'
#STREAM_TYPE = 'sim'
SIM_START_FRAME = 2250
VID_FPS = 13

## general params
CALC_DIFFS = True
DEBUG = True # Show video and diffs
SAVE_DEBUG = True
TIME_RATIO = 1000.0
MAX_SINGLE_VIDEO_TIME = 300


## dirs
MAIN_DIR = '../Recordings'
REC_DIR = '/Detection'
REC_DIR_COMPRESSED = '/Compressed'
REC_DIR_COMPRESSED_WITH_NN = '/After_NN'
REC_DIR_COMPRESSED_FOR_NN = '/Before_NN'
REC_DIR_2del = '/noDetection'
if STREAM_TYPE == 'sim':
    MAIN_DIR += '_sim'

##diff params
BOX_WIDTH_HEIGHT = np.array([[0.05, 0.05+0.4, 0.4, 0.4+0.55],
                             [0.5, 0.5+0.15, 0.2, 0.2+0.3],
                             [0.675, 0.675+0.125, 0.1, 0.1+0.175],
                             [0.8, 0.8+0.18, 0.45, 0.45+0.5]])
minBBArea = (150, 100, 50, 150)
#BOX_WIDTH_HEIGHT =  np.array([[0.05,0.05+0.3,0.05,0.05+0.3]])
BOX_WIDTH_HEIGHT_REF = np.array([0.5, 0.5+0.2, 0.6, 0.6+0.3])
NUM_OF_BOXES = len(BOX_WIDTH_HEIGHT)
RESIZE_ORG = 2
REF_WINDOW_LEN = 2
REF_WINDOW_LEN_LOG = np.log2(REF_WINDOW_LEN).astype('uint8') ## for fast div/mul

FFMPEG_LOC = 'ffmpeg' # In windows it will be full path
FFMPEG_COMMAND = [FFMPEG_LOC, '-i', 'fileName', '-c:v', 'libx264', '-tune', 'film', '-preset', 'fast',
                   '-profile:v', 'high444', '-crf', '38', 'fileNameMP4', '-c:a', 'copy', '-loglevel', 'error', '-y']
TIME2DELETE_SEC = 2*86400 #24 Hours
TIME2ACTIVATE_DELETE_SEC = MAX_SINGLE_VIDEO_TIME + 60
font = cv2.FONT_HERSHEY_SIMPLEX

DETECT_DATA_LEN_MAX = 40
DETECT_DATA_LEN_MIN = 26
SUMALLAREAS_MAX = 25
SUMNEAR_MIN = 0.1
SUMFAR_MAX = 40
MAXSINGLEAREA_MAX = 25
MAXALLAREA_MAX = 90
LEN_BB_MAX = 25
NEG_MAX = 10
POS_MAX = 10
POS_DETECT_MIN_BUFFER = 25

