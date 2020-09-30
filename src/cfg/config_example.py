import numpy as np

IPCAM_CONFIG = {
'rtspurl': 'rtsp://UN:PW@IP:PORT/CAM_DIR/',
'sim_path': '../sim/20200911_181738_main.mp4'
}


STREAM_TYPE = 'rtsp'
#STREAM_TYPE = 'sim'

if STREAM_TYPE == 'sim':
    SIM_START_FRAME = 3500
    VID_FPS = 13
    MAX_SINGLE_VIDEO_TIME = 300
    FH_ENABLE = False
else:
    SIM_START_FRAME = 0
    VID_FPS = 13 # estimated from camera
    MAX_SINGLE_VIDEO_TIME = 300
    FH_ENABLE = True
## general params
SHOW_STREAM = False # Show video and diffs


## dirs
DIRS = {
        'level': 0,
        'main_dir': '../',
        'dir_name': 'recordings',
        'diff_detection': {
            'level': 1,
            'dir_name': 'detection',
            'compressed': 'compressed',
            'ready_for_NN': 'before_NN'
        },
        'no_diff_detection': {
            'level': 1,
            'dir_name': 'no_detection',
            'compressed': 'compressed'
        },
        'final_detection': {
            'level': 1,
            'dir_name': 'final_detection',
        }
}
REC_DIR_COMPRESSED = DIRS['diff_detection']['compressed']
REC_DIR_COMPRESSED_FOR_NN = DIRS['diff_detection']['ready_for_NN']
if STREAM_TYPE == 'sim':
    DIRS['dir_name'] += '_sim'


##diff params
DIFF = {
    'box_wh': np.array([[0.05, 0.05+0.4, 0.4, 0.4+0.55],
                             [0.5, 0.5+0.15, 0.2, 0.2+0.3],
                             [0.675, 0.675+0.125, 0.1, 0.1+0.175],
                             [0.8, 0.8+0.18, 0.45, 0.45+0.5]]),
    'min_bb_area': (150, 100, 50, 150),
    'resize_org': 2
}

##FFMPEG params
FFMPEG_PARAMS = {
    'command': ['ffmpeg', '-i', 'fileName', '-c:v', 'libx264', '-tune', 'film', '-preset', 'fast',
                 '-profile:v', 'high444', '-crf', '38', 'fileNameMP4', '-c:a', 'copy', '-loglevel', 'error', '-y']
    # In windows instead of just 'ffmpeg' - write full path C:/...
}

##FALSE ALARM params
FALSE_ALARM = {
    'detect_data_len_max': 40,
    'detect_data_len_min': 26,
    'sum_all_areas_max': 25,
    'sum_near_min': 0.1,
    'sum_far_max': 40,
    'max_single_area_max': 25,
    'max_all_area_max': 90,
    'len_bb_max': 25,
    'neg_max': 10,
    'pos_max': 5,
    'min_fa_counter': 15,
    'fa_counte_decay': 0.925
}

FILES = {
    'max_history_detected': 2*24*3600, #2*24 Hours
    'max_history_not_detected': 24*3600 #24 Hours
}

MOV_FRAME_HANDLE = {
    'frames_to_expand': VID_FPS*4,
    'min_gap_to_stop_capture': VID_FPS*10  # this must be larger than 'frames_to_expand'
}

DARKNET = {
    'keep_categories': ['person', 'dog', 'cat'],
    'search_box': np.array([[0.05, 0.05+0.93, 0.4, 0.4+0.55],
                             [0.5, 0.5+0.3, 0.2, 0.2+0.22],
                             [0.675, 0.675+0.125, 0.1, 0.1+0.175]]),
    'known_false_alarms': [
        {'MEAN': [276.61611357, 74.80160704, 15.15296069, 55.08387102],
         'STD':[0.41327334, 1.61500166, 0.96064496, 2.96354952]},
        {'MEAN': [275.264134, 65.36172485, 10.54579503, 33.71312032],
         'STD': [0.20919952, 1.9582495, 0.4635283, 2.36697068]}
    ],
    'known_false_alarms_thresh': 2,
    'min_detection_to_save_file': 2,
    'min_pct_from_first_res': 0.00454  # 0.454% from 2202 (1920X1080) is 10 pixels
}

GENERAL = {
    'queue_warning_length': 50
}
