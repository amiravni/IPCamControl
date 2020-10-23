from cfg import *
import cv2
import shutil

class VideosListHandler:
    def __init__(self):
        self.video_dict = dict()

    def add_video(self, name, full_path, height, width, debug=False, only_if_not_exist=False, vid_fps=VID_FPS):
        if only_if_not_exist and name in self.video_dict:
            return
        else:
            self.video_dict[name] = VideoHandler(full_path, height, width, debug, vid_fps)

    def add_frame(self, name, frame):
        if name in self.video_dict:
            self.video_dict[name].add_frame(frame)
        else:
            LOGGER.error("video not initialized")

    def is_exist(self, name):
        if name in self.video_dict:
            return True
        else:
            return False

    def close_video(self, name, move_to_path='', remove_key=True):
        if name in self.video_dict:
            self.video_dict[name].close_and_move(move_to_path)
            if remove_key:
                self.video_dict.pop(name, None)
        else:
            LOGGER.error('VLH (close_video): No Video by that name')

    def close_all_videos(self, move_to_path=''):
        for key in self.video_dict:
            self.close_video(key, move_to_path=move_to_path, remove_key=False)
        self.video_dict = dict()


class VideoHandler:

    def __init__(self, full_path, height, width, debug=False, vid_fps=VID_FPS):
        self.debug = debug
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter(full_path, fourcc, vid_fps, (width, height))
        self.video_path = full_path
        self.video_name = full_path.split('/')[-1]

    def add_frame(self, frame):
        self.video.write(frame)


    def close_video(self):
        self.video.release()

    def close_and_move(self, move_to_path=''):
        self.close_video()
        if move_to_path == 'delete':
            os.remove(self.video_path)
            return
        if len(move_to_path) > 0:
            if move_to_path[-1] != '/':
                move_to_path = move_to_path + '/'
            shutil.move(self.video_path, move_to_path + self.video_name)


