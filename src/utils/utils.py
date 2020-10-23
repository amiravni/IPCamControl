from cfg import *
import datetime
from pathlib import Path
from glob import glob
import json
import shutil
import time
from munkres import Munkres, print_matrix
import traceback
import cv2

def time2string(input_time):
    return datetime.datetime.fromtimestamp(input_time).strftime('%Y%m%d_%H%M%S')

def scale_bbox(bbox, bbox_im_w, bbox_im_h, org_w, org_h):
    # bbox - left, top, right, bottom
    x_scale = bbox_im_w / org_w
    y_scale = bbox_im_h / org_h
    org_left, org_top, org_right, org_bottom = \
        int(np.round(bbox[0] / x_scale)), \
        int(np.round(bbox[1] / y_scale)), \
        int(np.round(bbox[2] / x_scale)), \
        int(np.round(bbox[3] / y_scale))
    org_left = max(0, org_left)
    org_top = max(0, org_top)
    org_right = min(org_w, org_right)
    org_bottom = min(org_h, org_bottom)
    return [org_left, org_top, org_right, org_bottom]

def association(last_list, curr_list, frame_number, input_type='yolo'):
    # TODO: better score (with W,H),
    # TODO: above thresh --> append new
    #print(frame_number)
    if input_type == 'yolo':
        curr_list = [list(item[2]) for item in curr_list]
        [item.append(frame_number) for item in curr_list]
        # Xc,Yc,W,H,num
    if last_list is None or len(last_list) == 0:
        return curr_list, list(range(len(curr_list)))
    if len(curr_list) == 0:
        return last_list, []
    score_matrix = np.zeros((len(last_list), len(curr_list)))
    if len(last_list) > len(curr_list):
        flip_matrix = True
    else:
        flip_matrix = False
    for iii, list1 in enumerate(last_list):
        for jjj, list2 in enumerate(curr_list):
            if frame_number - list1[-1] > VID_FPS*2.5:
                score_matrix[iii, jjj] = 1e5
            else:
                score_matrix[iii, jjj] = np.linalg.norm(np.array(list1[:2]) - np.array(list2[:2]))

    if flip_matrix:
        score_matrix = score_matrix.T
    #print_matrix(score_matrix, msg='Highest profit through this matrix:')
    indexes = Munkres().compute(score_matrix)
    arranged_list = last_list.copy()
    remove_idxs_curr = []
    sorted_idxs = [None]*len(curr_list)
    for idxs in indexes:
        if flip_matrix:
            idxs = idxs[::-1]
        arranged_list[idxs[0]] = curr_list[idxs[1]]
        sorted_idxs[idxs[1]] = idxs[0]
        arranged_list[idxs[0]][-1] = frame_number
        remove_idxs_curr.append(idxs[1])
    if len(curr_list) > len(last_list):
        for index in sorted(remove_idxs_curr, reverse=True):
            del curr_list[index]
        for item in curr_list:
            arranged_list.append(item)

    return arranged_list, sorted_idxs

def is_night_vision(frame):
    test1 = np.mean(np.abs(frame[:, :, 2].astype('int32') - frame[:, :, 0].astype('int32')))
    test2 = np.mean(np.abs(frame[:, :, 1].astype('int32') - frame[:, :, 0].astype('int32')))
    test3 = np.mean(np.abs(frame[:, :, 1].astype('int32') - frame[:, :, 2].astype('int32')))
    return all(np.array([test1, test2, test3]) < IMAGE['is_night_vision_score_thresh'])


def is_intersect(veci, vecj, margin):
    xi0 = veci[0]
    yi0 = veci[1]
    xi1 = veci[0] + veci[2]
    yi1 = veci[1] + veci[3]
    xj0 = vecj[0] - margin
    yj0 = vecj[1] - margin
    xj1 = vecj[0] + vecj[2] + margin
    yj1 = vecj[1] + vecj[3] + margin
    if (xi0 >= xj0 and xi0 <= xj1 and yi0 >= yj0 and yi0 <= yj1) or \
            (xi1 >= xj0 and xi1 <= xj1 and yi0 >= yj0 and yi0 <= yj1) or \
            (xi0 >= xj0 and xi0 <= xj1 and yi1 >= yj0 and yi1 <= yj1) or \
            (xi1 >= xj0 and xi1 <= xj1 and yi1 >= yj0 and yi1 <= yj1):
        return True
    else:
        return False


def is_in_bb(center_of_mass, img_shape, bbox=None, box_wh=DIFF['box_wh']):
    height, width, layers = img_shape
    if bbox is not None:
        center_of_mass = (bbox[0]+int(bbox[2]/2), bbox[1]+int(bbox[3]/2))
    vbox = (center_of_mass[0], center_of_mass[1], 1, 1)
    for i, box in enumerate(box_wh):
        bbox = (int(width * box[0]), int(height * box[2]), int(width * box[1] - width * box[0]),
                int(height * box[3] - height * box[2]))
        if is_intersect(vbox, bbox, 0):
            return i
    return -1


def get_connected_components(bb_final_graph, seen, idx, cnt, bb_graph):
    if len(bb_graph[idx]) == 0:
      return bb_final_graph , seen
    for j,bbj in enumerate(bb_graph[idx]):
       if bbj not in seen:
           bb_final_graph[cnt].append(bbj)
           seen.append(bbj)
           bb_final_graph , seen = get_connected_components(bb_final_graph, seen, bbj, cnt, bb_graph)
    return bb_final_graph, seen


def get_relevant_bb(bb_final_list, img_shape):
      rel_bb = []
      for bbox in bb_final_list:
            COM = (bbox[0]+int(bbox[2]/2), bbox[1]+int(bbox[3]/2))
            boxNum = is_in_bb(COM, img_shape)
            if boxNum >= 0:
               rel_bb.append((boxNum, bbox[0], bbox[1], bbox[2], bbox[3]) )
      return rel_bb

def join_strings_as_path(dirs_list):
    return Path('/'.join(dirs_list)).as_posix()

def save_mov_info(frames, dir_path, filename):
    if len(frames) == 0:
        LOGGER.warn('save_mov_info --> didnt find any movement')
        return
    final_frames = []
    for frame_couple in frames:
            final_frames.append(frame_couple)
            final_frames[-1][0] -= MOV_FRAME_HANDLE['frames_to_expand']
            final_frames[-1][1] += MOV_FRAME_HANDLE['frames_to_expand']

    with open(join_strings_as_path([dir_path, filename+'.txt']), 'w') as f:
        f.write(json.dumps(final_frames))

def make_dir_if_not_exist(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def resize_keep_ratio(image, size):
    try:
        h, w = image.shape[:2]
        h_new, w_new = size
        c = None if len(image.shape) < 3 else image.shape[2]
        #TODO: Deal with c = None
        image_final = np.zeros((size[0], size[1], c), np.uint8)
        aspect_ratio_h = h / h_new
        aspect_ratio_w = w / w_new
        if aspect_ratio_h == aspect_ratio_w:
            return cv2.resize(image, (size[1], size[0]))
        elif aspect_ratio_w < aspect_ratio_h:
            image_new_width = w / aspect_ratio_h
            image_new_height = size[0]
        else:
            image_new_height = h / aspect_ratio_w
            image_new_width = size[1]
        image_new = cv2.resize(image, (int(image_new_width), int(image_new_height)))

        if image_new.shape[0] == image_final.shape[0]:
            gap = int(np.floor((image_final.shape[1] - image_new.shape[1])/2))
            image_final[:, gap:(gap + image_new.shape[1]), :] = image_new
        else:
            gap = int(np.floor((image_final.shape[0] - image_new.shape[0])/2))
            image_final[gap:(gap + image_new.shape[0]), :, :] = image_new
        return image_final
    except:
        LOGGER.error('resize issue')
        LOGGER.error(str(traceback.format_exc()))
        return image_final


def copy_all_video_refrences(video_name, dir_tree, target='final_detection', wait_ffmpeg=False):
    video_name_as_dir = join_strings_as_path(video_name.split('_'))
    target_path = join_strings_as_path([dir_tree.get_path_string(['final_detection']), video_name_as_dir])
    make_dir_if_not_exist(target_path)
    vid_locations = dir_tree.find_files(video_name)
    tmp_location = join_strings_as_path([dir_tree.sub_dirs['diff_detection'].curr_dir, 'tmp'])
    while wait_ffmpeg and \
            any([os.path.dirname(idx) == tmp_location for idx in vid_locations]):
        LOGGER.warning('Darknet finished but ffmpeg didnt, waiting a few seconds ({})'.format(video_name))
        time.sleep(10)
        vid_locations = dir_tree.find_files(video_name)
    for file in vid_locations:
        if os.path.isfile(file) and os.path.dirname(file) != target_path:
            target_file_path = join_strings_as_path([target_path, os.path.basename(file)])
            shutil.move(file, target_file_path, copy_function=shutil.copy2)
    return target_path

def is_str_in_file(input, file_name):
    if isinstance(input, list):
        return any([my_str in file_name for my_str in input])
    elif isinstance(input, str):
        return input in file_name
    else:
        LOGGER.error('Dont know what to do with {}'.format(str(input)))

class DirsHandler:
    def __init__(self, dir_dict, main_dir='', curr_key=''):
        try:
            if 'main_dir' in dir_dict:
                _main_dir = dir_dict['main_dir']
            else:
                _main_dir = main_dir
            self.curr_dir = join_strings_as_path([_main_dir, dir_dict['dir_name']])
            self.curr_key = curr_key
            self.sub_dirs = dict()
            for key in dir_dict:
                if key == 'main_dir' or key == 'dir_name':
                    continue
                elif isinstance(dir_dict[key], dict):
                    self.sub_dirs[key] = DirsHandler(dir_dict[key], main_dir=self.curr_dir,
                                                     curr_key=(self.curr_key + '_' + key).lstrip('_'))
                else:
                    self.sub_dirs[key] = dir_dict[key]
            self.all_dirs = self.make_toc_dict()
        except:
            LOGGER.error('Dir Tree init error')
            return None

    def make_toc_dict(self):
        all_dirs = dict()
        if self.curr_key == '':
            all_dirs['main'] = self.curr_dir
        else:
            all_dirs[self.curr_key] = self.curr_dir
        for key in self.sub_dirs:
            if isinstance(self.sub_dirs[key], DirsHandler):
                all_dirs = {**all_dirs, **self.sub_dirs[key].make_toc_dict()}
            elif isinstance(self.sub_dirs[key], str):
                all_dirs[self.curr_key+'_'+key] = join_strings_as_path([self.curr_dir, self.sub_dirs[key]])
            else:
                if not key == 'level':
                    print('Something wrong --> {}'.format(str(key)))
        self.all_dirs = all_dirs
        return all_dirs



    def find_files(self, string, regex=False, dirs_list=[]):
        res = []
        if len(dirs_list) == 0 or isinstance(self.sub_dirs[dirs_list[0]], str):
            if len(dirs_list) == 0:
                search_dir = self.curr_dir
            else:
                search_dir = join_strings_as_path([self.curr_dir, self.sub_dirs[dirs_list[0]]])
            for dir, _, _ in os.walk(self.curr_dir):
                if regex:
                    res.extend(glob(join_strings_as_path([dir, string])))
                else:
                    res.extend(glob(join_strings_as_path([dir, '*' + string + '*'])))
        else:
            key = dirs_list[0]
            res = self.sub_dirs[key].find_files(string, dirs_list=dirs_list[1:])
        return res



    def get_path_string(self, dirs_list, file_name=''):
        try:
            if len(dirs_list) == 0:
                if len(file_name) > 0:
                    return join_strings_as_path([self.curr_dir, file_name])
                else:
                    return self.curr_dir
            if isinstance(dirs_list, str):
                dirs_list = [dirs_list]

            key = dirs_list[0]
            if isinstance(self.sub_dirs[key], DirsHandler):
                return self.sub_dirs[key].get_path_string(dirs_list[1:], file_name=file_name)
            else:
                if len(file_name) > 0:
                    return join_strings_as_path([self.curr_dir, self.sub_dirs[key], file_name])
                else:
                    return join_strings_as_path([self.curr_dir, self.sub_dirs[key]])
        except:
            LOGGER.error('Dir Tree get path string error')
            return None


    def exec_func(self, func):
        func(self.curr_dir)
        for key in self.sub_dirs:
            if isinstance(self.sub_dirs[key], DirsHandler):
                self.sub_dirs[key].exec_func(func)
            elif isinstance(self.sub_dirs[key], str):
                func(join_strings_as_path([self.curr_dir, self.sub_dirs[key]]))
            else:
                if not key == 'level':
                    print('Something wrong --> {}'.format(str(key)))






