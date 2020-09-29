from cfg import *
import datetime
from pathlib import Path
from glob import glob
import json

def time2string(input_time):
    return datetime.datetime.fromtimestamp(input_time).strftime('%Y%m%d_%H%M%S')

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


def is_in_bb(center_of_mass, img_shape, bbox=None, box_wh=BOX_WIDTH_HEIGHT):
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

class DirsHandler: #TODO: add dictionary with all paths
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






