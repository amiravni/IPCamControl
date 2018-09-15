# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:05:22 2017

@author: PC-BRO
"""

'''
TODO: 
5. save also diff video
'''




import numpy as np
import cv2
import time
import datetime
import requests
from threading import Thread, ThreadError, Lock
from ConfigParser import SafeConfigParser
import shutil
import os
import subprocess

## constants
STREAM_SIM = 2
STREAM_RTSP = 1
STREAM_HTTP = 0

## general params
CALC_DIFFS = True
DEBUG = False # Show video and diffs
STREAM_TYPE = 'rtsp';
VID_FPS = -1 ## changes on "main"
TIME_RATIO = 1000.0
MAX_SINGLE_VIDEO_TIME = 300
MAIN_DIR = 'D:\\IPcam\\Recordings'
REC_DIR = '\\Detection'
REC_DIR_2del = '\\noDetection'
sim_fileName = 'D:\\IPcam\\Recordings\\20180913_080339_test.mkv'
mainLogFilename_CAM = MAIN_DIR+'\\MainLog_CAM.log'
mainLogFilename_FH = MAIN_DIR+'\\MainLog_FH.log'

##diff params
BOX_WIDTH_HEIGHT = np.array([[0.55,0.55+0.25,0.15,0.15+0.3],[0.05,0.05+0.4,0.4,0.4+0.55],[0.8,0.8+0.18,0.45,0.45+0.5]])
#BOX_WIDTH_HEIGHT =  np.array([[0.05,0.05+0.3,0.05,0.05+0.3]])
BOX_WIDTH_HEIGHT_REF = np.array([0.5,0.5+0.2,0.6,0.6+0.3])
NUM_OF_BOXES = len(BOX_WIDTH_HEIGHT)
diffImage = [np.zeros(1) for _ in range(NUM_OF_BOXES)]
diffScore = np.zeros(NUM_OF_BOXES)
meanDScore = np.zeros(NUM_OF_BOXES)
issue = np.zeros(NUM_OF_BOXES)
MIN_AVG_SCORE_MIN_MAX = [1,8]
MIN_AVG_SCORE_RATIO = 1.5
MIN_AVG_SCORE_RATIO_ADD = 1
MIN_AVG_SCORE_SMALL_BOX_RATIO = 1.3
MIN_AVG_SCORE_SMALL_BOX_RATIO_ADD = 1
MAX_AVG_SCORE_REF_DAY = 7
MAX_AVG_SCORE_REF_NIGHT = 7
MIN_ISSUE_COUNT = 3

RESIZE_ORG = 4
REF_WINDOW_LEN = 64
REF_WINDOW_LEN_LOG = np.log2(REF_WINDOW_LEN).astype('uint8') ## for fast div/mul
KERNEL = np.ones((3,3),np.float32)/9


## file handle params
global files2Handle
global threadsLock
class files2HandleClass:
    name = []
    nLen = 0
files2Handle = files2HandleClass()
threadsLock = Lock()
FFMPEG_EXE_LOC = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'
TIME2DELETE_SEC = 86400
TIME2ACTIVATE_DELETE_SEC = MAX_SINGLE_VIDEO_TIME + 60


startTime = time.time()


class Cam():

  def Print(self,string,var=0):
      string = str(time.time())+"\t"+str(string)
      if var == 0:
          print string
          fid = open(mainLogFilename_CAM,'a+')
          print >> fid, string
          fid.close()
      elif var == 1:
          print string
      elif var == 2:
          fid = open(mainLogFilename_CAM,'a+')
          print >> fid, string
          fid.close()          

  def __init__(self, url):
    

    if url[0:4] == 'rtsp':
        self.stream = cv2.VideoCapture(url)
        self.stream.set(cv2.CAP_PROP_FOURCC , cv2.VideoWriter_fourcc('H', '2', '6', '4'));
        #self.stream.get(cv2.cv.CV_CAP_PROP_BUFFERSIZE)
        self.stream_type = STREAM_RTSP        
    elif url[0:4] == 'http':
        self.stream = requests.get(url, stream=True)
        self.stream_type = STREAM_HTTP
    elif url[0:3] == 'sim':
        self.stream = cv2.VideoCapture(sim_fileName)
        self.stream.set(cv2.CAP_PROP_POS_FRAMES,0);
        self.stream_type = STREAM_SIM
    self.thread_cancelled = False
    self.thread = Thread(target=self.run)
    self.sysTime = time.time()
    self.firstSysTime = self.sysTime
    self.timeString = datetime.datetime.fromtimestamp( self.firstSysTime ).strftime('%Y_%m_%d_%H_%M_%S')
    #self.img = -1
    self.img_lrg = -1
    self.img_2show = -1
    self.img_curr = -1
    self.img_curr_2show = -1
    self.img_last_vec = []
    self.img_last = -1
    self.img_last_sum = -1
    self.img_count = -1
    self.video = -1
    self.video_mov = -1
    self.diffScoreVec = np.zeros(10)
    self.diffScore2Use = -1
    self.movmentDetection = 0
    self.isDay = True
    
    self.Print(time.time())
    self.Print("camera initialized")

    
  def start(self):
    self.thread.start()
    self.Print("camera stream started")

  def add_image_to_video(self,flag):
    if flag == 0:
        self.video.write(self.img_lrg)
        self.img_count = self.img_count + 1
    else:
        self.video_mov.write(self.img_lrg)
    #print (time.time() - self.sysTime)*TIME_RATIO
    self.sysTime = time.time()


  def get_diff_and_score(self,box):
      isDay = self.isDay
      height , width , layers =  self.img_curr_2show.shape
      cv2.rectangle(self.img_curr_2show, (int(width*box[0]), int(height*box[2])), (int(width*box[1]), int(height*box[3])), (0, 0, 255), 2)
      gray_curr = cv2.cvtColor(self.img_curr[int(height*box[2]): int(height*box[3]) , int(width*box[0]): int(width*box[1])], cv2.COLOR_BGR2LAB)
      gray_last = cv2.cvtColor(self.img_last[int(height*box[2]): int(height*box[3]) , int(width*box[0]): int(width*box[1])], cv2.COLOR_BGR2LAB)
      isColor = np.max(gray_curr[0:,0:,1:]) - np.min(gray_curr[0:,0:,1:])
      if isColor > 10:
          self.isDay = True
          diffImage1 = cv2.absdiff(gray_curr[0:,0:,1],gray_last[0:,0:,1])
          diffImage2 = cv2.absdiff(gray_curr[0:,0:,2],gray_last[0:,0:,2])
          diffImage = (diffImage1 + diffImage2 ) << 1
          diffScore = np.mean(diffImage)
      else:
          self.isDay = False
          diffImage = cv2.absdiff(gray_curr[0:,0:,0],gray_last[0:,0:,0])
          diffScore = np.mean(diffImage)
          
      if isDay is not self.isDay:
          self.Print("DAY/NIGHT Changed")
          
          
      meanDScore = -1
      if self.diffScore2Use > 0 and diffScore > min(max(self.diffScore2Use,MIN_AVG_SCORE_MIN_MAX[0]),MIN_AVG_SCORE_MIN_MAX[1]):
          diffWidth = diffImage[0,:].size
          diffHeight =  diffImage[:,0].size
          meanD = np.zeros(5)
          meanD[0] = np.mean(diffImage[0:int(diffWidth/2),0:int(diffHeight/2)])
          meanD[1] = np.mean(diffImage[0:int(diffWidth/2),int(diffHeight/2):-1])
          meanD[2] = np.mean(diffImage[0:int(diffWidth/2),0:int(diffHeight/2)])
          meanD[3] = np.mean(diffImage[int(diffWidth/2):-1,int(diffHeight/2):-1])
          meanD[4] = np.mean(diffImage[int(diffWidth/4):int(3*diffWidth/4),int(diffHeight/4):int(3*diffHeight/4)])
          meanDScore = sum((meanD - MIN_AVG_SCORE_SMALL_BOX_RATIO_ADD)>(self.diffScore2Use*MIN_AVG_SCORE_SMALL_BOX_RATIO))
      #print box[0],diffScore,meanDScore,self.diffScore2Use
      return diffImage,diffScore,meanDScore

  def time2string(self,input_time):
      return datetime.datetime.fromtimestamp( input_time ).strftime('%Y%m%d_%H%M%S')

  def handleLastVideo(self):
    global files2Handle
    global threadsLock
    if type(self.video) is int:
        return -1 
    else:
        self.video.release()       
        self.video_mov.release() 
        if self.movmentDetection == 1:
            ### keep file
            REC_DIR_TMP = REC_DIR
        else:
#            ### delete file
#            os.remove(self.fileName_video)
#            return self.movmentDetection
            ### keep file in other dir      
            REC_DIR_TMP = REC_DIR_2del
        ### move file and save data for FileHandle thread
        shutil.move(self.fileName_video, MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video)
        shutil.move(self.filename_log, MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_log)
        threadsLock.acquire()
        try:
            files2Handle.name.append(MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video)
            files2Handle.nLen = files2Handle.nLen + 1  
            self.Print([files2Handle.name, files2Handle.nLen])
        finally:
            threadsLock.release() 
            
        if self.movmentDetection == 1:
            shutil.move(self.fileName_video_mov, MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video_mov)
            threadsLock.acquire()
            try:
                files2Handle.name.append(MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video_mov)
                files2Handle.nLen = files2Handle.nLen + 1  
                self.Print([files2Handle.name, files2Handle.nLen])
            finally:
                threadsLock.release() 
        else:
            os.remove(self.fileName_video_mov)
                        
            
            
        return self.movmentDetection
    
  def initVideo(self):
    height , width , layers =  self.img_lrg.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    self.firstSysTime = time.time()
    self.timeString = self.time2string(self.firstSysTime)
    self.filename_short_video = self.timeString+'.mkv'
    self.filename_short_video_mov = self.timeString+'_mov.mkv'
    self.filename_short_log = self.timeString+'_log.txt'     
    self.fileName_video = MAIN_DIR+'\\'+self.filename_short_video
    self.fileName_video_mov = MAIN_DIR+'\\'+self.filename_short_video_mov
    self.filename_log = MAIN_DIR+'\\'+self.filename_short_log
    self.Print(self.fileName_video)
    if type(self.video) is not int:
        self.video.release() 
    if type(self.video_mov) is not int:
        self.video.release()         
    self.video = cv2.VideoWriter(self.fileName_video,fourcc,VID_FPS,(width,height))
    self.video_mov = cv2.VideoWriter(self.fileName_video_mov,fourcc,VID_FPS,(width,height))      
    self.add_image_to_video(0)
    self.add_image_to_video(1)
    fid = open(self.filename_log,'w')                  
    print >>fid ,self.timeString
    fid.close()    

  def getImage(self,bytes):
    stream_new_flag = 0
    img_tmp = 0
    if self.stream_type == STREAM_HTTP:
        bytes+=self.stream.raw.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')            
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]                
            stream_new_flag = 1
            img_tmp = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
    elif self.stream_type == STREAM_RTSP:
        ret, img_tmp = self.stream.read()
        if ret is True:
            stream_new_flag = 1
    elif self.stream_type == STREAM_SIM:
        ret, img_tmp = self.stream.read()
        if ret is True:
            stream_new_flag = 1  
    return stream_new_flag , img_tmp , bytes   

  def updateImages(self):

      img_tmp = self.img_lrg[0::RESIZE_ORG,0::RESIZE_ORG,0:]
      #tic = time.time()
      #img_tmp = cv2.filter2D(img_tmp,-1,KERNEL)
      img_tmp = cv2.blur(img_tmp,(2,2))
      #img_tmp = cv2.medianBlur(img_tmp,5)
      #print(time.time() - tic)
      #img_tmp = cv2.medianBlur(img_tmp,5)
      #img_tmp = cv2.GaussianBlur(img_tmp,(3,3),0)      
      
      if len(self.img_last_vec) is 0:
          self.img_last_vec.append(img_tmp.astype('uint16'))
          self.img_last_sum = self.img_last_vec[0].copy()
      else:
          if len(self.img_last_vec) == REF_WINDOW_LEN:
              self.img_last_sum = self.img_last_sum - self.img_last_vec[0]
              self.img_last_vec.remove(self.img_last_vec[0])
          self.img_last_vec.append((self.img_curr).astype('uint16'))
          self.img_last_sum = self.img_last_sum + self.img_last_vec[-1]      
      #self.img_last = ((sum(np.array(self.img_last_vec)))/len(self.img_last_vec) )
      
      if len(self.img_last_vec) == REF_WINDOW_LEN:
          self.img_last = (self.img_last_sum >> REF_WINDOW_LEN_LOG).astype('uint8')#
      else:
          self.img_last = (self.img_last_sum / len(self.img_last_vec)).astype('uint8')      
          
      self.img_curr = img_tmp.copy()
      self.img_curr_2show = (self.img_curr).copy()
      if type(self.img_last) is int:
          self.img_last = self.img_curr

  def run(self):
    bytes=''
    self.img_last_sum = -1
    tic = time.time()
    while not self.thread_cancelled:
      try:
        ### Init Video after X seconds
        if self.img_count == 1:
            self.handleLastVideo()
            self.initVideo()
            issue[0:] = issue[0:]*0
            self.movmentDetection = 0
        
        ### get new Image
        #print(time.time() - tic)
        stream_new_flag , self.img_lrg , bytes = self.getImage(bytes)
        #tic = time.time()
      
        if stream_new_flag == 1:
          
          self.updateImages()

          ### calc diff image and score
          if CALC_DIFFS:
              height , width , layers =  self.img_curr_2show.shape
              diffImageRef,diffScoreRef,meanDScoreRef = self.get_diff_and_score(BOX_WIDTH_HEIGHT_REF)   
              self.diffScoreVec[0:-1] = self.diffScoreVec[1:]
              self.diffScoreVec[-1] = diffScoreRef
              if self.diffScoreVec[0] > 0:
                  self.diffScore2Use = np.mean(self.diffScoreVec) * MIN_AVG_SCORE_RATIO
                  if self.isDay is True and self.diffScore2Use > MAX_AVG_SCORE_REF_DAY:
                      self.diffScore2Use = 999999
                  if self.isDay is False and self.diffScore2Use > MAX_AVG_SCORE_REF_NIGHT:
                      self.diffScore2Use = 999999                      
              for diffIdx in range(0,NUM_OF_BOXES):
                  diffImage[diffIdx] = np.zeros([int(height*(BOX_WIDTH_HEIGHT[diffIdx][1] - BOX_WIDTH_HEIGHT[diffIdx][0] )) , int(width*(BOX_WIDTH_HEIGHT[diffIdx][3] - BOX_WIDTH_HEIGHT[diffIdx][2] ))])
                  diffImage[diffIdx],diffScore[diffIdx],meanDScore[diffIdx] = self.get_diff_and_score(BOX_WIDTH_HEIGHT[diffIdx])
                  if self.diffScore2Use > 0 and (diffScore[diffIdx] - MIN_AVG_SCORE_RATIO_ADD) > self.diffScore2Use and meanDScore[diffIdx] > 0 and meanDScore[diffIdx] < 5:
                      issue[diffIdx] =  issue[diffIdx] + 1
                      #print "\n ############## ISSUE! ############## \n"
                  else:
                      issue[diffIdx] =  max(0,issue[diffIdx] - 0.5)      
              
          ### copy image (Just if we want to change something on the image we save to video)
          self.img_2show = np.array(self.img_curr_2show)

          ### Show image, save to video, and find issues
          if self.img_count > 1:
              if DEBUG: 
                  cv2.imshow('cam',self.img_2show)
                  #cv2.moveWindow('cam',)
                  #cv2.imshow('camLast',self.img_last)
              if CALC_DIFFS:
                  if DEBUG:
                      for diffIdx in range(0,NUM_OF_BOXES):
                          cv2.imshow(('diff'+str(diffIdx)),diffImage[diffIdx])
                      cv2.imshow('diffRef',diffImageRef)
                  #print 'REF: ',self.diffScore2Use
                  if self.diffScore2Use > 0 and (diffScore > self.diffScore2Use).any() :
                      #if sum(issue) > 0 or (diffScore > 2*self.diffScore2Use).any():
                          currTimeString = self.time2string(time.time())                   
                          if DEBUG:
                              print currTimeString ,diffScore,meanDScore,issue, 'REF: ',self.diffScore2Use
                          fid = open(self.filename_log,'a+') 
                          print >>fid ,currTimeString ,diffScore,meanDScore,issue, 'REF: ',self.diffScore2Use
                          fid.close()
                  if sum(issue) > MIN_ISSUE_COUNT: #(issue> MIN_ISSUE_COUNT ).any():
                      if self.movmentDetection is not 1:
                          print "\n ############## MOVEMENT DETECTED! ############## \n"
                          fid = open(self.filename_log,'a+') 
                          print >>fid ,"\n ############## MOVEMENT DETECTED! ############## \n"
                          fid.close()
                      self.movmentDetection = 1
                      height , width , layers =  self.img_lrg.shape
                      cv2.rectangle(self.img_lrg, (int(width*0), int(height*0)), (int(width*1), int(height*1)), (0, 0, 255), 10)
                      self.add_image_to_video(1)
              
              self.add_image_to_video(0)
              
          
          ### first time (no "diff" image)
          if self.img_count == -1:
              self.img_count = 1
          
          ### Init Video after X seconds
          if time.time() - self.firstSysTime > MAX_SINGLE_VIDEO_TIME:
              self.img_count = 1
              
          if cv2.waitKey(1) == 999:
            exit(0)
      except ThreadError:
        self.Print("SOMETHING WENT WRONG!")
        self.thread_cancelled = True
        self.video.release() 
        self.video_mov.release() 


  def is_running(self):
    return self.thread.isAlive()
      
    
  def shut_down(self):
    self.thread_cancelled = True
    #block while waiting for thread to terminate
    while self.thread.isAlive():
      time.sleep(1)
    return True



class FileHandler():

  def Print(self,string,var=0):
      string = str(time.time())+"\t"+str(string)
      if var == 0:
          print string
          fid = open(mainLogFilename_FH,'a+')
          print >> fid, string
          fid.close()
      elif var == 1:
          print string
      elif var == 2:
          fid = open(mainLogFilename_FH,'a+')
          print >> fid, string
          fid.close()    
          
         
  def __init__(self):
    

    self.thread_cancelled = False
    self.thread = Thread(target=self.run)
    self.Print("FileHandler initialized")
    
  def start(self):
    self.thread.start()
    self.Print("FileHandler started")
    
  def is_running(self):
    return self.thread.isAlive()
      
    
  def shut_down(self):
    self.thread_cancelled = True
    #block while waiting for thread to terminate
    while self.thread.isAlive():
      time.sleep(1)
    return True   

  def DeleteOldFiles(self,dir_to_search):
    for dirpath, dirnames, filenames in os.walk(dir_to_search):
       for file in filenames:
          curpath = os.path.join(dirpath, file)
          deltaTimeSeconds = time.time() - os.path.getmtime(curpath)
          if deltaTimeSeconds > TIME2DELETE_SEC:
              self.Print("deleting: " + curpath)
              os.remove(curpath)

  def run(self):
      global files2Handle
      global threadsLock 
      command = [FFMPEG_EXE_LOC, '-i', 'fileName', '-c:v', 'libx264', '-tune', 'film', '-preset', 'fast', '-profile:v', 'high444', '-crf', '38', 'fileNameMP4','-c:a', 'copy','-loglevel', 'error']
      lastDeleted = 0
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
            if nLen_local > 0:
                name = ''.join(filename_local.split('.')[:-1])
                output = '{}.mp4'.format(name)
                command[2] = filename_local
                command[13] = output
                self.Print("START ENCODING: "+filename_local[len(MAIN_DIR):]+" --> " + output[len(MAIN_DIR):])
                try:                
                    subprocess.call(command)
                    self.Print("DONE ENCODING: "+filename_local[len(MAIN_DIR):]+" --> " + output[len(MAIN_DIR):])
                    os.remove(filename_local)
                except:
                    self.Print("ERROR ENCODING: "+filename_local[len(MAIN_DIR):]+" --> " + output[len(MAIN_DIR):])
                threadsLock.acquire()
                try:
                    files2Handle.name.remove(files2Handle.name[0])
                    files2Handle.nLen = files2Handle.nLen - 1
                finally:
                    threadsLock.release()                 
            else:
                #print "Sleeping..."
                time.sleep(10)
                if time.time() - lastDeleted > TIME2ACTIVATE_DELETE_SEC:
                    self.DeleteOldFiles(MAIN_DIR)
                    self.DeleteOldFiles(MAIN_DIR+REC_DIR)
                    self.DeleteOldFiles(MAIN_DIR+REC_DIR_2del)
                    
  
    
if __name__ == "__main__":
    #print time.time()    
    #time.sleep(3600*3)   
    parser = SafeConfigParser()
    parser.read('.\\config.ini')
    if STREAM_TYPE == 'http':    
        url = parser.get('ipcamconfig', 'httpurl')
        VID_FPS = 4
    elif STREAM_TYPE == 'rtsp':
        url = parser.get('ipcamconfig', 'rtspurl')
        VID_FPS = 13
    elif STREAM_TYPE == 'sim':
        url = 'sim'
        VID_FPS = 13
    if not os.path.exists(MAIN_DIR+REC_DIR):
        os.makedirs(MAIN_DIR+REC_DIR)   
    if not os.path.exists(MAIN_DIR+REC_DIR_2del):
        os.makedirs(MAIN_DIR+REC_DIR_2del)          
    cam = Cam(url)
    cam.start()
    _FileHandler = FileHandler()
    _FileHandler.start()