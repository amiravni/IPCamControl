# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:05:22 2017

@author: PC-BRO
"""

'''
TODO: 
4. compress file if not deleted --> with another thread? 
5. save also diff video
6. add DEBUG mode to watch to video
'''

import numpy as np
import cv2
import time
import datetime
import requests
from threading import Thread, Event, ThreadError
from ConfigParser import SafeConfigParser
import shutil
import os

HTTP_FLAG = True
TIME_RATIO = 1000.0
MAX_SINGLE_VIDEO_TIME = 390
MAIN_DIR = 'D:\\IPcam\\Recordings'
REC_DIR = '\\Detection'
BOX_WIDTH_HEIGHT = np.array([[0.5,0.5+0.3,0.05,0.05+0.4],[0.05,0.05+0.4,0.4,0.4+0.55],[0.8,0.8+0.18,0.45,0.45+0.5]])
BOX_WIDTH_HEIGHT_REF = np.array([0.6,0.6+0.1,0.6,0.6+0.1])
NUM_OF_BOXES = len(BOX_WIDTH_HEIGHT)
diffImage = [[np.zeros(1)],[np.zeros(1)],[np.zeros(1)]]  ###< --- how to do that automatically???
diffScore = np.zeros(NUM_OF_BOXES)
meanDScore = np.zeros(NUM_OF_BOXES)
issue = np.zeros(NUM_OF_BOXES)
MIN_AVG_SCORE_MIN_MAX = [4,8]
MIN_AVG_SCORE_RATIO = 1.5
MIN_AVG_SCORE_SMALL_BOX_RATIO = 1.3
MIN_ISSUE_COUNT = 3
startTime = time.time()

class Cam():

  def __init__(self, url):
    

    if url[0:4] == 'rtsp':
        self.stream = cv2.VideoCapture(url)
        self.stream.set(cv2.cv.CV_CAP_PROP_FOURCC, cv2.cv.CV_FOURCC('H', '2', '6', '4'));
        #self.stream.get(cv2.cv.CV_CAP_PROP_BUFFERSIZE)
        self.stream_type = 1        
    elif url[0:4] == 'http':
        self.stream = requests.get(url, stream=True)
        self.stream_type = 0
    self.thread_cancelled = False
    self.thread = Thread(target=self.run)
    self.sysTime = time.time()
    self.firstSysTime = self.sysTime
    self.timeString = datetime.datetime.fromtimestamp( self.firstSysTime ).strftime('%Y_%m_%d_%H_%M_%S')
    self.img = -1
    self.img_2show = -1
    self.img_curr = -1
    self.img_curr_2show = -1
    self.img_last = -1
    self.img_count = -1
    self.video = -1
    self.diffScoreVec = np.zeros(10)
    self.diffScore2Use = -1
    self.movmentDetection = 0
    print "camera initialised"
#    fid = open(MAIN_DIR+'log_'+self.timeString+'.txt','w')                  
#    print >>fid ,"START"
#    fid.close()    


    
  def start(self):
    self.thread.start()
    print "camera stream started"

  def add_image_to_video(self):
    self.video.write(self.img)
    self.img_count = self.img_count + 1
    #print (time.time() - self.sysTime)*TIME_RATIO
    self.sysTime = time.time()
  
  def get_diff_and_score(self,box):
      height , width , layers =  self.img_curr_2show.shape
      cv2.rectangle(self.img_curr_2show, (int(width*box[0]), int(height*box[2])), (int(width*box[1]), int(height*box[3])), (0, 0, 255), 2)
      gray_curr = cv2.cvtColor(self.img_curr[int(height*box[2]): int(height*box[3]) , int(width*box[0]): int(width*box[1])], cv2.COLOR_BGR2GRAY)  
      gray_last = cv2.cvtColor(self.img_last[int(height*box[2]): int(height*box[3]) , int(width*box[0]): int(width*box[1])], cv2.COLOR_BGR2GRAY)   
      diffImage = cv2.absdiff(gray_curr,gray_last)
      diffScore = np.mean(diffImage)
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
          meanDScore = sum(meanD>(self.diffScore2Use*MIN_AVG_SCORE_SMALL_BOX_RATIO))
      return diffImage,diffScore,meanDScore

  def time2string(self,input_time):
      return datetime.datetime.fromtimestamp( input_time ).strftime('%Y_%m_%d_%H_%M_%S')

  def handleLastVideo(self):
    if type(self.video) is int:
        return -1 
    else:
        self.video.release()         
        if self.movmentDetection == 1:
            ### keep file
            shutil.move(self.fileName_video, MAIN_DIR+REC_DIR+'\\'+self.filename_short_video)
            shutil.move(self.filename_log, MAIN_DIR+REC_DIR+'\\'+self.filename_short_log)            
            return 1
        else:
            ### delete file
            #os.remove(self.fileName_video) 
            return 0
    
  def initVideo(self):
    height , width , layers =  self.img.shape
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    self.firstSysTime = time.time()
    self.timeString = self.time2string(self.firstSysTime)
    self.filename_short_video = self.timeString+'.mkv'
    self.filename_short_log = self.timeString+'_log.txt'     
    self.fileName_video = MAIN_DIR+'\\'+self.filename_short_video
    self.filename_log = MAIN_DIR+'\\'+self.filename_short_log
    print self.fileName_video
    if type(self.video) is not int:
        self.video.release() 
    self.video = cv2.VideoWriter(self.fileName_video,fourcc,4,(width,height))      
    self.add_image_to_video()
    fid = open(self.filename_log,'w')                  
    print >>fid ,self.timeString
    fid.close()    

  def getImage(self,bytes):
    stream_new_flag = 0
    img_tmp = 0
    if self.stream_type == 0:
        bytes+=self.stream.raw.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')            
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]                
            stream_new_flag = 1
            img_tmp = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
    elif self.stream_type == 1:
        ret, img_tmp = self.stream.read()
        if ret is True:
            stream_new_flag = 1
    return stream_new_flag , img_tmp , bytes   

  def run(self):
    bytes=''
    while not self.thread_cancelled:
      try:
        ### Init Video after X seconds
        if self.img_count == 1:
            reslt = self.handleLastVideo()
            if reslt != -1:
              fid = open(self.filename_log,'a+') 
              print >>fid ,'HANDLE RESULTS: ' , reslt
              print >>fid ,'HANDLE RESULTS: ' , reslt
              fid.close()
            self.initVideo()
            issue[0:] = issue[0:]*0
            self.movmentDetection = 0
        
        ### get new Image
        stream_new_flag , img_tmp , bytes = self.getImage(bytes)
        
        if stream_new_flag == 1:
          ### copy image to class
          self.img_last = self.img_curr
          self.img_curr = img_tmp
          self.img_curr_2show = (self.img_curr).copy()
          if type(self.img_last) is int:
              self.img_last = self.img_curr
          
          ### calc diff image and score
          height , width , layers =  self.img_curr_2show.shape
          diffImageRef,diffScoreRef,meanDScoreRef = self.get_diff_and_score(BOX_WIDTH_HEIGHT_REF)   
          self.diffScoreVec[0:-1] = self.diffScoreVec[1:]
          self.diffScoreVec[-1] = diffScoreRef
          if self.diffScoreVec[0] > 0:
              self.diffScore2Use = np.mean(self.diffScoreVec) * MIN_AVG_SCORE_RATIO
          for diffIdx in range(0,len(BOX_WIDTH_HEIGHT)):
              diffImage[diffIdx] = np.zeros([int(height*(BOX_WIDTH_HEIGHT[diffIdx][1] - BOX_WIDTH_HEIGHT[diffIdx][0] )) , int(width*(BOX_WIDTH_HEIGHT[diffIdx][3] - BOX_WIDTH_HEIGHT[diffIdx][2] ))])
              diffImage[diffIdx],diffScore[diffIdx],meanDScore[diffIdx] = self.get_diff_and_score(BOX_WIDTH_HEIGHT[diffIdx])
              if self.diffScore2Use > 0 and diffScore[diffIdx] > self.diffScore2Use and meanDScore[diffIdx] > 0 and meanDScore[diffIdx] < 5:
                  issue[diffIdx] =  issue[diffIdx] + 1
              else:
                  issue[diffIdx] =  max(0,issue[diffIdx] - 0.5)      
          
          ### copy image (Just if we want to change something on the image we save to video)
          self.img =  np.array(self.img_curr) #np.hstack(( np.array(self.img_curr) , np.array(self.img_last)))
          self.img_2show = np.array(self.img_curr_2show) #np.hstack(( np.array(self.img_curr_2show) , np.array(self.img_last)))
          
          ### Show image, save to video, and find issues
          if self.img_count > 1:
              cv2.imshow('cam',self.img_2show)
              for diffIdx in range(0,len(BOX_WIDTH_HEIGHT)):
                  cv2.imshow(('diff'+str(diffIdx)),diffImage[diffIdx])
              cv2.imshow('diffRef',diffImageRef)
              print 'REF: ',self.diffScore2Use
              if self.diffScore2Use > 0 and (diffScore > self.diffScore2Use).any() :
                  currTimeString = self.time2string(time.time())                   
                  print currTimeString ,diffScore,meanDScore,issue, 'REF: ',self.diffScore2Use
                  fid = open(self.filename_log,'a+') 
                  print >>fid ,currTimeString ,diffScore,meanDScore,issue, 'REF: ',self.diffScore2Use
                  fid.close()
              if sum(issue) > MIN_ISSUE_COUNT: #(issue> MIN_ISSUE_COUNT ).any():
                  self.movmentDetection = 1
              self.add_image_to_video()
              
          
          ### first time (no "diff" image)
          if self.img_count == -1:
              self.img_count = 1
          
          ### Init Video after X seconds
          if time.time() - self.firstSysTime > MAX_SINGLE_VIDEO_TIME:
              self.img_count = 1
              
          if cv2.waitKey(1) == 999:
            exit(0)
      except ThreadError:
        self.thread_cancelled = True
        self.video.release() 


  def is_running(self):
    return self.thread.isAlive()
      
    
  def shut_down(self):
    self.thread_cancelled = True
    #block while waiting for thread to terminate
    while self.thread.isAlive():
      time.sleep(1)
    return True

  
    
if __name__ == "__main__":
    print time.time()    
    #time.sleep(3600*3)    
    print time.time()
    parser = SafeConfigParser()
    parser.read('.\\config.ini')
    if HTTP_FLAG:    
        url = parser.get('ipcamconfig', 'httpurl')
    else:
        url = parser.get('ipcamconfig', 'rtspurl')
    cam = Cam(url)
    cam.start()