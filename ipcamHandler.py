# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:05:22 2017

@author: PC-BRO
"""

'''
TODO: 
1. make NN work on big video (now works on small "debug" video with all the rectangles)
2. save NN file into original file
3. make Hysterezis in NN computations to avoid "jumps" between positive detections
4. make fileHandler more dynamic when choosing files (like the NN class)
5. add all values into config file
6. split into several files
'''




import numpy as np
import cv2
import time
import datetime
import requests
from threading import Thread, ThreadError, Lock
from configparser import ConfigParser
import shutil
import os
import subprocess


## constants
STREAM_SIM = 2
STREAM_RTSP = 1
STREAM_HTTP = 0

## general params
CALC_DIFFS = True
DEBUG = True # Show video and diffs
SAVE_DEBUG = True
STREAM_TYPE = 'sim';
VID_FPS = -1 ## changes on "main"
TIME_RATIO = 1000.0
MAX_SINGLE_VIDEO_TIME = 100#300
MAIN_DIR = 'D:\\IPcam\\Recordings'
REC_DIR = '\\Detection'
REC_DIR_2del = '\\noDetection'
sim_fileName = 'D:\\IPcam\\Recordings__forSim\\test\\arrangingMahsan\\20180928_104527.mp4'
simStartFrame = 1000
#sim_fileName = 'D:\\IPcam\\Recordings__forSim\\test\\20180926_011851.mp4'
#simStartFrame = 1500
#sim_fileName = 'D:\\IPcam\\Recordings__forSim\\test\\20180926_142924.mp4'
#simStartFrame = 390
mainLogFilename_CAM = MAIN_DIR+'\\MainLog_CAM.log'
mainLogFilename_FH = MAIN_DIR+'\\MainLog_FH.log'
mainLogFilename_NN = MAIN_DIR+'\\MainLog_NN.log'

##diff params
BOX_WIDTH_HEIGHT = np.array([[0.05,0.05+0.4,0.4,0.4+0.55],[0.5,0.5+0.15,0.2,0.2+0.3],[0.675,0.675+0.125,0.05,0.05+0.225],[0.8,0.8+0.18,0.45,0.45+0.5]])
minBBArea = (150,100,50,150)
#BOX_WIDTH_HEIGHT =  np.array([[0.05,0.05+0.3,0.05,0.05+0.3]])
BOX_WIDTH_HEIGHT_REF = np.array([0.5,0.5+0.2,0.6,0.6+0.3])
NUM_OF_BOXES = len(BOX_WIDTH_HEIGHT)
RESIZE_ORG = 2
REF_WINDOW_LEN = 2
REF_WINDOW_LEN_LOG = np.log2(REF_WINDOW_LEN).astype('uint8') ## for fast div/mul


## file handle params
global files2Handle
global threadsLock
class files2HandleClass:
    name = []
    afterNN = []
    dataForNN = []
    nLen = 0
files2Handle = files2HandleClass()
threadsLock = Lock()
FFMPEG_EXE_LOC = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'
TIME2DELETE_SEC = 86400 #24 Hours
TIME2ACTIVATE_DELETE_SEC = MAX_SINGLE_VIDEO_TIME + 60
font = cv2.FONT_HERSHEY_SIMPLEX

'''
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]
numOfTrackers = 4
trackerWorkTime = 5
'''

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

startTime = time.time()


class Cam():

  def addTextFromTFnet(self,result,imgcv):

    for res in result:
        x1 = res['topleft']['x']
        y1 = res['topleft']['y']
        x2 = res['bottomright']['x']
        y2 = res['bottomright']['y']   
        label = res['label']
        conf = res['confidence']
        if label is not 'car':
            print(label+':'+str(conf))
        cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(imgcv,label+':'+str(conf),(x1,y1), font, 0.5,(0,0,255),1,cv2.LINE_AA)
    return  imgcv

  def Print(self,string,var=0):
      string = str(time.time())+"\t"+str(string)
      if var == 0:
          print(string)
          fid = open(mainLogFilename_CAM,'a+')
          print(string, file=fid)
          fid.close()
      elif var == 1:
          print(string)
      elif var == 2:
          fid = open(mainLogFilename_CAM,'a+')
          print(string, file=fid)
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
        self.stream.set(cv2.CAP_PROP_POS_FRAMES,simStartFrame)
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
    self.img_count = -1
    self.img_count_mov = -1
    self.video = -1
    self.video_mov = -1
    self.movmentDetection = 0
    self.isDay = True
    self.detectionData = []
    
    self.Print(time.time())
    self.Print("camera initialized")

    
  def start(self):
    self.thread.start()
    self.Print("camera stream started")

  def add_image_to_video(self,flag,img=0):
    if flag == 0:
        if SAVE_DEBUG:
            self.video_debug.write(self.img_curr_2show)
        self.video.write(self.img_lrg)
        self.img_count = self.img_count + 1
    elif flag == 1:
        if type(img) is int:
            self.video_mov.write(self.img_lrg)
        else:
            self.video_mov.write(img)
        self.img_count_mov = self.img_count_mov + 1
    #print (time.time() - self.sysTime)*TIME_RATIO
    self.sysTime = time.time()


  def time2string(self,input_time):
      return datetime.datetime.fromtimestamp( input_time ).strftime('%Y%m%d_%H%M%S')

  def handleLastVideo(self,FlaseAlarmVec):
    global files2Handle
    global threadsLock
    if type(self.video) is int:
        return -1 
    else:
        self.video.release()       
        self.video_mov.release() 
        self.video_debug.release()
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
        if SAVE_DEBUG:
            shutil.move(self.fileName_video_debug, MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video_debug)
            
        threadsLock.acquire()
        try:
            files2Handle.name.append(MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video)
            files2Handle.nLen = files2Handle.nLen + 1  
            files2Handle.dataForNN.append(0)
            files2Handle.afterNN.append(True)              
            if SAVE_DEBUG:
                files2Handle.name.append(MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video_debug)
                files2Handle.nLen = files2Handle.nLen + 1    
                files2Handle.dataForNN.append(self.dataForNN)
                files2Handle.afterNN.append(False)                
            self.Print([files2Handle.name, files2Handle.nLen])
        finally:
            threadsLock.release() 
            
        if self.movmentDetection == 1:
            shutil.move(self.fileName_video_mov, MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video_mov)
            threadsLock.acquire()
            try:
                files2Handle.name.append(MAIN_DIR+REC_DIR_TMP+'\\'+self.filename_short_video_mov)
                files2Handle.nLen = files2Handle.nLen + 1  
                files2Handle.dataForNN.append(0)
                files2Handle.afterNN.append(False)

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
    self.filename_short_video_debug = self.timeString+'_debug.mkv'   
    self.filename_short_log = self.timeString+'_log.txt'     
    self.fileName_video = MAIN_DIR+'\\'+self.filename_short_video
    self.fileName_video_mov = MAIN_DIR+'\\'+self.filename_short_video_mov
    self.fileName_video_debug = MAIN_DIR+'\\'+self.filename_short_video_debug    
    self.filename_log = MAIN_DIR+'\\'+self.filename_short_log
    self.Print(self.fileName_video)
    if type(self.video) is not int:
        self.video.release() 
    if type(self.video_mov) is not int:
        self.video.release()         
    self.video = cv2.VideoWriter(self.fileName_video,fourcc,VID_FPS,(width,height))
    self.video_mov = cv2.VideoWriter(self.fileName_video_mov,fourcc,VID_FPS,(width,height))      
    self.video_debug = cv2.VideoWriter(self.fileName_video_debug,fourcc,VID_FPS,(int(width/RESIZE_ORG),int(height/RESIZE_ORG)))          
    self.add_image_to_video(0)
    self.add_image_to_video(1)
    fid = open(self.filename_log,'w')                  
    print(self.timeString, file=fid)
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
      img_tmp = cv2.blur(img_tmp,(2,2))
      
      self.img_curr = img_tmp.copy()
      self.img_curr_2show = (self.img_curr).copy()

      height , width , layers =  self.img_curr_2show.shape
      for iii,box in enumerate(BOX_WIDTH_HEIGHT):
          cv2.rectangle(self.img_curr_2show, (int(width*box[0]), int(height*box[2])), (int(width*box[1]), int(height*box[3])), (0, 0, 255), 2)
          cv2.putText(self.img_curr_2show, str(iii) , (int(width*box[0]), int(height*box[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)  


  def isIntersect(self,veci,vecj,margin):
      xi0 = veci[0]
      yi0 = veci[1]
      xi1 = veci[0] + veci[2]
      yi1 = veci[1] + veci[3]
      xj0 = vecj[0] - margin
      yj0 = vecj[1] - margin
      xj1 = vecj[0] + vecj[2] + margin
      yj1 = vecj[1] + vecj[3] + margin      
      if  (xi0 >= xj0 and xi0 <= xj1 and yi0 >= yj0 and yi0 <= yj1) or  \
          (xi1 >= xj0 and xi1 <= xj1 and yi0 >= yj0 and yi0 <= yj1) or  \
          (xi0 >= xj0 and xi0 <= xj1 and yi1 >= yj0 and yi1 <= yj1) or  \
          (xi1 >= xj0 and xi1 <= xj1 and yi1 >= yj0 and yi1 <= yj1) :
              return True
      else:
          return False
      
  def getConnectedComponents(self,bb_final_graph , seen , idx ,cnt, bb_graph):
      if len(bb_graph[idx]) == 0:
          return bb_final_graph , seen 
      for j,bbj in enumerate(bb_graph[idx]):
           if bbj not in seen:
               bb_final_graph[cnt].append(bbj)
               seen.append(bbj)
               bb_final_graph , seen = self.getConnectedComponents(bb_final_graph , seen , bbj ,cnt, bb_graph)
      return bb_final_graph , seen 
  
  def isInBoundindBox(self,COM):
      height , width , layers =  self.img_curr.shape
      vbox = (COM[0],COM[1],1,1 )
      for i,box in enumerate(BOX_WIDTH_HEIGHT):
          bbox = (int(width*box[0]), int(height*box[2]), int(width*box[1]-width*box[0]), int(height*box[3]-height*box[2]))
          if self.isIntersect(vbox ,bbox,0):
              return i
      return 0
   
  def getContours(self):
    fgmask = self.fgbg.apply(self.img_curr)
    kernel = np.ones((3,3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)    
    _, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    height , width , layers =  self.img_curr.shape
    cnt = 0
    bb_list = []
    for i,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min(minBBArea):
            cnt = cnt + 1
            #cv2.drawContours(self.img_curr_2show, contour, -1, (0, 0, 255), 3)
            x,y,w,h = cv2.boundingRect(contour) 
            COM = (x+int(w/2),y+int(h/2))
            pctMinBBArea = (  ( COM[1] / height ) ) * (max(minBBArea) - min(minBBArea))  + min(minBBArea)
            #print(pctMinBBArea,area,COM)
            boxNum = self.isInBoundindBox(COM)
            if (boxNum > 0 and area > minBBArea[boxNum]) or (area > pctMinBBArea):
                bb_list.append((x,y,w,h))
            #if (cnt > 0 ):
            #    cv2.rectangle(self.img_curr_2show, (x,y), (x+w,y+h), (0, 0, 255), 4)
            #    cv2.putText(self.img_curr_2show,str(cnt-1),(x,y), font, 0.75,(255,0,0),1,cv2.LINE_AA)
                #print("------>",(x,y), (x+w,y+h))
  

    bb_mtx = np.zeros((len(bb_list),len(bb_list)))
    bb_pairs = []
    bb_graph = []
    bb_final_list = []
    bb_final_graph = []
    seen = []
    if len(bb_list) > 1:
        for i,bbi in enumerate(bb_list):
            for j,bbj in enumerate(bb_list):
                if len(bb_graph) == i:
                    bb_graph.append([])
                if i >= j:
                    if i == j:
                        bb_graph[i].append(j)
                    continue                    
                veci = np.array(bbi)
                vecj = np.array(bbj)
                if self.isIntersect(veci,vecj,20) or self.isIntersect(vecj,veci,20):
                        bb_mtx[(i,j)] = 1
                        bb_mtx[(j,i)] = 1
                        bb_pairs.append((i,j))
                        bb_graph[i].append(j)
        #print("--->",bb_graph)

        cnt = 0
        for i,bbi in enumerate(bb_graph):
            if bbi[0] not in seen:
                if len(bb_final_graph) == cnt:
                    bb_final_graph.append([])
                bb_final_graph[cnt].append(bbi[0])
                seen.append(bbi[0])                
                [bb_final_graph , seen] = self.getConnectedComponents(bb_final_graph , seen , i , cnt , bb_graph)
                cnt = cnt + 1
        
        #print(bb_final_graph)
        bb_list_np = np.array(bb_list)

        #print(bb_list_np)
        for i,bbi in enumerate(bb_final_graph):
            x = np.min(bb_list_np[bbi,0])
            y = np.min(bb_list_np[bbi,1])
            x1 = np.max(bb_list_np[bbi,0]+bb_list_np[bbi,2])
            y1 = np.max(bb_list_np[bbi,1]+bb_list_np[bbi,3])
            w = x1 - x
            h = y1 - y
            bb_final_list.append((x,y,w,h,w*h))
            #print("--x-->",x,y,x1,y1)
            cv2.rectangle(self.img_curr_2show, (x,y), (x1,y1), (0, 255, 0), 1)
        '''
        if len(bb_list) > 6:
            while True:
                cv2.imshow('cam',self.img_curr_2show)
                if cv2.waitKey(1) == 999:
                    exit(0)                    
        '''
    return fgmask, sorted(bb_final_list,key=lambda x: x[4],reverse=True),contours

  def get_relevant_bb(self):
      rel_bb = []
      for bbox in self.bb_final_list:
            COM = (bbox[0]+int(bbox[2]/2),bbox[1]+int(bbox[3]/2))
            boxNum = self.isInBoundindBox(COM)
            if boxNum > 0:
               rel_bb.append( (boxNum,bbox[0],bbox[1],bbox[2],bbox[3]) )
      return rel_bb

  def isFalseAlarm(self):
      if len(self.bb_final_list) == 0:
          self.detectionData.append( (time.time() , 0 , 0 , 0, 0 ,0 , 0 , []) )
          if len(self.detectionData) > DETECT_DATA_LEN_MAX:
              self.detectionData.remove(self.detectionData[0])          
          return -1
      height , width , layers =  self.img_curr.shape
      maxImageArea = height*width
      sumAllAreas = sum([x[-1] for x in self.bb_final_list])
      sumNear = sum([x[-1]*((x[1]+x[3]/2) > height/2) for x in self.bb_final_list])
      sumFar = sum([x[-1]*((x[1]+x[3]/2) < height/2) for x in self.bb_final_list])
      maxSingleArea = self.bb_final_list[0][-1]
      xMin = min([x[0] for x in self.bb_final_list])
      yMin = min([x[1] for x in self.bb_final_list])
      xMax = max([x[0]+x[2] for x in self.bb_final_list])
      yMax = max([x[1]+x[3] for x in self.bb_final_list])      
      maxAllAreas = (xMax - xMin) * (yMax - yMin)
      
      rel_bb = self.get_relevant_bb()
      
      #print(sumAllAreas,maxSingleArea,maxAllAreas,sumNear,sumFar)
      self.detectionData.append( (time.time() , 100.0*sumAllAreas/maxImageArea , 100.0*sumNear/(maxImageArea/2) , 100.0*sumFar/(maxImageArea/2) , 100.0*maxSingleArea/maxImageArea , 100*maxAllAreas/maxImageArea , len(self.bb_final_list), rel_bb) )
      #print(self.detectionData[-1])
      
      #print(len(self.detectionData))
      
      if len(self.detectionData) > DETECT_DATA_LEN_MAX:
          self.detectionData.remove(self.detectionData[0])
      
      if len(self.detectionData) > DETECT_DATA_LEN_MIN:
          negCount = 0
          rel_bb_cnt = np.zeros(len(BOX_WIDTH_HEIGHT))
          for data in self.detectionData:
              if data[1] is not 0:
                  if data[1] > SUMALLAREAS_MAX:
                      negCount = negCount + 1
                      self.Print(("FA #1 - ",data[1]))
                  if data[2] > 0 and data[2] < SUMNEAR_MIN:
                      negCount = negCount + 1              
                      self.Print(("FA #2 - ",data[2]))                   
                  if data[3] > SUMFAR_MAX:
                      negCount = negCount + 1 
                      self.Print(("FA #3 - ",data[3]))                    
                  if data[4] > MAXSINGLEAREA_MAX:
                      negCount = negCount + 1  
                      self.Print(("FA #4 - ",data[4]))                     
                  if data[5] > MAXALLAREA_MAX:
                      negCount = negCount + 1       
                      self.Print(("FA #5 - ",data[5]))
                  if data[6] > LEN_BB_MAX:
                      negCount = negCount + 1       
                      self.Print(("FA #6 - ",data[6]))                                  
                  
                  used = []
                  for relData in data[7]:
                      if sum([x == relData[0] for x in used]) > 0:
                          continue
                      else:
                          rel_bb_cnt[relData[0]] = rel_bb_cnt[relData[0]] + 1
                          used.append(relData[0])
                          
                      
              if negCount > NEG_MAX:
                  cv2.putText(self.img_curr_2show, "False Alarm" , (int(width*0.05), int(height*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)  
                  self.Print("False Alarm",2)
                  return 1
              
          if rel_bb_cnt[rel_bb_cnt > POS_MAX].any():
              cv2.putText(self.img_curr_2show, "Positive Detection" , (int(width*0.05), int(height*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)  
              self.Print("Positive Detection",2)
              return 0                 
          
      #self.Print("UNKNOWN")
      return -1  
      #print(self.detectionData[-1] )

   
      
  def run(self):
    bytes=''
    self.img_last_sum = -1  
    #tic = time.time()
    self.fgbg = cv2.createBackgroundSubtractorMOG2()
    #self.trackers,self.trackersValid,self.trackersLastTime,self.trackerBB = self.initTrackers()
    self.bb_final_list = []
    FlaseAlarmVec = []
    self.posImgBuffer = []
    self.dataForNN = []
    while not self.thread_cancelled:
      try:
        ### Init Video after X seconds
        if self.img_count == 1:
            self.handleLastVideo(FlaseAlarmVec)
            self.initVideo()
            self.dataForNN = []
            self.movmentDetection = 0
            if len(FlaseAlarmVec) > 0:
                FlaseAlarmVec = []
        
        ### get new Image
        #print(time.time() - tic)
        stream_new_flag , self.img_lrg , bytes = self.getImage(bytes)
        #tic = time.time()
      
        if stream_new_flag == 1:
          
          self.updateImages()

          #print(len(contours))
          #tic = time.time()
          self.bb_final_list_last = self.bb_final_list.copy()
          fgmask, self.bb_final_list,contours = self.getContours()
          isFalseAlarm = self.isFalseAlarm()
          FlaseAlarmVec.append(isFalseAlarm)
          if len(self.bb_final_list) > 0:
              if not isFalseAlarm:
                  self.posImgBuffer.append(self.img_lrg)
              else:
                  self.posImgBuffer = []

              
          ### Show image, save to video, and find issues
          if self.img_count > 1:
              if DEBUG: 
                  cv2.imshow('cam',self.img_curr_2show)
                  #if flag_NN:
                  #    cv2.imshow('cam_net',img_2show_net)
                  #cv2.imshow('cam_MOG',fgmask)
                 
                  #cv2.moveWindow('cam',)
                  #cv2.imshow('camLast',self.img_last)

              
              if len(self.posImgBuffer) > 0:
                  if len(self.posImgBuffer) > POS_DETECT_MIN_BUFFER or type(self.posImgBuffer[0]) is int:
                      self.movmentDetection = 1
                      BW_cnt = 0
                      for img in self.posImgBuffer:
                          if type(img) is not int:
                              if BW_cnt < len(self.detectionData):
                                  #print(BW_cnt)
                                  #print(len(self.detectionData))
                                  #print(self.detectionData[-(1+BW_cnt)])
                                  self.dataForNN.append( (self.timeString, self.img_count,self.img_count_mov, self.detectionData[-(1+BW_cnt)][7] ) )
                              else:
                                  self.Print("dataForNN --> SOMETHING WRONG!!")
                              self.add_image_to_video(1,img)
                              BW_cnt = BW_cnt + 1
                      self.posImgBuffer = []
                      self.posImgBuffer.append(0)
                      
              self.add_image_to_video(0)
          
          ### first time (no "diff" image)
          if self.img_count == -1:
              self.img_count = 1
              self.img_count_mov = 1
          
          ### Init Video after X seconds
          if time.time() - self.firstSysTime > MAX_SINGLE_VIDEO_TIME:
              self.img_count = 1
              self.img_count_mov = 1 
              
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

''' ### Trackers ###
  def createTracker(self,tracker_type):

        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()      
        return tracker
    
  def initTrackers(self):

    trackers = []
    trackersValid = np.zeros(10)
    trackersLastTime = np.zeros(10)
    trackerBB = []
    for iii in range(0,numOfTrackers):
        trackers.append(self.createTracker(tracker_type))
        trackerBB.append((0,0,0,0))
    return trackers,trackersValid,trackersLastTime,trackerBB

  def isBBUnique(self,count):
     bbox = self.bb_final_list[count][0:4]
     for iii,tracker in enumerate(self.trackers):
          if self.trackersValid[iii]:
             bbox2 = self.trackerBB[iii]
             if self.isIntersect(bbox,bbox2,0) or self.isIntersect(bbox2,bbox,0):
                 #print(bbox,bbox2,"-->TRUE")
                 return False
             #else:
                 #print(bbox,bbox2,"-->FALSE")
     return True
              
          
  def getTrackingSpeed(self, bboxN,bboxO,timeN,timeO):
    centerXN = bboxN[0] + bboxN[2]/2
    centerXO = bboxO[0] + bboxO[2]/2
    centerYN = bboxN[1] + bboxN[3]/2
    centerYO = bboxO[1] + bboxO[3]/2  
    dt = 1#timeN - timeO
    dPixel = np.array((centerXN - centerXO , centerYN - centerYO))
    #print("-----",bboxN,bboxO,dPixel)
    return np.linalg.norm(dPixel) / dt
 
  def setUpdateTrackers(self):
    count = 0
    for iii,tracker in enumerate(self.trackers):
        lastTime = self.trackersLastTime[iii]
        isValid = self.trackersValid[iii]
        #print(iii,lastTime,isValid,len(bb_final_list))
        if ((time.time() - lastTime) < trackerWorkTime) and (isValid > 0):
            ok, bbox = tracker.update(self.img_curr)
            #print(str(iii)," --> " ,self.trackerBB[iii] , bbox)
            trackVel = self.getTrackingSpeed(bbox,self.trackerBB[iii],time.time(),lastTime)
            if trackVel >= 5:
                self.trackersValid[iii] = self.trackersValid[iii] + 1
            else:
                self.trackersValid[iii] = self.trackersValid[iii] - 1
            if self.trackersValid[iii] > 10:
                print("################################## MOVEMENT  " + str(iii) +"#################################")
            self.trackerBB[iii] = bbox
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(self.img_curr_2show, p1, p2, (255,0,0), 2, 1)
                cv2.putText(self.img_curr_2show, str(iii) + " | " + str(int(trackVel))  , p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)  
            else :
                # Tracking failure
                self.trackersValid[iii] = self.trackersValid[iii] - 2
                cv2.putText(self.img_curr_2show, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)  
        else:
            while count < len(self.bb_final_list) and self.isBBUnique(count) == False:
                count = count + 1
            print(iii,count)
            if count >= len(self.bb_final_list):
                break            
            self.trackers[iii].clear()
            self.trackers[iii] = self.createTracker(tracker_type)
            ok = self.trackers[iii].init(self.img_curr, self.bb_final_list[count][0:4])
            if ok:
                self.trackersValid[iii] = 1
                self.trackersLastTime[iii] = time.time()
                self.trackerBB[iii] = self.bb_final_list[count][0:4]
                count = count + 1

            else: 
                self.trackersValid[iii] = 0

            
                
    return True 
'''

class FileHandler():

  def Print(self,string,var=0):
      string = str(time.time())+"\t"+str(string)
      if var == 0:
          print(string)
          fid = open(mainLogFilename_FH,'a+')
          print(string, file=fid)
          fid.close()
      elif var == 1:
          print(string)
      elif var == 2:
          fid = open(mainLogFilename_FH,'a+')
          print(string, file=fid)
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
            if nLen_local > 0 and files2Handle.afterNN[0] is True:
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
                    
                    
                    
class imageDetection():  


  def Print(self,string,var=0):
      string = str(time.time())+"\t"+str(string)
      if var == 0:
          print(string)
          fid = open(mainLogFilename_NN,'a+')
          print(string, file=fid)
          fid.close()
      elif var == 1:
          print(string)
      elif var == 2:
          fid = open(mainLogFilename_NN,'a+')
          print(string, file=fid)
          fid.close()    
          
         
  def __init__(self):
    

    self.thread_cancelled = False
    self.thread = Thread(target=self.run)
    self.Print("imageDetection initialized")
    
  def start(self):
    self.thread.start()
    self.Print("imageDetection started")
    
  def is_running(self):
    return self.thread.isAlive()
      
    
  def shut_down(self):
    self.thread_cancelled = True
    #block while waiting for thread to terminate
    while self.thread.isAlive():
      time.sleep(1)
    return True   
    
  def addTextFromTFnet(self,result,imgcv):
    font = cv2.FONT_HERSHEY_SIMPLEX

    for res in result:
        x1 = res['topleft']['x']
        y1 = res['topleft']['y']
        x2 = res['bottomright']['x']
        y2 = res['bottomright']['y']   
        label = res['label']
        conf = res['confidence']
        cv2.rectangle(imgcv,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(imgcv,label+':'+str(conf),(x1,y1), font, 0.5,(0,0,255),1,cv2.LINE_AA)
    return  imgcv

  def getMaxDist(self,veci,vecj,margin=0):
      xi0 = veci[0]
      yi0 = veci[1]
      xi1 = veci[0] + veci[2]
      yi1 = veci[1] + veci[3]
      xj0 = vecj[0] - margin
      yj0 = vecj[1] - margin
      xj1 = vecj[0] + vecj[2] + margin
      yj1 = vecj[1] + vecj[3] + margin      
      max_x = max(xi0,xi1,xj0,xj1)
      min_x = min(xi0,xi1,xj0,xj1)
      max_y = max(yi0,yi1,yj0,yj1)
      min_y = min(yi0,yi1,yj0,yj1)
      dx = max_x - min_x
      dy = max_y - min_y
      return np.sqrt(dx**2+dy**2), (min_x,min_y,max_x-min_x,max_y-min_y )

      
      
      
  def getNNBB(self,bb_all):
      used = []
      NN_bb = []
      for i,bbi in enumerate(bb_all):
          if sum([x == i for x in used]) > 0:
              continue
          else:    
              used.append(i)
              bbTmp = bbi[1:]
          for j,bbj in enumerate(bb_all):
              if i == j:
                    continue
              if sum([x == j for x in used]) > 0:
                    continue
              else:
                    bbTmp_j = bbj[1:]
                    print(i,j,bbTmp , bbTmp_j)
                    dist,new_bbox = self.getMaxDist(bbTmp,bbTmp_j)
                    if dist < self.netImageSize:
                        bbTmp = new_bbox
                        used.append(j)
          NN_bb.append(bbTmp)
      return NN_bb    
          
  def runImageDetection(self,vid_name,iii):

      self.stream = cv2.VideoCapture(vid_name)
      VID_FPS = int(self.stream.get(cv2.CAP_PROP_FPS))
      width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
      tempName = vid_name.split('.mkv')[0]+"_TMP.mkv"
      fourcc = cv2.VideoWriter_fourcc(*'XVID')
      videoWrite = cv2.VideoWriter(tempName,fourcc,VID_FPS,(width,height))
      imgCnt_vid = 0
      for data in self.files2Handle_local.dataForNN[iii]:
        ret, img_tmp = self.stream.read()
        if ret is True:
            imgCnt = data[1]
            #print("--->",data)
            #imgCnt_mov = self.files2Handle_local.dataForNN[iii][2]
            rel_bb = data[3]     # data[3] = (bbNum,(x,y,w,h))         
            while (ret is True) and (imgCnt_vid < imgCnt):
                videoWrite.write(img_tmp)
                imgCnt_vid = imgCnt_vid + 1
                ret, img_tmp = self.stream.read()
            imgCnt_vid = imgCnt_vid + 1 # for next time
            if ret is False:
                continue
            rel_bb_sorted = sorted(rel_bb,key=lambda x: x[1],reverse=False)  
            NN_bb = self.getNNBB(rel_bb_sorted)
            for bbox in NN_bb:

              imageSize_NN = self.netImageSize
              x = bbox[0]
              y = bbox[1]              
              w = bbox[2]
              h = bbox[3]
              if w <= imageSize_NN and h <= imageSize_NN:
                  tic = time.time()
                  gapx = int((imageSize_NN - w)/2)
                  gapy = int((imageSize_NN - h)/2)
                  height , width , layers =  img_tmp.shape
                  if x-gapx < 0:
                      x0 = 0
                      x1 = imageSize_NN
                  elif x+w+gapx >= width:
                      x0 = width - imageSize_NN
                      x1 = width 
                  else:
                      x0 = x-gapx
                      x1 = x+w+gapx 
                  if y-gapy < 0:
                      y0 = 0
                      y1 = imageSize_NN
                  elif y+h+gapy >= height:
                      y0 = height - imageSize_NN
                      y1 = height 
                  else:
                      y0 = y-gapy
                      y1 = y+h+gapy                                
                  #print(x0,x1,y0,y1,self.img_curr.shape)
                  imgcv = img_tmp[y0:y1,x0:x1,:]#[x0:x1,y0:y1,:]
                  result = self.tfnet.return_predict(imgcv)
                  print('netTime = '+str(time.time()-tic))
                  img_2show_net = self.addTextFromTFnet(result,imgcv)
                  img_tmp[y0:y1,x0:x1,:] = img_2show_net
                  videoWrite.write(img_tmp)
              else:
                  self.Print("bbox size -- ERROR! ")
            cv2.imshow('NN',img_tmp)
        if cv2.waitKey(1) == 999:
            exit(0)
      videoWrite.release()    
                          

  def run(self):
      global files2Handle
      global threadsLock   
      from darkflow.net.build import TFNet
      options = {"model": ".//cfg//tiny-yolo.cfg", "load": ".//bin//tiny-yolo.weights", "threshold": 0.3}
      self.tfnet = TFNet(options)      
      self.netImageSize = 416

      while not self.thread_cancelled:        
            threadsLock.acquire()
            try:
                if files2Handle.nLen > 0:
                    self.files2Handle_local = files2Handle
                else:
                    self.files2Handle_local = 0
            finally:
                threadsLock.release()   
            
            flagWorked = False
            if type(self.files2Handle_local) is not int:
                for iii in range(0,self.files2Handle_local.nLen):
                    if self.files2Handle_local.afterNN[iii] == False:
                        if type(self.files2Handle_local.dataForNN[iii]) is not int:
                            if  len(self.files2Handle_local.dataForNN[iii]) > 0 :
                                vid_name = self.files2Handle_local.name[iii]
                                self.runImageDetection(vid_name,iii)
                                                
                                flagWorked = True
                        
                        threadsLock.acquire()
                        try:
                            for jjj in range(0,files2Handle.nLen):
                                if files2Handle.name[jjj] is self.files2Handle_local.name[iii]:
                                    files2Handle.afterNN[jjj] = True
                                    break
                        finally:
                            threadsLock.release()  
            
            if flagWorked is False:
                time.sleep(10)
                    
  
    
if __name__ == "__main__":
    #print time.time()    
    #time.sleep(3600*3)   
    parser = ConfigParser()
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
    #_FileHandler = FileHandler()
    #_FileHandler.start()
    _imageDetection = imageDetection()
    _imageDetection.start()    