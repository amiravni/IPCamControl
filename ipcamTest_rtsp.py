# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:33:38 2017

@author: PC-BRO
"""
import numpy as np
import cv2 as cv
import time
import datetime
import requests
import threading
from threading import Thread, Event, ThreadError


vcap = cv.VideoCapture("rtsp://192.168.14.150:5433/11")

while(1):

    ret, frame = vcap.read()
    cv.imshow('VIDEO', frame)
    cv.waitKey(10)