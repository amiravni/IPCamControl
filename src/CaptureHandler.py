from threading import Thread
from queue import Queue
import time
from config import *

class CaptureHandler:
	def __init__(self, path, stream_type='rtsp', sim_start_frame=0, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		#self.stream_type = stream_type
		self.stream = cv2.VideoCapture(path)
		self.delay_between_frames = 0.0
		if stream_type == 'rtsp':
			self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
		elif stream_type == 'sim':
			self.stream.set(cv2.CAP_PROP_POS_FRAMES, sim_start_frame)
			self.delay_between_frames = 1/VID_FPS
		self.stopped = False
		self.thread = None
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)



	def start(self):
		# start a thread to read frames from the file video stream
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()
		return self

	def is_alive(self):
		if self.thread:
			return self.thread.is_alive()
		else:
			return False

	def update(self):
		# keep looping infinitely
		frame_time = time.time()
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				if self.delay_between_frames > 0:
					dt = self.delay_between_frames - (time.time() - frame_time)
					if dt > 0:
						time.sleep(dt)
				(grabbed, frame) = self.stream.read()
				frame_time = time.time()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)
				if self.Q.qsize() > 3:
					LOGGER.warn('CAPTURE: Frame Queue size is {}'.format(str(self.Q.qsize())))

	def read(self, timeout=5):
		# return next frame in the queue
		return self.Q.get(timeout=timeout)


	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


if __name__ == '__main__':
	from config import *

	CH = CaptureHandler(IPCAM_CONFIG['rtspurl']).start()
	while True:
		time.sleep(10000)
