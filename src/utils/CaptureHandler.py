from threading import Thread
from multiprocessing import Process, Queue, current_process
#from queue import Queue
import time
from cfg import *
import cv2

class CaptureHandlerAsProcess:
	def __init__(self, path, stream_type='rtsp',
				 sim_start_frame=0, queueSize=500,
				 name='', fps=VID_FPS, is_subprocess=False, proc_Q=None):

		self.frame_error_counter = 0
		self.queue_size = queueSize
		#self.proc_Q = Queue(maxsize=queueSize)
		#self.args = [path, stream_type, sim_start_frame, queueSize, name, fps, True, self.proc_Q]
		self.args = [path, stream_type, sim_start_frame, queueSize, name, fps, True]
		self.process = Process(name='CaptureStream', target=CaptureHandler, args=tuple(self.args))


	def start(self, restart=False):
		if restart:
			if self.frame_error_counter > 5:
				LOGGER.error('Streaming process issue for too long [network dead?] waiting 10 minutes')
				time.sleep(600)  # Wait 10 minutes
			self.process.terminate()
			#self.proc_Q = Queue(maxsize=self.queue_size)
			#self.args[-1] = self.proc_Q
			capture_queue = Queue(maxsize=self.queue_size)
			self.process = Process(name='CaptureStream', target=CaptureHandler, args=self.args)
		self.process.daemon = False
		self.process.start()
		time.sleep(2)
		return self

	def read(self, timeout=5):
		# return next frame in the queue
		try:
			#frame = self.proc_Q.get(timeout=timeout)
			frame = capture_queue.get(timeout=timeout)
			self.frame_error_counter = 0
			return frame
		except:
			try:
				self.frame_error_counter += 1
				if self.frame_error_counter > 4:
					LOGGER.error('Streaming process issue [stream stack#1]... Restarting')
					self.start(restart=True)
					return None
				LOGGER.warn('Streaming process: didnt get frame for {} seconds'.format(str(timeout)))
				#frame = self.proc_Q.get(timeout=max([0.001, 5.0-timeout]))
				frame = capture_queue.get(timeout=max([0.001, 5.0-timeout]))
				self.frame_error_counter = 0
				return frame
			except:
				LOGGER.error('Streaming process issue [stream stack#2]... Restarting')
				self.start(restart=True)
				return None

	def is_alive(self):
		return self.process.is_alive()

	def ext_watchdog(self):
		if not self.process.is_alive() or \
				capture_queue.qsize() > capture_queue._maxsize / 2:
				#self.proc_Q.qsize() > self.proc_Q._maxsize / 2:
			self.frame_error_counter += 1
			LOGGER.error('Streaming process issue... Restarting')
			self.start(restart=True)
			return False
		return True


class CaptureHandler:
	def __init__(self, path, stream_type='rtsp',
				 sim_start_frame=0, queueSize=500,
				 name='', fps=VID_FPS, is_subprocess=False, proc_Q=None):
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
			self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
			self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.sim_fps = int(self.stream.get(cv2.CAP_PROP_FPS))
		self.stopped = False
		self.thread = None
		self.process = None
		# initialize the queue used to store frames read from
		# the video file
		self.name = name
		#self.Q = Queue(maxsize=queueSize)
		if is_subprocess:
			p = current_process()
			print('Starting Process: {}, {}'.format(str(p.name), str(p.pid)))
			#self.Q = proc_Q
			self.update()



	def start(self):
		# start a thread to read frames from the file video stream
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = False
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
			#if not self.Q.full():
			if not capture_queue.full():
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
				capture_queue.put(frame)
				if capture_queue.qsize() > GENERAL['queue_warning_length']:
					LOGGER.warn('CAPTURE {}: Frame Queue size is {}'.format(str(self.name), str(capture_queue.qsize())))

				#self.Q.put(frame)
				#if self.Q.qsize() > GENERAL['queue_warning_length']:
				#	LOGGER.warn('CAPTURE {}: Frame Queue size is {}'.format(str(self.name), str(self.Q.qsize())))

	def read(self, timeout=5):
		# return next frame in the queue
		#return self.Q.get(timeout=timeout)
		return capture_queue.get(timeout=timeout)

	def is_half_full(self):
		#return self.Q.qsize() > self.Q._maxsize / 2
		return capture_queue.qsize() > capture_queue._maxsize / 2

	def more(self):
		# return True if there are still frames in the queue
		return capture_queue.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


if __name__ == '__main__':
	from cfg import *

	# proc_Q = Queue(maxsize=500)

	#process = Process(target=CaptureHandler(IPCAM_CONFIG['rtspurl']).update_process, args=(proc_Q, IPCAM_CONFIG['rtspurl']))
	# kwargs = {
	# 	'path': IPCAM_CONFIG['rtspurl'],
	# 	'is_subprocess': True,
	# 	'proc_Q': proc_Q
	# 		  }
	# process = Process(target=CaptureHandler,
	# 				  args=(IPCAM_CONFIG['rtspurl'], 'rtsp',
	# 			 0, 500, '', VID_FPS, True, proc_Q))
	# process = Process(target=CaptureHandler,
	# 				  kwargs=kwargs)
	#
	# process.daemon = False
	# process.start()

	CHAP = CaptureHandlerAsProcess(IPCAM_CONFIG['rtspurl']).start()
	while True:
		if CHAP.proc_Q.qsize() > 0:
			print(CHAP.proc_Q.qsize())
		#print(CHAP.process.is_alive())
		if CHAP.proc_Q.qsize() > 30:
			tmp = CHAP.proc_Q.get()
		time.sleep(0.05)
		CHAP.ext_watchdog()
	while True:
		time.sleep(100000)

