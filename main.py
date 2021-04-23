
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
from input_retrieval import *
from sort import *

list1=[]
tracker = Sort()
memory = {}
line = [(43, 500), (1200, 500)]
counter = 0
trucks = 0
motorcycle = 0
car = 0
bus = 0
#All these classes will be counted as 'vehicles'
list_of_vehicles = ["bicycle","car","motorbike","bus","truck"]
# Setting the threshold for the number of frames to search a vehicle for
FRAMES_BEFORE_CURRENT = 10  
inputWidth, inputHeight = 416, 416

#Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath,\
	preDefinedConfidence, preDefinedThreshold, USE_GPU= parseCommandLineArguments()

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

def displayVehicleCount(frame, vehicle_count):
	cv2.putText(
		frame, #Image
		'Detected Vehicles: ' + str(vehicle_count), #Label
		(20, 20), #Position
		cv2.FONT_HERSHEY_SIMPLEX, #Font
		0.8, #Size
		(0, 0xFF, 0), #Color
		2, #Thickness
		cv2.FONT_HERSHEY_COMPLEX_SMALL,
		)

def boxAndLineOverlap(x_mid_point, y_mid_point):
	x1_line, y1_line, x2_line, y2_line = 43,500,1200,500

	if (x_mid_point >= x1_line-10 and x_mid_point <= x2_line+10) and\
		(y_mid_point >= y1_line-10 and y_mid_point <= y2_line+10):
		return True
	return False


def displayFPS(start_time, num_frames):
	current_time = int(time.time())
	if(current_time > start_time):
		num_frames = 0
		start_time = current_time
	return start_time, num_frames

# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			if (LABELS[classIDs[i]] in list_of_vehicles):			
			# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				if LABELS[classIDs[i]] in list_of_vehicles:
				# draw a bounding box rectangle and label on the frame
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
					text = "{}".format(LABELS[classIDs[i]])
					cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			

def initializeVideoWriter(video_width, video_height, videoStream):
	# Getting the fps of the source video
	sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
	# initialize our video writer
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
		(video_width, video_height), True)

def boxInPreviousFrames(previous_frame_detections, current_box, current_detections):
	centerX, centerY, width, height = current_box
	dist = np.inf #Initializing the minimum distance
	# Iterating through all the k-dimensional trees
	for i in range(FRAMES_BEFORE_CURRENT):
		coordinate_list = list(previous_frame_detections[i].keys())
		if len(coordinate_list) == 0: # When there are no detections in the previous frame
			continue
		# Finding the distance to the closest point and the index
		temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
		if (temp_dist < dist):
			dist = temp_dist
			frame_num = i
			coord = coordinate_list[index[0]]

	if (dist > (max(width, height)/2)):
		return False

	# Keeping the vehicle ID constant
	current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
	return True

def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
	current_detections= {}
	global list1    
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indices we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
		
			
			centerX = x + (w//2)
			centerY = y+ (h//2)
			if (LABELS[classIDs[i]] in list_of_vehicles):
				current_detections[(centerX, centerY)] = vehicle_count 
				if (not boxInPreviousFrames(previous_frame_detections, (centerX, centerY, w, h), current_detections)):
					vehicle_count += 1

				ID = current_detections.get((centerX, centerY))
				# If there are two detections having the same ID due to being too close, 
				# then assign a new ID to current detection.
				if (list(current_detections.values()).count(ID) > 1):
					current_detections[(centerX, centerY)] = vehicle_count
					vehicle_count += 1 

				#Display the ID at the center of the box
				text = "ID:{}".format(ID)
				cv2.putText(frame, text, (centerX, centerY),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)                

			
			


			# When the detection is in the list of vehicles, AND
			# it crosses the line AND
			# the ID of the detection is not present in the vehicles

			if (LABELS[classIDs[i]] in list_of_vehicles):
				if boxAndLineOverlap(centerX, centerY) and ID not in list1:

					list1.append(ID)                 
					global car
					global bus
					global trucks
					global motorcycle
					global counter
					if LABELS[classIDs[i]]=='car':
						car += 1
					if LABELS[classIDs[i]]=='truck':
						trucks += 1
					if LABELS[classIDs[i]]=='bus':
						bus += 1	
					if LABELS[classIDs[i]]=='motorbike':
						motorcycle += 1
					if LABELS[classIDs[i]]=='bicycle':
						motorcycle += 1	
					counter=car+trucks+bus+motorcycle
			motorcount = "Motorcycle : {0}".format(motorcycle)
			carcount = "car : {0}".format(car)
			truckcount = "truck : {0}".format(trucks)
			buscount = "bus : {0}".format(bus)
			totalcount = "total count : {0}".format(counter)
			cv2.putText(frame, str(motorcount), (10,25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
			cv2.putText(frame, str(carcount), (10,50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
			cv2.putText(frame, str(truckcount), (10,75), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
			cv2.putText(frame, str(buscount), (10,100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
			cv2.putText(frame, str(totalcount), (10,125), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)

	return vehicle_count, current_detections

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Using GPU if flag is passed
if USE_GPU:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
print(ln)
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Initialization
previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
# loop over frames from the video file stream
while True:
	print("new frame")
	
	num_frames+= 1
	# Initialization for each iteration
	boxes, confidences, classIDs = [], [], [] 
	vehicle_crossed_line_flag = False 

	#Calculating fps each second
	start_time, num_frames = displayFPS(start_time, num_frames)
	# read the next frame from the file
	(grabbed, frame) = videoStream.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for i, detection in enumerate(output):
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > preDefinedConfidence:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
                
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
		preDefinedThreshold)

	# Draw detection box 
	drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

	vehicle_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)


    
	dets = []

	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			dets.append([x, y, x+w, y+h, confidences[i]])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	dets = np.asarray(dets)
	tracks = tracker.update(dets)

	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = {}

	for track in tracks:
		boxes.append([track[0], track[1], track[2], track[3]])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]

	# draw line
	cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

	writer.write(frame)

	cv2.imshow('Frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break	
	
	# Updating with the current frame detections
	previous_frame_detections.pop(0) #Removing the first frame from the list
	# previous_frame_detections.append(spatial.KDTree(current_detections))
	previous_frame_detections.append(current_detections)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
videoStream.release()
