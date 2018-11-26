import csv
import numpy as np
import matplotlib.pyplot as plt
import sys, getopt


# Method that takes the csv file as input and return the informations about the bounding boxes per frame
def get_raw_boxes_information(csv_filename) :
    # Define the output
    boxes_output = []
    box_per_frame = []

    # Open the file
    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile)
        # Loop over the rows in the file
        for row in reader :
            # Get the number of detection for a given row
            num_detect = int(row['num_detections'])
            box_per_frame.append(num_detect)
            #print(num_detect)

            # Get the detection scores of those detections
            detect_score = row['detection_scores'][1:-1].split()[:num_detect]
            # Convert the string to a float
            detect_score = [float(item) for item in detect_score]
            #print(detect_score)

            # Get the classes of those detections
            classes = row['detection_classes'][1:-1].split()[:num_detect]
            # Convert the string to int
            classes = [int(item) for item in classes]
            #print(classes)

            # Get the bounding boxes positions
            xmin = []
            ymin = []
            xmax = []
            ymax = []
            bounding_box = row['detection_boxes'][2:-2]
            tmp_values_allboxes = bounding_box.split(']\n [')
            for val in tmp_values_allboxes[:num_detect]:
                ymin.append(float(val.split()[0]))
                xmin.append(float(val.split()[1]))
                ymax.append(float(val.split()[2]))
                xmax.append(float(val.split()[3]))

            # append the data
            boxes_output.append([detect_score, classes, xmin, ymin, xmax, ymax])

    
    return boxes_output, box_per_frame
    

# perform non max suppression on boxes to remove double detections in each frame
def non_max_suppression(boxes_info, iou_threshold = 0.8):
    output_boxes = []
    # loop over all frames
    for frame in boxes_info :
        # initialize the list of picked indexes	
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = np.asarray(frame)[2,:]
        y1 = np.asarray(frame)[3,:]
        x2 = np.asarray(frame)[4,:]
        y2 = np.asarray(frame)[5,:]
        
         
        # compute the area of the bounding boxes
        area = (x2 - x1) * (y2 - y1)
        # and sort the bounding by detection score
        idxs = np.argsort(frame[0])
        
        # keep looping while some indices still remain in the index list
        while len(idxs) > 0:
            # grab the last index in the indices list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the intersection area
            intersect = (xx2-xx1)*(yy2-yy1)
            # compute the union area
            union = (x2[i]-x1[i])*(y2[i]-y1[i]) - intersect + (x2[idxs[:last]]-x1[idxs[:last]])*(y2[idxs[:last]]-y1[idxs[:last]])
            # compute the iou
            iou = intersect / union
            # delete all indexes from the index list that have an iou greater
            # than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                   np.where(iou > iou_threshold)[0])))
            
        # return only the bounding boxes that were picked using the
        # integer data type
        #print(pick)
        score = []
        classes = []
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for i in pick:
            score.append(frame[0][i])
            classes.append(frame[1][i])
            xmin.append(frame[2][i])
            ymin.append(frame[3][i])
            xmax.append(frame[4][i])
            ymax.append(frame[5][i])
        output_boxes.append([score,classes,xmin,ymin,xmax,ymax])
        #output_boxes.append(boxes_info[pick])
        #for box in boxes_info[pick] :
        #    output_boxes.append(box)
    return output_boxes

# Method that returns the number of human boxes per frame for a given threshold over detection confidence
def get_human_boxes_per_frame(boxes_info, detect_threshold = 0.5, size_threshold = 0.0) :
    human_per_frame = []
    # loop over all frames
    for frame in boxes_info :
        # define a mean to zero
        count = 0
        # loop over all boxes in that frame
        for i in range(len(frame[0])):
            # check the score is greather than the threshold and the class is human
            if frame[0][i] > detect_threshold and frame[1][i]==1:
                # if a size threshold is given check the box is larger
                if size_threshold > 0.:
                    size = (frame[4][i]-frame[2][i])*(frame[5][i]-frame[3][i])
                    if size > size_threshold:
                        count = count+1
                else :
                    count = count+1
        human_per_frame.append(count)        
            
    return human_per_frame


# Method that returns the mean human boxes size per frame for a given threshold over detection confidence
def get_mean_human_boxes_size_per_frame(boxes_info, detect_threshold = 0.5) :
    mean_box_size = []
    # loop over all frames
    for frame in boxes_info :
        # define a mean to zero
        mean = 0
        count = 0
        # loop over all boxes in that frame
        for i in range(len(frame[0])):
            # check the score is greather than the threshold and the class is human
            if frame[0][i] > detect_threshold and frame[1][i]==1:
                mean = (frame[4][i]-frame[2][i])*(frame[5][i]-frame[3][i])
                count += 1
        if count > 0:
            mean_box_size.append(mean/count)        
            
    return mean_box_size

# Method that computes the histogram of the human boxes sizes
def get_human_boxes_size_hist(boxes_info, detect_threshold = 0.5):
    human_box_size = []
    # loop over all frames
    for frame in boxes_info :
        # loop over all boxes in that frame
        for i in range(len(frame[0])):
            # check the score is greather than the threshold and the class is human
            if frame[0][i] > detect_threshold and frame[1][i]==1:
                human_box_size.append((frame[4][i]-frame[2][i])*(frame[5][i]-frame[3][i]))
            
    return human_box_size

# Method that computes the histogram of the detection scores
def get_detection_score_hist(boxes_info):
    detect_score_hist = []
    # loop over all frames
    for frame in boxes_info :
        # loop over all boxes in that frame
        for i in range(len(frame[0])):
            detect_score_hist.append(frame[0][i])
            
    return detect_score_hist

# method that computes the intersection over union of two boxes
def get_iou(box1, box2):
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_area = (xi2-xi1) * (yi2-yi1) 

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area

    return iou

# smoothing curve with rounding 
def curve_smoothing(y, width) :
    yhat = y.copy()
    for i in range(len(y)) :
        low = max(0,int(i-width/2 + 1))
        high = min(len(y)-1, int(i+width/2))
                  
        mean = np.mean(y[low:high])
        yhat[i] = int(round(mean))
        
    return yhat

# method that takes as input a bunch of values and computes the distance
# to the mean for a given width (e.g. 1 second)
def compute_average_distance_to_mean(y, width, mean=0):
    if mean==0:
        round_mean_val = round(np.mean(y))
    else :
        round_mean_val = round(mean)
    output = np.zeros(int(len(y)/width))
    for i in range(int(len(y)/width)):
        output[i] = (np.mean(y[i*width:min((i+1)*width,len(y)-1)]))
        #print(output[i]-round_mean_val)
        
    return (output-round_mean_val)

def gauss(x, mu, sigma):
    return np.exp(-np.square(x-mu)/(2*sigma*sigma))

def logistic(x, alpha):
    return 1./(1+np.exp(-alpha*x))

# method that takes an input CSV with bounding box data and outputs
# a value for each second (a higher value means less likely to be a good frame)
def compute_frame_classification(csv_filename, video_duration = 30, size_threshold = 0.005, detect_threshold = 0.7, iou_threshold=0.7):
    # Get the raw boxes from the CSV
    raw_boxes_output, box_per_frame = get_raw_boxes_information(csv_filename)
    # Compute the average number of frame per second
    frame_per_sec = int(len(box_per_frame)/video_duration)
    print('frame per second:', frame_per_sec)
    # Compute non max suppression
    boxes_output = non_max_suppression(raw_boxes_output, iou_threshold = iou_threshold)
    # Compute the number of humans per frame
    humans_per_frame = get_human_boxes_per_frame(boxes_output, detect_threshold = detect_threshold, size_threshold = size_threshold)
    # Compute the global mean humans per frame (with higher size threshold)
    global_mean = np.mean(get_human_boxes_per_frame(boxes_output, detect_threshold = detect_threshold, size_threshold = 0.02))
    print(global_mean)
    # Compute the distance
    dist = compute_average_distance_to_mean(humans_per_frame, width = frame_per_sec, mean = global_mean)

    return dist

# method that saves the distance to an output CSV file
def save_frame_classification(distance, output_filename):
    time = np.arange(len(dist))
    output = np.stack((time, dist), axis=1)
    np.savetxt(output_filename, output, delimiter=",")




# Read the parameters of the script and check they are OK
inputfile = ''
outputfile = ''
video_duration = -1
size_threshold = 0.005
detect_threshold = 0.7
iou_threshold = 0.7

try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:v:s:d:iou")
except getopt.GetoptError:
    print('usage: computeDistance.py -i <inputfile> -o <outputfile> -v <video_duration> [-s <size_threshold> -d <detection_threshold> -iou <iou_threshold>]')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('usage: computeBoundingBoxes.py -i <inputfile> -s <start time (s)> -e <end time (s)> [-o <outputfile>]')
        sys.exit()
    elif opt in ("-i"):
        inputfile = arg
    elif opt in ("-o"):
        outputfile = arg
    elif opt in ("-v"):
        video_duration = float(arg)
    elif opt in ("-s"):
        size_threshold = float(arg)
    elif opt in ("-d"):
        detect_threshold = float(arg)
    elif opt in ("-iou"):
        iou_threshold = float(arg)

if inputfile=='' or outputfile=='' or video_duration==-1:
    print('usage: computeDistance.py -i <inputfile> -o <outputfile> -v <video_duration> [-s <size_threshold> -d <detection_threshold> -iou <iou_threshold>]')
    sys.exit()

# Compute the distance
dist = compute_frame_classification(csv_filename = inputfile, video_duration = video_duration,
                                    size_threshold = size_threshold, detect_threshold = detect_threshold,
                                    iou_threshold=iou_threshold)

# Save the distances to an output CSV
save_frame_classification(dist, outputfile)
