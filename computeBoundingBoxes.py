import numpy as np
import sys
import getopt
import tensorflow as tf
import csv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


# Read the parameters of the script and check they are OK
inputfile = ''
outputfile = ''
tmin = -1
tmax = -1

try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:s:e:")
except getopt.GetoptError:
    print('usage: computeBoundingBoxes.py -i <inputfile> -s <start time (s)> -e <end time (s)> [-o <outputfile>]')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('usage: computeBoundingBoxes.py -i <inputfile> -s <start time (s)> -e <end time (s)> [-o <outputfile>]')
        sys.exit()
    elif opt in "-i":
        inputfile = arg
    elif opt in "-o":
        outputfile = arg
    elif opt in "-s":
        tmin = int(arg)
    elif opt in "-e":
        tmax = int(arg)
if inputfile == '' or tmin == -1 or tmax == -1:
    print('usage: computeBoundingBoxes.py -i <inputfile> -s <start time (s)> -e <end time (s)> [-o <outputfile>]')
    sys.exit()  
outputCSVfile = inputfile.split('.')[0]+'_'+str(tmin)+'-'+str(tmax)+'out.csv'


def detect_videos(image_np, sess, detection_graph):
    
    with detection_graph.as_default():
        
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        # Uncomment the key 'detection_masks' to output also the masks
        for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes'#, 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(image_np, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=1)
        
    return image_np, output_dict


# Method to make a CSV from a dict
def dict_to_csv(scraped_values_dict, csv_filename): 
    my_dict = scraped_values_dict
    with open(csv_filename, 'a') as f:
        w = csv.DictWriter(f, my_dict.keys())
        if f.tell() == 0:
            w.writeheader()
            w.writerow(my_dict)
        else: 
            w.writerow(my_dict)


# Method to process image, to be used with fl_image
def process_image(image):
    
    global counter
    
    if counter % 1 == 0:
   
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_np, output_dict = detect_videos(image, sess, detection_graph) 

    counter += 1
    
    dict_to_csv(output_dict, outputCSVfile)
    
    return image


# method to load an image into a numpy array
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_MODEL = 'network_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '../models/research/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# Import the model
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Needed at the moment
counter = 0

if outputfile == '':
    clip1 = VideoFileClip(inputfile).subclip(tmin, tmax)
    new_clip = [process_image(frame) for frame in clip1.iter_frames()]
else:
    clip1 = VideoFileClip(inputfile).subclip(tmin, tmax)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(outputfile, audio=False)

