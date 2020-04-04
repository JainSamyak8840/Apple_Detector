import os
import tkinter
import tkinter.filedialog
from tkinter import *
from PIL import Image
from PIL import ImageTk
from numpy import *
import cv2
import numpy as np
import tensorflow as tf

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')

NUM_CLASSES = 1
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def process(img):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image =img
    image_expanded = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
       image,
       np.squeeze(boxes),
       np.squeeze(classes).astype(np.int32),
       np.squeeze(scores),
       category_index,
       use_normalized_coordinates=True,
       line_thickness=8,
       min_score_thresh=0.80)

    return image
def select_image():
    global panelA,panelB
    path=tkinter.filedialog.askopenfilename()

    if len(path)>0:
        image=cv2.imread(path)
        edge=process(image)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(image)
        edged=Image.fromarray(edged)
        image=ImageTk.PhotoImage(image)
        edged=ImageTk.PhotoImage(edged)

    if panelA is None or panelB is None:
       panelA=Label(image=image)
       panelA.image=image
       panelA.pack(side="left",padx=10,pady=10)
       panelB=Label(image=edged)
       panelB.image=edged
       panelB.pack(side="right",padx=10,pady=10)
    else:
       panelA.configure(image=image)
       panelB.configure(image=edged)
       panelA.image=image
       panelB.image=edged


root = Tk()
panelA = None
panelB = None

btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

root.mainloop()
