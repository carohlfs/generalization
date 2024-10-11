import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load the configuration and pretrained model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

# Set the device to CPU
cfg.MODEL.DEVICE = 'cpu'

predictor = DefaultPredictor(cfg)

# Load and preprocess the image
image_path = 'julia.png'  # Replace with your image path
image = cv2.imread(image_path)

# Check if image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Image at path {image_path} could not be loaded. Please check the file path.")

outputs = predictor(image)

# Visualize the results
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save and display the result
result_image = out.get_image()[:, :, ::-1]
cv2.imwrite('julia_labeled_test.png', result_image)
