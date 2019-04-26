import os

BASE_DIR = os.path.join(".")
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")
FINAL_IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(FINAL_IMAGES_DIR, "annotations.csv")
LABELS_FILE = os.path.join(FINAL_IMAGES_DIR, "labels.csv")
CALTECH_DIR = os.path.join(DATA_DIR, "caltech")
INRIA_DIR = os.path.join(DATA_DIR, "inria")
DAIMLER_DIR = os.path.join(DATA_DIR, "daimler")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "predictions")
YOLO_DIR = os.path.join(BASE_DIR, 'keras-yolo3')

MAX_VIDEO_LENGTH = 10000000
MAX_WORKERS = 1  # for parallelization


IMAGES_READING_VERBOSE = True
