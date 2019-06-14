import os

allPossibleOutputLabels = ["person", "people", "person-fa", "person?"]

BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
FINAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "groundTruth")
ANNOTATIONS_FILE = os.path.join(FINAL_IMAGES_DIR, "annotations.csv")
LABELS_FILE = os.path.join(FINAL_IMAGES_DIR, "labels.csv")
CALTECH_DIR = os.path.join(DATA_DIR, "caltech")
INRIA_DIR = os.path.join(DATA_DIR, "inria")
DAIMLER_DIR = os.path.join(DATA_DIR, "daimler")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "predictions")
YOLO_DIR = os.path.join(BASE_DIR, 'keras-yolo3')

IMAGE_GENERATION_SAVE_PATH = os.path.join(IMAGES_DIR, "predicted")
TEMP_IMAGES_FOLDER = os.path.join(IMAGES_DIR, "temp")
TEMP_FOLDER = os.path.join(BASE_DIR, ".temp")

MAX_VIDEO_LENGTH = 10000000
MAX_WORKERS = 8  # for parallelization
BATCH_SPLIT = (0.2, 0.1, 0.05, 0.65)

IMAGES_READING_VERBOSE = True

USE_GLOBAL_PREDICTION_CACHE = True # increases ram usage


