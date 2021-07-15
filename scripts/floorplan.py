"""
Train Mask R-CNN on floorplan dataset - Commercial
Written by Yasmeen George
Date: 18/03/2021
------------------------------------------------------------

Usage: check the jupyter notebook or run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 floorplan.py train --mode heads --gpu 1 --dataset=../dataset/deakin_bc_floor_plans --weights=coco

    # Resume training a model that you had trained earlier
    python3 floorplan.py train --mode heads --gpu 1 --dataset=../dataset/deakin_bc_floor_plans --weights=../models/floorplam-mrcnn.h5 --logs=../logs/floorplan20210308/

    # Train a new model starting from ImageNet pretrained weights
    python3 floorplan.py train --mode heads --gpu 1 --dataset=../dataset/deakin_bc_floor_plans --weights=imagenet
"""
import json
import skimage.draw
import cv2

from config import Config
from model import *
from utils import *

# TENSORBOARD
# %load_ext tensorboard
# %tensorboard --logdir MODEL_DIR/floorplan20210303T1257/

############################################################
#  Configurations
############################################################


class DeakinFloorplanConfig(Config):

    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "Deakin Floorplan"
    # Defines whether JSON Annotation Files are read as one file, or seperately
    MODE = "Combined"  # 'Combined' or 'Separate'

    GPU_COUNT = 3
    IMAGES_PER_GPU = 2
    LOSS_WEIGHTS = {
        "rpn_class_loss": 2,
        "rpn_bbox_loss": 1,
        "mrcnn_class_loss": 1,
        "mrcnn_bbox_loss": 1,
        "mrcnn_mask_loss": 0.2
    }
    NUM_CLASSES = 1 + 5  # Background + 5 Element Classes
    CLASSES = ["lift", "stairs", "bathroom", "door", "entrance"]

    STEPS_PER_EPOCH = 50

    VALIDATION_STEPS = 5

    DETECTION_MIN_CONFIDENCE = 0.2

    # Backbone network architecture - Supported values are: resnet50, resnet101
    #BACKBONE = "resnet50"

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    RPN_NMS_THRESHOLD = 0.9

    MEAN_PIXEL = np.array([49, 49, 49])

    MINI_MASK_SHAPE = (56, 56)

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100


class SchoolFloorplanConfig(Config):

    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "School Floorplan"
    # Defines whether JSON Annotation Files are read as one file, or seperately
    MODE = "Separate"  # 'Combined' or 'Separate'

    GPU_COUNT = 2
    IMAGES_PER_GPU = 1
    LOSS_WEIGHTS = {
        "rpn_class_loss": 2,
        "rpn_bbox_loss": 1,
        "mrcnn_class_loss": 1,
        "mrcnn_bbox_loss": 1,
        "mrcnn_mask_loss": 0.2
    }
    NUM_CLASSES = 1 + 7  # Background + Element Classes
    CLASSES = ['Bathroom', 'Bathroom Stall', 'Door (Curve)', 'Door (Double)', 'Door (Extending)', 'Door (Flat)', 'Stairs']
    #CLASSES = ['Bathroom', 'Bathroom Stall', 'Door (Curve)', 'Door (Double)', 'Door (Extending)',
    #           'Door (Flat)', 'Door (Revolving)', 'Entrance', 'Lift', 'Meeting Room', 'Stairs']

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5

    DETECTION_MIN_CONFIDENCE = 0.2

    # Backbone network architecture - Supported values are: resnet50, resnet101
    # BACKBONE = "resnet50"

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 2048
    IMAGE_MAX_DIM = 2048

    # IMAGE_MIN_SCALE = 2

    RPN_ANCHOR_SCALES = (8, 32, 64, 128, 256)

    RPN_NMS_THRESHOLD = 0.9

    MEAN_PIXEL = np.array([49, 49, 49])

    MINI_MASK_SHAPE = (56, 56)

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 300

    # Maximum number of possible regions + 10% of the maximum number of regions as a buffer. This is to account for 100% detection accuracy + some noise
    DETECTION_MAX_INSTANCES = int(MAX_GT_INSTANCES + 0.1*MAX_GT_INSTANCES)

    ######## mask_rcnn_floorplan_i6_final_1.h5 ###########
    # IMAGE_MIN_SCALE = 0
    # IMAGE_MIN_DIM = 2048
    # IMAGE_MAX_DIM = 2048
    # LOSS_WEIGHTS = (2, 1, 1, 1, 0.3)
    # BACKBONE = "resnet101"
    # [PRESET] EPOCHS = 200 in train function
    # [PRESET] LR = 0.001 in train function

    ######## mask_rcnn_floorplan_i6_final_2.h5 ###########
    # IMAGE_MIN_SCALE = 02
    # IMAGE_MIN_DIM = 2048
    # IMAGE_MAX_DIM = 2048
    # LOSS_WEIGHTS = (1, 1, 1, 1, 1)
    # BACKBONE = "resnet101"
    # [PRESET] EPOCHS = 200 in train function
    # [PRESET] LR = 0.001 in train function

    ######## mask_rcnn_floorplan_i6_final_3.h5 ###########
    # IMAGE_MIN_SCALE = 4
    # IMAGE_MIN_DIM = 2048
    # IMAGE_MAX_DIM = 2048
    # LOSS_WEIGHTS = (2, 1, 1, 1, 0.3)
    # BACKBONE = "resnet101"
    # [PRESET] EPOCHS = 200 in train function
    # [PRESET] LR = 0.001 in train function
############################################################
#  Dataset
############################################################


class FloorplanDataset(Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def load_floorplan(self, dataset_dir, subset):
        """Load a subset of the floor-plan dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Set class set based on provided config default classes
        class_set = self.config.CLASSES

        # Add Classes to inner collection
        for idx, c in enumerate(class_set):
            self.add_class("floorplan", idx + 1, c)

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        if (self.config.MODE == "Combined"):
            # Note: Annotations are created using VIA Tool 2.0
            if subset == "train":
                annotations = json.load(
                    open(os.path.join(dataset_dir, "deakin_bc_train.json")))
            elif subset == "val":
                annotations = json.load(
                    open(os.path.join(dataset_dir, "deakin_bc_val.json")))
            # don't need the dict keys
            annotations = list(annotations['_via_img_metadata'].values())
            # The VIA tool saves images in the JSON even if they don't have any
            # annotations. Skip unannotated images.
            annotations = [a for a in annotations if a['regions']]
        elif (self.config.MODE == "Separate"):
            annotations = dict()
            # Note: Annotations are created using VIA Tool 2.0
            for annotation_file in os.listdir(dataset_dir):
                if (annotation_file.endswith('.json')):
                    json_file = json.load(
                        open(os.path.join(dataset_dir, annotation_file), mode='r'))
                    annotations[annotation_file] = json_file['_via_img_metadata'][json_file['_via_image_id_list'][0]]

        # Add images
        # Note: Only loading regions where Class Name is in the above Class set
        if (self.config.MODE == "Combined"):
            for a in annotations:
                if type(a['regions']) is dict:
                    polygons = [r for r in a['regions'].values()]
                else:
                    polygons = [r for r in a['regions']]

                # load_mask() needs the image size to convert polygons to masks.
                # We must read the image to get the size since VIA doesn't include it in JSON
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "floorplan",
                    image_id=a['filename'],
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons)

        elif (self.config.MODE == "Separate"):
            for json_filename, contents in annotations.items():
                if type(contents['regions']) is dict:
                    polygons = [r for r in contents['regions'].values(
                    ) if r['region_attributes']['Class'] in class_set]
                else:
                    polygons = [r for r in contents['regions']
                                if r['region_attributes']['Class'] in class_set]

                # load_mask() needs the image size to convert polygons to masks.
                # We must read the image to get the size since VIA doesn't include it in JSON
                image_path = os.path.join(dataset_dir, contents['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                self.add_image(
                    "floorplan",
                    image_id=contents['filename'],
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        shapes = info['polygons']

        for i, p in enumerate(info['polygons']):
            shape = p['shape_attributes']['name']
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, p, 1)

        # Map class names to class IDs.
        if (self.config.MODE == "Combined"):
            class_ids = np.array([self.class_names.index(s['region_attributes']['element_type'])
                                  if 'element_type' in s['region_attributes'].keys() else self.class_names.index('door') for s in shapes])
        elif (self.config.MODE == "Separate"):
            class_ids = np.array([self.class_names.index(s['region_attributes']['Class']) if 'Class' in s['region_attributes'].keys(
            ) else self.class_names.index('Door (Curve)') for s in shapes])

        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, p, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        p = p['shape_attributes']
        if shape == 'rect':
            image = cv2.rectangle(
                image, (p['x'], p['y']), (p['x'] + p['width'], p['y'] + p['height']), color, -1)
        elif shape == "circle":
            #image = cv2.circle(image, (p['cx'], p['cy']), np.int(p['r']), color, -1)
            image = cv2.rectangle(image, (p['cx']-np.int32(p['r']/2.0), p['cy']-np.int32(
                p['r']/2.0)), (p['cx'] + np.int32(p['r']), p['cy'] + np.int32(p['r'])), color, -1)
        elif shape == "point":
            #image = cv2.circle(image, (p['cx'], p['cy']), 15, color, -1)
            image = cv2.rectangle(
                image, (p['cx']-8, p['cy']-8), (p['cx']+16, p['cy']+16), color, -1)
        elif shape == "polygon":
            pts = np.zeros((len(p['all_points_x']), 2), np.int32)
            for i in range(len(p['all_points_x'])):
                pts[i] = [p['all_points_x'][i], p['all_points_y'][i]]
            if (self.config.MODE == "Combined"):
                pts = pts.reshape((-1, 1, 2))
            elif (self.config.MODE == "Separate"):
                pts = pts.reshape((1, -1, 2))
            image = cv2.fillPoly(image, pts, color, lineType=cv2.LINE_AA)

        return image

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "floorplan":
            return len(info["polygons"])
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################


def train(train_mode='heads', config='Deakin', gpu_count=1, weights_path=None, log_dir=None, dataset_dir=None):
    # Configurations
    config = DeakinFloorplanConfig() if config == "Deakin" else SchoolFloorplanConfig()
    config.GPU_COUNT = gpu_count
    config.display()

    # Create model
    model = MaskRCNN(mode="training", config=config,
                     model_dir=log_dir)

    # Download weights file
    if not os.path.exists(weights_path):
        if not os.path.exists(os.path.dirname(weights_path)):
            os.makedirs(os.path.dirname(weights_path))
        download_trained_weights(weights_path)

    # Load model
    # Changed input Model path to imagenet resnet50
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    # Training dataset.
    dataset_train = FloorplanDataset()
    dataset_train.load_floorplan(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FloorplanDataset()
    dataset_val.load_floorplan(dataset_dir, "val")
    dataset_val.prepare()

    if train_mode == 'heads':
        # STAGE1: Train the head branches
        # Passing layers="heads" freezes all layers except the head layers.
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=20,
                    layers='heads')

    elif train_mode == 'partial':
        # STAGE 2: Partial training
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=100,
                    layers="3+")
    elif train_mode == 'full':
        # STAGE 3: Fine tune all layers
        # Passing layers="all" trains all layers.
        print("Full training")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=100,
                    layers="all",
                    augmentation=None)
    else:
        print("'{}' is not recognized. "
              "Use 'heads' or 'partial' or 'full'".format(train_mode))
