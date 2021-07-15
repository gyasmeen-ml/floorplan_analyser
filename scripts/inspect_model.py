"""
Test and evaluate Mask R-CNN on floorplan dataset
Written by Yasmeen George
Date: 18/03/2021
"""

from random import randint
import pandas as pd
import skimage
import os
import sys
sys.path.append('../scripts/')
sys.path.append('../')
from floorplan import *
from model import *
from visualize import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

############################################################
#  Configuration
############################################################


def get_config(confidence=0.5, config="School"):

    class DeakinInferenceConfig(DeakinFloorplanConfig):

        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        IMAGE_RESIZE_MODE = "pad64"

        DETECTION_MIN_CONFIDENCE = confidence

    class SchoolInferenceConfig(SchoolFloorplanConfig):

        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        IMAGE_RESIZE_MODE = "pad64"

        DETECTION_MIN_CONFIDENCE = confidence

    return DeakinInferenceConfig() if config == "Deakin" else SchoolInferenceConfig()


def load_dataset(subset="val", FLOORPLAN_DIR=None, config=DeakinFloorplanConfig()):
    # Load validation dataset
    dataset = FloorplanDataset(config)
    dataset.load_floorplan(FLOORPLAN_DIR, subset)

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(
        len(dataset.image_ids), dataset.class_names))
    return dataset


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def visualize_annotations(image_id=0, dataset=None, save_path=None, axis=None):
    # Load random image and mask.
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = extract_bboxes(mask)

    # Display image and additional stats
    print("image id: {} and number of annotated objects: {}".format(
        image_id, dataset.image_reference(image_id)))

    log("image", image)

    # Display image and instances
    if axis:
        ax = axis
    else:
        ax = get_ax(rows=1, cols=1, size=16)
    d_clrs = random_colors(len(dataset.class_names),
                           bright=True, shuffle=False)
    clrs = [d_clrs[x - 1] for x in class_ids]

    im = display_instances(image, bbox, mask, class_ids,
                           dataset.class_names, ax=ax, colors=clrs)
    if save_path is not None:

        plt.savefig("{}/{}_{}".format(save_path,image_id,"gt.jpg")
                                   #dataset.image_info[image_id]["id"].split('.')[0] + "_gt.jpg")
                    , bbox_inches='tight', pad_inches=0)

    return im


def load_model(config=None, weights_path=None):
    # Create model in inference mode

    with tf.device(DEVICE):
        model = MaskRCNN(mode=TEST_MODE, model_dir=weights_path,
                         config=config)
    print("Loading weights: {} ".format(weights_path.split('/')[-1]))
    model.load_weights(weights_path, by_name=True)
    return model

def run(im_path=None, model=None, dataset=None, save_path=None, show_mask=False):
    """Test the model on a given image
    im_path: path of the image
    model: inference model
    dataset: the dataset which has the class names
    save_path: path to save the predictions
    """
    image = skimage.io.imread(im_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]

    results = model.detect([image], verbose=0)

    # Display results
    ax = get_ax(rows=1, cols=1, size=40)
    r = results[0]
    print("image ID: {} ".format(im_path.split('/')[-1]))

    d_clrs = random_colors(len(dataset.class_names),
                           bright=False, shuffle=False)
    clrs = [d_clrs[x - 1] for x in r['class_ids']]

    im = display_instances(image, r['rois'], r['masks'], r['class_ids'],
                           dataset.class_names, r['scores'], ax=ax, show_mask=show_mask,
                           # show_bbox=False,
                           title="Predictions", colors=clrs)

    if save_path is not None:
        plt.savefig("{}/{}".format(save_path, im_path.split('/')[-1]))
    return im


def run_detection(im_id=1, config=None, model=None, dataset=None, save_path=None, show_mask=False, axis=None):
    """Test the model on a given image_id from the dataset
        config: inference configuration
        model: inference model
        dataset: the dataset which has the class names
        save_path: path to save the predictions
        visualise: boolean to display extra performance plots based on the ground truth
    """
    image_id = dataset.image_ids[im_id]  # random.choice(dataset.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ".format(info["source"], info["id"]))

    # Run object detection
    results = model.detect([image], verbose=0)

    # Display results
    if axis:
        ax = axis
    else:
        ax = get_ax(rows=1, cols=1, size=40)
    r = results[0]


    d_clrs = random_colors(len(dataset.class_names),
                           bright=False, shuffle=False)
    clrs = [d_clrs[x - 1] for x in r['class_ids']]

    im = display_instances(image, r['rois'], r['masks'], r['class_ids'],
                           # show_bbox=False,
                           dataset.class_names, r['scores'], ax=ax,
                           show_mask=show_mask, title="Predictions", colors=clrs)
    if save_path is not None:
        plt.savefig("{}/{}_{}".format(save_path,image_id,
                                   dataset.image_info[image_id]["id"].split('.')[0] + "_pred.jpg"))
    return im


def compute_batch_ap(image_ids, config=None, model=None, dataset=None, iou_threshold=0.5):
    # Compute Average Precision for an image
    cols = ['Name', "Width", "Height", "GT_Count", 'CorrectMatches-TP',
            'IncorrectMatches-FN', 'MissingMatches-FP', 'AP', 'Accuracy']

    df = pd.DataFrame(columns=cols)
    APs = []

    for image_id in image_ids:

        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            load_image_gt(dataset, config,
                          image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]

        AP, precisions, recalls, overlaps, gt_class_ids, pred_match_class_ids, matches_stats = \
            compute_ap(gt_bbox, gt_class_id, gt_mask,
                       r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=iou_threshold)

        df.loc[len(df)] = [dataset.image_info[image_id]['id'], dataset.image_info[image_id]['width'], dataset.image_info[image_id]['height'],
                           len(dataset.image_info[image_id]['polygons'])] + matches_stats[0:3] + [np.round(AP, 3), np.round(matches_stats[0]/(matches_stats[1]+matches_stats[2]+matches_stats[0]), 3)]
        APs.append(AP)
    return APs, df


def compute_batch_ap_ew(image_ids, config=None, model=None, dataset=None, iou_threshold=0.5):
    # Compute Average Precision for an image
    cols = ['Name', "Width", "Height", "GT_Count", 'CorrectMatches-TP',
            'IncorrectMatches-FN', 'MissingMatches-FP', 'AP', 'Accuracy']

    df = pd.DataFrame(columns=cols)
    class_df = None
    r_gt_class_ids = []
    r_pd_class_ids = []
    r_image_dims = []
    r_image_names = []
    APs = []

    for i in range(len(image_ids)):

        image_id = image_ids[i]

        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            load_image_gt(dataset, config,
                          image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]

        AP, precisions, recalls, overlaps, gt_class_ids, pred_match_class_ids, matches_stats = \
            compute_ap(gt_bbox, gt_class_id, gt_mask,
                       r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=iou_threshold)

        class_id_matches = pred_match_class_ids == gt_class_ids
        overall_class_acc = round(len(class_id_matches[class_id_matches])/(
            matches_stats[1]+matches_stats[2]+matches_stats[0]))

        total_class_counts = dict()
        matched_class_counts = dict()
        for i in range(len(pred_match_class_ids)):
            if (class_id_matches[i] == True):
                matched_class_counts[gt_class_ids[i]] = matched_class_counts.get(
                    gt_class_ids[i], 0) + 1
            total_class_counts[gt_class_ids[i]] = total_class_counts.get(
                gt_class_ids[i], 0) + 1

        class_acc = dict()
        for class_id, count in total_class_counts.items():
            if (class_id in matched_class_counts.keys()):
                class_acc[class_id] = matched_class_counts.get(
                    class_id) / count
                continue
            else:
                class_acc[class_id] = 0

        import collections
        class_acc = collections.OrderedDict(sorted(class_acc.items()))

        if (class_df is None):
            class_df_columns = ["Name", "Total Regions", "Overall Acc"] + \
                ["{} Acc / Prec".format(x) for x in dataset.class_names[1:]]
            class_df = pd.DataFrame(columns=class_df_columns)

        column_count = len(class_df.columns)
        fp = matches_stats[3]
        tp = matched_class_counts

        temp_row = [dataset.image_info[image_id]['id'],
                    len(gt_class_ids), overall_class_acc]

        for x in class_acc.keys():
            temp = "{0} of {1}  ({2:.2f}%) / {3:.2f}%"
            matched = matched_class_counts.get(x, 0)
            total = total_class_counts[x]
            avg_class_acc = class_acc[x]*100
            num_fp = len(fp.get(x, list()))
            num_tp = tp.get(x, 0)
            temp_row.append(temp.format(matched, total, avg_class_acc,
                            (num_tp / ((num_tp + 1 if num_tp == 0 else num_tp) + num_fp))*100))

        new_row = np.zeros(shape=(column_count), dtype=object)
        new_row[0] = dataset.image_info[image_id]['id']
        new_row[1] = len(gt_class_ids)
        new_row[2] = overall_class_acc

        for idx, key in enumerate(class_acc.keys()):
            new_row[2 + key] = temp_row[3 + idx]

        class_df.loc[len(class_df)] = new_row

        df.loc[len(df)] = [dataset.image_info[image_id]['id'], dataset.image_info[image_id]['width'], dataset.image_info[image_id]['height'], len(
            dataset.image_info[image_id]['polygons'])] + matches_stats[:3] + [np.round(AP, 3), np.round(matches_stats[0]/(matches_stats[1]+matches_stats[2]+matches_stats[0]), 3)]
        APs.append(AP)

        r_gt_class_ids.append(gt_class_ids)
        r_pd_class_ids.append(pred_match_class_ids)
        r_image_dims.append([dataset.image_info[image_id]['width'], dataset.image_info[image_id]['height']])
        r_image_names.append(dataset.image_info[image_id]['id'])

    return APs, df, class_df, r_gt_class_ids, r_pd_class_ids, r_image_dims, r_image_names, r['rois']

class Detection:
    """Encapsulates the related data of a Detection into a single representative object
    """

    def __init__(self, model_id, pred_box, pred_class_id, pred_score, pred_mask):
        self.model_id = model_id
        self.pred_box = pred_box
        self.pred_class_id = pred_class_id
        self.pred_score = pred_score
        self.pred_mask = pred_mask

class Detections(list):
    
    def __init__(self):
        super().__init__()
        self.next = 0
        
    def add_detection(self, detection):
        detection.id = self.next
        self.next += 1
        self.append(detection)
        
    def get_class_ids(self):
        cids = [] 
        for detection in self:
            cids.append(detection.pred_class_id)
        return np.array(cids)
        
    def get_boxes(self):
        boxes = [] 
        for detection in self:
            boxes.append(detection.pred_box)
        return np.array(boxes)
        
    def get_masks(self):
        masks = [] 
        for detection in self:
            masks.append(detection.pred_mask)
        return np.dstack(tup=masks)
    
    def get_scores(self):
        scores = [] 
        for detection in self:
            scores.append(detection.pred_score)
        return scores

class DetectionRegion:
    
    def __init__(self, region_id, detections=np.array([])):
        self.region_id = region_id
        self.detections = detections
        
    def add_detection(self, detection):
        self.detections = np.append(self.detections, detection)
    
    def get_detection_ids(self):
        ids = np.array([])
        for detection in self.detections:
            ids = np.append(ids, detection.id)
        return ids
    
    def get_masks(self):
        masks = [] 
        for detection in self.detections:
            masks.append(detection.pred_mask)
        return masks
                
def calculate_regions(detections_1, detections_2, iou_threshold=0.5):
    
    # for max int size
    # TODO: replace with less error prone design
    import sys
    
    regions = np.array([], dtype=object)
    
    # combine masks
    masks_1 = detections_1.get_masks()
    masks_2 = detections_2.get_masks()
    # masks_3 = detections_3.get_masks()
    
    # compute overlaps
    overlaps = compute_overlaps_masks(masks_1, masks_2)
    
    # group masks by overlap > iou_threshold (from masks_2)
    for idx, overlap in enumerate(overlaps):   

        # get all overlap indices with an IoU more than zero
        non_zero_overlap = np.nonzero(overlap)[0]
        # get the corresponding values of the non zero indices
        values = overlap[non_zero_overlap]
        # filter the non zero values for all indices with a value more than the threshold and less than 1
        # 1 is the IoU for the mask overlapping the same mask
        w = np.where((values < 1) & (values > iou_threshold))[0]
        
        # if there are overlaps in the above filter, attempt to find a detection region with the same overlap indices
        # or create new one with all indices from w + idx
        if (len(w) != 0):
            
            # get the filtered indices with respect to the original detection array
            w_detection_idx = non_zero_overlap[w] 
            detection_region = None
            
            # attempt to find a region with the same overlapping detections as found in w
            for region in regions:
                region_indices = region.get_detection_ids()
                if np.any(np.isin(region_indices, idx)):
                    detection_region = -1
                    break
                if np.any(np.isin(region_indices, w_detection_idx)):
                    # report detection region for adding region (idx)
                    detection_region = region 
                    break
            
            # store the detection instance at idx
            idx_detection = detections_1[idx]
            
            # if the detection indices are already in an enclosed detection region, move to next detection
            # 'enclosed' refers to to the detection region having the same indices as the proposed region
            if (detection_region == -1):
                continue
            # add detection to the reported detection region
            elif detection_region != None:
                detection_region.add_detection(idx_detection)
            # create new detection region 
            else:
                new_dr = DetectionRegion(randint(0, sys.maxsize))
                new_dr.add_detection(idx_detection)
                w_detections = [detections_2[k] for k in w_detection_idx]
                for detection in w_detections:
                    new_dr.add_detection(detection)
                regions = np.append(regions, new_dr)
                
        # if there are no overlaps for idx (besides itself), create a new detection region in the image including detection (idx)
        else:
            idx_detection = detections_1[idx]
            new_dr = DetectionRegion(randint(0, sys.maxsize))
            new_dr.add_detection(idx_detection)
            regions = np.append(regions, new_dr)
    
    return regions

def find_optimal_detection(class_ids, scores, masks, fusing_operation="AND"):
    """Compares the provided class ids and scores to find the optimal class id for the collection (used for model fusion)

    Parameters
    ----------
    class_ids : (list/array) 
        list of class ids to compare
    scores : (list/array) 
        list of detection scores (corresponding to the above class ids)
    masks : (list/array) 
        list of detection masks corresponding to the above parameters
    fusing_operation : str, optional
        AND, OR - bitwise operation to be completed for masks, by default "AND"

    Returns
    -------
    tuple
        optimal class id, optimal confidence, merged mask
    """

    # ensure the inputs are of the same shape / size
    assert len(class_ids) == len(scores) == len(masks)

    unique, counts = np.unique(class_ids, return_counts=True)
    count_max_indices = np.argwhere(counts == np.amax(counts)).flatten()

    temp_class_ids = []
    temp_scores = []
    temp_masks = []
    
    for idx in count_max_indices:

        sum_score = 0  # confidence for this max instance
        # number of instances at max count (idx) in the original element detections
        inst_count = counts[idx]
        inst_class_id = unique[idx]
        inst_mask = []

        # iterates all instances of a unique maximum value in 'class_ids' 
        for inst_idx in np.argwhere(class_ids == unique[idx]).flatten():
            sum_score += scores[inst_idx]
            # - compare overlap of detection masks
            #     - use fusing_operation to determine how to merge detections masks
            #     - OR operation:  if any of the masks have a 1, add that specific pixel (column, row) to the result buffer (size of the image)
            #     - AND operation: if all masks have a 1, add that specific pixel (column, row) to the result buffer (size of the image)
            if len(inst_mask) == 0:
                inst_mask = masks[inst_idx]
                continue
            inst_mask = inst_mask & masks[inst_idx] if fusing_operation == "AND" else inst_mask | masks[inst_idx] # merges the previous instance mask with the current instance mask

        temp_class_ids.append(inst_class_id)
        temp_scores.append(sum_score / inst_count)
        temp_masks.append(inst_mask)

    # - compare detection ids
    #     - if unbalanced (e.g. 2 of one id, 1 of another): take the average confidence of those two
    #     - if all the same (i.e. all with same id): take average of the sum of confidence
    #     - if all different, take highest confidence
    #     - TODO: add weighted average
    max_temp_score_idx = np.argmax(temp_scores)
    
    return (temp_class_ids[max_temp_score_idx], temp_scores[max_temp_score_idx], temp_masks[max_temp_score_idx])

def calc_bbox(mask):

    mask_idx = np.argwhere(mask == True)

    if (len(mask_idx) == 0):
        return np.array([0, 0, 0, 0])

    # create the bbox for the mask
    x_0 = np.min(mask_idx[:, 1]) + 1
    x_1 = np.max(mask_idx[:, 1]) + 1
    y_0 = np.min(mask_idx[:, 0]) + 1
    y_1 = np.max(mask_idx[:, 0]) + 1

    return np.array([y_0, x_0, y_1, x_1])

def get_optimal(regions, fusing_operation):

    # determine optimal detection class_id, score, and mask for each detection region with n detections > 1 
    detections = Detections()

    for idx, region in enumerate(regions):
        # add single detection in region (no other detections to compare) to list of new detections
        if len(region.detections) == 1:
            detections.append(region.detections[0])
            continue

        class_ids = np.array([], dtype=int)
        scores = np.array([], dtype=int)
        masks = region.get_masks()

        # collate detection data
        for detection in region.detections:
            class_ids = np.append(class_ids, detection.pred_class_id)
            scores = np.append(scores, detection.pred_score)
        
        # create new detection instance for region
        op_class_id, op_score, merged_mask = find_optimal_detection(class_ids, scores, masks, fusing_operation=fusing_operation)

        op_detection = Detection(-1, calc_bbox(merged_mask), op_class_id, op_score, merged_mask)
        detections.add_detection(op_detection)

    return detections
def get_fusion_results(image, models=None, iou_threshold=0.5, fusing_operation="AND"):
    # make detections
    # model 1 detections
    mr_1 = models[0].detect([image], verbose=0)[0]
    # model 2 detections
    mr_2 = models[1].detect([image], verbose=0)[0]
    # model 3 detections
    mr_3 = models[2].detect([image], verbose=0)[0]

    # create detection instances
    detections = Detections()

    m1_len = len(mr_1['rois'])  # number of detections from model 1
    m2_len = len(mr_2['rois'])  # number of detections from model 2
    m3_len = len(mr_3['rois'])  # number of detections from model 3

    # model 1 detections instances
    for i in range(m1_len):
        detection = Detection(0, mr_1['rois'][i], mr_1['class_ids'][i], mr_1['scores'][i], mr_1['masks'][:, :, i])
        detections.add_detection(detection)
    # model 2 detections instances
    for i in range(m2_len):
        detection = Detection(1, mr_2['rois'][i], mr_2['class_ids'][i], mr_2['scores'][i], mr_2['masks'][:, :, i])
        detections.add_detection(detection)
    # model 3 detections instances
    for i in range(m3_len):
        detection = Detection(2, mr_3['rois'][i], mr_3['class_ids'][i], mr_3['scores'][i], mr_3['masks'][:, :, i])
        detections.add_detection(detection)

    # calculate regions
    regions = calculate_regions(detections, detections, iou_threshold=iou_threshold)

    # calculate optimal detections for each region
    optimal = get_optimal(regions, fusing_operation)
    return optimal
def compute_fusion(image_ids, config=None, models=None, dataset=None, iou_threshold=0.5, fusing_operation="AND"):

    image_fusion_results = dict()

    for im in range(len(image_ids)):

        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config, image_ids[im], use_mini_mask=False)

        optimal = get_fusion_results(image,models=models, iou_threshold=iou_threshold, fusing_operation=fusing_operation)
        
        image_fusion_results[image_ids[im]] = optimal

    return image_fusion_results

def compute_fusion_gt(image_ids, config=None, models=None, dataset=None, iou_threshold=0.7, fusing_operation="AND"):

    image_fusion_results = dict()

    for im in range(len(image_ids)):

        image_id = image_ids[im]

        image_width = dataset.image_info[image_id]['width']
        image_height = dataset.image_info[image_id]['height']

        # load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, config, image_id, use_mini_mask=False)

        model_results = np.zeros(shape=3, dtype=object)

        for i in range(3):
            # Run object detection
            results = models[i].detect([image], verbose=0)
            r = results[0]
            
            model_fusion_data = FusionData(
                gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'], compute_overlaps_masks(r['masks'], gt_mask))
            
            model_results[i] = model_fusion_data

        # each model with the corresponding detection data categorized based on max overlap over a gt element
        element_detections_class_ids = dict()
        element_detections_scores = dict()
        element_detections_masks = dict()

        for i in range(3):

            # - for each detection
            #     - get detection with max overlap over a gt element
            #     - add detection into np array associated with the gt element index
            
            temp_ed_class_ids = dict()
            temp_ed_scores = dict()
            temp_ed_masks = dict()
                            
            for idx, det_over in enumerate(model_results[i].overlaps):
                
                # gt element index of the max overlap found at detection idx
                max_overlap_idx = np.argmax(det_over)
                
                # detection class id
                class_id = model_results[i].pred_class_ids[idx]
                # detection score for current detection instance
                score = model_results[i].pred_scores[idx]
                # detection mask (width x height)
                mask = model_results[i].pred_masks[:, :, idx]

                # list of detections associated with a region within in the image (different models have different lists)
                element_cids = temp_ed_class_ids.get(
                    max_overlap_idx, np.array([], dtype=int))
                element_scores = temp_ed_scores.get(
                    max_overlap_idx, np.array([]))  # list of element scores
                element_mask = temp_ed_masks.get(
                    max_overlap_idx, list())  # list of class ids

                element_cids = np.append(element_cids, class_id)
                element_scores = np.append(element_scores, score)
                element_mask.append(mask)

                temp_ed_class_ids[max_overlap_idx] = element_cids
                temp_ed_scores[max_overlap_idx] = element_scores
                temp_ed_masks[max_overlap_idx] = element_mask
                
            # - determine the most optimal detection for each class -- max 1 result for each region (for model i)
            for i in list(temp_ed_class_ids.keys()):
                
                class_id, score, mask = find_optimal_detection(temp_ed_class_ids[i], temp_ed_scores[i], temp_ed_masks[i], fusing_operation=fusing_operation)

                ed_cid = element_detections_class_ids.get(i, np.array([], dtype=int))
                ed_scores = element_detections_scores.get(i, np.array([], dtype=int))
                ed_masks = element_detections_masks.get(i, list())
                
                ed_cid = np.append(ed_cid, class_id)
                ed_scores = np.append(ed_scores, score)
                ed_masks.append(mask)

                element_detections_class_ids[i] = ed_cid
                element_detections_scores[i] = ed_scores
                element_detections_masks[i] = ed_masks
                

        final_gt_region_matches = np.array([], dtype=int)
        final_element_ids = dict()
        final_element_scores = dict()
        final_masks = dict()

        for i in list(element_detections_class_ids.keys()):
            
            class_id, score, mask = find_optimal_detection(element_detections_class_ids[i], element_detections_scores[i], element_detections_masks[i], fusing_operation=fusing_operation)
            
            final_element_ids[i] = class_id
            final_element_scores[i] = score
            final_masks[i] = mask
            final_gt_region_matches = np.append(final_gt_region_matches, i)

        # - create bounding box based on 2d search of 1 pixels (of the above fused segmentation mask)
        #     - x_start = left most 1 pixel (col)
        #     - x_end = right most 1 pixel (col)
        #     - y_start = top most 1 pixel (row)
        #     - y_end = bottom most 1 pixel (row)
        final_bboxes = dict()

        for k, v in final_masks.items():

            mask_idx = np.argwhere(v == True)

            if (len(mask_idx) == 0):
                final_bboxes[k] = np.array([0, 0, 0, 0])
                continue

            # create the bbox for the region mask
            x_0 = np.min(mask_idx[:, 1]) + 1
            x_1 = np.max(mask_idx[:, 1]) + 1
            y_0 = np.min(mask_idx[:, 0]) + 1
            y_1 = np.max(mask_idx[:, 0]) + 1

            final_bboxes[k] = np.array([y_0, x_0, y_1, x_1])

        # - store fused detected elements (with superior id, merged segmentation masks, and calculated bbox)
        image_fusion_results[im] = {"fused_class_ids": final_element_ids, "fused_class_scores": final_element_scores, "fused_masks": final_masks, "fused_bboxes": final_bboxes,
                                    "gt_region_matches": final_gt_region_matches, "image_width": image_width, "image_height": image_height, "ttl_valid_detects": len([1 for z in final_masks.values() if len(z) != 0])}

    return image_fusion_results
