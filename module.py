import multiprocessing as mp
import numpy as np
import sys
import torch
from torch.nn import functional as F

# detectron2
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
# CenterNet2
sys.path.insert(0, '/usr/local/Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
# Detic
sys.path.insert(0, '/usr/local/Detic/')
from detic.config import add_detic_config
from detic.modeling.text.text_encoder import build_text_encoder


# path to config file
CONFIG_FILE = '/usr/local/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
# Modify config options using the command-line 'KEY VALUE' pairs
OPTS = [
    'MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth',
    'MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH', '/usr/local/Detic/datasets/metadata/lvis_v1_train_cat_info.json',
]
# Minimum score for instance predictions to be shown
CONFIDENCE_THRESHOLD = 0.5

# Habitat Challenge 2022 goals
OBJECT_GOAL_CATEGORIES = ['chair', 'couch', 'potted plant', 'bed', 'toilet', 'tv']


def setup_cfg():
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(CONFIG_FILE)
    cfg.merge_from_list(OPTS)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.freeze()
    return cfg


class Detic(object):
    def __init__(self, cfg):
        # CLIP
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        # add 'a ' before each words.  ex) chair -> a chair
        prompt = 'a '
        texts = [prompt + x for x in OBJECT_GOAL_CATEGORIES]
        # encode vocabulary
        classifier = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()

        num_classes = len(OBJECT_GOAL_CATEGORIES)
        self.predictor = DefaultPredictor(cfg)

        # https://github.com/facebookresearch/Detic/blob/main/detic/modeling/utils.py (line 32)
        self.predictor.model.roi_heads.num_classes = num_classes
        zs_weight = torch.cat([classifier, classifier.new_zeros((classifier.shape[0], 1))], dim=1)
        zs_weight = F.normalize(zs_weight, p=2, dim=0)
        zs_weight = zs_weight.to(self.predictor.model.device)
        for k in range(len(self.predictor.model.roi_heads.box_predictor)):
            del self.predictor.model.roi_heads.box_predictor[k].cls_score.zs_weight
            self.predictor.model.roi_heads.box_predictor[k].cls_score.zs_weight = zs_weight
    
    def __call__(self, image, goal):
        """
        Args:
            image(np.ndarray): an image of shape (H, W, C) (in RGB order).
            goal(int): the index of the goal
        Returns:
            is_detected(int): detect the goal object (1) or not (0)
            center(np.array): the center of the goal object 
        """
        is_detected = 0
        center = np.array([-1, -1])
        # convert image from RGB format to BGR format
        image = image[:, :, ::-1]
        # detect objects
        predictions = self.predictor(image)

        goal_instance = None
        # find the target goal
        for i in range(len(predictions['instances'])):
            prediction = predictions['instances'][i]

            # select the most confident prediction
            if prediction.pred_classes == goal:
                if not is_detected:
                    is_detected = 1
                    goal_instance = prediction
                elif goal_instance.scores.item() < prediction.scores.item():
                    goal_instance = prediction
        
        if is_detected:
            # calculate the center of the goal object
            seg_area = goal_instance.pred_masks.to('cpu').detach().numpy().copy().astype(np.int32)[0]
            ys, xs = np.where(seg_area==1)
            cx = int(np.average(xs))
            cy = int(np.average(ys))
            center = np.array([cx, cy])

        return is_detected, center


if __name__ == "__main__":
    from PIL import Image

    image = Image.open('habitat_img/00.png')
    image = np.asarray(image)

    mp.set_start_method("spawn", force=True)
    cfg = setup_cfg()
    detic = Detic(cfg)

    is_detected, center = detic(image, 0)
    print(is_detected, center)
