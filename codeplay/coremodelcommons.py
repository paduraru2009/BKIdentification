import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE

USE_GPU = 1
DEVICE = '/device:gpu:0' if USE_GPU else '/device:cpu:0'

# Dataset processing utilities and parameters
# ------------------------------------------------------------------------
# TODO: add more groups when we have data. E.g. (2,2), (1,4), (4,1), (4,2), (2,4), (4,4)
AspectRatioGroups_ranges = [(np.NINF, 0.67),  (0.67, 1.5), (1.5, np.PINF)] # width / height
AspectRatioGroups_names = [(1,2), (1,1), (2,1)]
#AspectRatioGroups_ranges = [(np.NINF, 0.27), (0.27, 0.67), (0.67, 1.5), (1.5, 3.6), (3.6, np.PINF)] # width / height
#AspectRatioGroups_names = [(1,4), (1,2), (1,1), (2,1), (4,1)]
assert len(AspectRatioGroups_names) == len(AspectRatioGroups_ranges)

classNames_lst = ["negatives", "positives"]
classNames = tf.convert_to_tensor(classNames_lst)
base_img_height = 224
base_img_width = 224  # Reuse imagenet in mind...
rescaleIn01 = True  # How to do rescaling, 01 or -11 ?
scaleMean = 0.0 if rescaleIn01 == True else 127.5
scaleFactor = 255.0 if rescaleIn01 else 127.5
half_batch_size = 32  # Because we bring one from pos and one from positive, together they make a 64 full batch !

# Other params
# -------------------------------------------------------------------------
# If true, when evaluating accuracy both training and eval datasets are taken into account. Used now only because we have a very small dataset...
USE_BOTH_TRAINING_AND_EVAL_ACC_STATS = True
EPOCHS = 1000
stepsToLog = 30  # At how many training steps to log something...
PROB_THRESHOLD_POSITIVE = 0.5  # see top top comment
SAVE_MODEL_PATH = "./bestAccModel"
TENSORBOARD_LOGS_FOLDER = "tensorboard/logs/scalars"
from datetime import datetime

initial_learning_rate = 0.001
PARAM_LR_SCHEDULE = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=1000,
                                                                   decay_rate=0.96, staircase=True)
PARAM_OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=PARAM_LR_SCHEDULE)
PARAM_LOSS_FN = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)


def findAspectRatioGroup(aspectRatio):
    aspectRatioIndexFound = None
    for aspectRatioIndex, aspectRatioRange in enumerate(AspectRatioGroups_ranges):
        if aspectRatioRange[0] <= aspectRatio < aspectRatioRange[1]:
            aspectRatioIndexFound = aspectRatioIndex
            break
    assert aspectRatioIndexFound != None, f"no aspect ratio found for input {aspectRatio}"
    return AspectRatioGroups_names[aspectRatioIndexFound]

# Returns the resolution of the batch based on the aspect ratio
def findResByAspectRatioSpec(aspectRatio):
    assert isinstance(aspectRatio, tuple) and len(aspectRatio) == 2
    resX = aspectRatio[0] * base_img_width
    resY = aspectRatio[1] * base_img_height
    return resX, resY

