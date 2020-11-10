import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from coremodelcommons import *

class CoreModelDatasetHelper:
    @staticmethod
    def basicProcessDatasetImgLoaded(img):

        return img

    @staticmethod
    def basicProcessDatasetImg(file_path, resX=None, resY=None, scaleX = None, scaleY=None, findLabelFromPath=True):
        # First, find the label from path
        label = None
        if findLabelFromPath:
            label = 1.0
            parts = tf.strings.split(file_path, os.path.sep)
            classForFile = parts[-2]

            if classForFile == classNames[0]:
                label = 0.0

        # Read image and resize
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, dtype=tf.float32)

        if resY != None and resX != None: # Maybe user doesn't want to resize
            img = tf.image.resize(img, [resY, resX])

        if scaleX != None and scaleY != None:
            actualHeight = img.shape[0]
            actualWidth = img.shape[1]
            img = tf.image.resize(img, [actualHeight * scaleY, actualWidth * scaleX])

        img = (img - scaleMean) / scaleFactor  # Rescale
        return img, label, file_path

    @staticmethod
    def processDatasetWithAugmentation(file_path, resX, resY):
        img, label, filename = CoreModelDatasetHelper.basicProcessDatasetImg(file_path, resX, resY)

        # img = tf.image.random_crop(img, size=[img_height, img_width, 3])
        # Flip - horizontal or vertical
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)

        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)

        # Add brighness
        if tf.random.uniform(()) > 0.8:
            img = tf.image.random_brightness(img, max_delta=0.1)  # Random brightness

        # Final clip in range
        img = tf.clip_by_value(img, 0, 1) if rescaleIn01 else tf.clip_by_Value(img, -1, 1)

        return img, label, filename

    @staticmethod
    def loadDatasets(basePath, tupleSpec, resX, resY):
        dataset_ratioX, dataset_ratioY = tupleSpec

        pathToDataSetFolder = os.path.join(basePath,
                                           #f"ProcessedDataset_{dataset_size}_{dataset_ratioX}_{dataset_ratioY}")
                                           f"ProcessedDataset_{dataset_ratioX}_{dataset_ratioY}")

        pathToTrainFolder = os.path.join(pathToDataSetFolder, "CoreDetector", "train")
        pathToValidationFolder = os.path.join(pathToDataSetFolder, "CoreDetector", "validation")

        pathToTrain_negFolder = os.path.join(pathToTrainFolder, "negatives")
        pathToTrain_posFolder = os.path.join(pathToTrainFolder, "positives")
        pathToValidation_negFolder = os.path.join(pathToValidationFolder, "negatives")
        pathToValidation_posFolder = os.path.join(pathToValidationFolder, "positives")

        # /Users/ciprian/Downloads/SplittedDatasets/ProcessedDataset_35_1_1/CoreDetector/train

        train_neg_ds = tf.data.Dataset.list_files(pathToTrain_negFolder + '/*.*', shuffle=False)
        train_pos_ds = tf.data.Dataset.list_files(pathToTrain_posFolder + '/*.*', shuffle=False)
        validation_neg_ds = tf.data.Dataset.list_files(pathToValidation_negFolder + '/*.*', shuffle=False)
        validation_pos_ds = tf.data.Dataset.list_files(pathToValidation_posFolder + '/*.*', shuffle=False)

        trainMaxSize = max(tf.data.experimental.cardinality(train_neg_ds).numpy(),
                           tf.data.experimental.cardinality(train_pos_ds).numpy())

        evalMaxSize = max(tf.data.experimental.cardinality(validation_neg_ds).numpy(),
                          tf.data.experimental.cardinality(validation_pos_ds).numpy())

        print(f'Train data size neg {tf.data.experimental.cardinality(train_neg_ds).numpy()}')
        print(f'Train data size pos {tf.data.experimental.cardinality(train_pos_ds).numpy()}')
        print(f'Eval data size neg {tf.data.experimental.cardinality(validation_neg_ds).numpy()}')
        print(f'Eval data size pos {tf.data.experimental.cardinality(validation_pos_ds).numpy()}')

        # Notes:
        # 1. training datasets does augmentation, but validation is NOT !
        # 2. training datasets are cached and shuffled , but validations are not
        # 3. training data is repeted at infinite ! check how many iterations you run then per training.
        # 4. both are prefetched to overlap computation and communication
        train_neg_ds = train_neg_ds.cache().shuffle(1000).map(lambda path : CoreModelDatasetHelper.processDatasetWithAugmentation(path, resX, resY),
                                                              num_parallel_calls=AUTOTUNE).repeat().batch(
            half_batch_size).prefetch(AUTOTUNE)
        train_pos_ds = train_pos_ds.cache().shuffle(1000).map(lambda path : CoreModelDatasetHelper.processDatasetWithAugmentation(path, resX, resY),
                                                              num_parallel_calls=AUTOTUNE).repeat().batch(
            half_batch_size).prefetch(AUTOTUNE)
        validation_neg_ds = validation_neg_ds.map(lambda path : CoreModelDatasetHelper.processDatasetWithAugmentation(path, resX, resY), num_parallel_calls=AUTOTUNE).batch(
            half_batch_size).prefetch(AUTOTUNE)
        validation_pos_ds = validation_pos_ds.map(lambda path : CoreModelDatasetHelper.processDatasetWithAugmentation(path, resX, resY), num_parallel_calls=AUTOTUNE).batch(
            half_batch_size).prefetch(AUTOTUNE)

        print(f"Train neg cardinality {tf.data.experimental.cardinality(train_neg_ds).numpy()}")
        print(f"Train pos cardinality {tf.data.experimental.cardinality(train_pos_ds).numpy()}")
        print(f"Validation neg cardinality {tf.data.experimental.cardinality(validation_neg_ds).numpy()}")
        print(f"Validation pos cardinality {tf.data.experimental.cardinality(validation_pos_ds).numpy()}")

        return train_neg_ds, train_pos_ds, validation_neg_ds, validation_pos_ds, trainMaxSize, evalMaxSize
