# THe purpose of this file is to read the original data set and produce a Processed dataset for training grouped on two levels:
# Positive samples and negative samples.
# The negative samples are currently based on exclusion: all the random bboxes that are not in the annotated image are considered as being NEGATIVES.
# This might be wrong and we need manual checks !



########## T1: ##########
"""
1. API for downloading the dataset

- need to download all the original pictures + CSV. These should be stored in a zip file on the server. We can use either Keras or wget (second is preferable here)

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

- need to download the split of positive/negative examples from a zip file that groups them on two sufolders, one for each of the two classes.
- need to have two subfolders, one for validation and one for training, let's say 80-20.

Then I can use the following in the code:
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

2. THERE ARE INVALID BBOXES !!!! Check the output of this script

Line 6 in CSV is wrong.
Please put asserts and sanity checks everywhere !!!

2. Sanity check scripts to check that all the files in the excel file are found on disk and INVERSE of this.

3. Sanity check to compare the images in the dataset with the one in the annotations made by doctors

4. Script to crop random img width/height from negative (all in image) example

"""


import os
import csv
import numpy as np
import cv2
import utils
import shutil
import random
from enum import IntEnum
from matplotlib import pyplot as plt
from coremodelcommons import AspectRatioGroups_ranges, AspectRatioGroups_names

# Where to read / write output
origDatasetPath = "OrigDataset"
#processedDatasetPath = os.path.join("ProcessedDataset", "CoreDetector")
#outputFolder=os.path.join(processedDatasetPath, NEGATIVE_FOLDER_NAME)

datasetDescPath_negatives = os.path.join(origDatasetPath, "Bacil-Koch_negative_boxes.csv")
datasetDescPath_positives = os.path.join(origDatasetPath, "Bacil-Koch_positive_boxes.csv")

POSITIVE_FOLDER_NAME = "positives"
NEGATIVE_FOLDER_NAME = "negatives"
TRAIN_FOLDER_NAME = "train"
VALIDATION_FOLDER_NAME="validation"


# Build dictionary of output folders data driven by above specs
OutputFolders = {}
OutputFolders["POSITIVES"] = {}
OutputFolders["NEGATIVES"] = {}
OutputFolders_train ={}
OutputFolders_train["POSITIVES"] = {}
OutputFolders_train["NEGATIVES"] = {}
OutputFolders_val ={}
OutputFolders_val["POSITIVES"] = {}
OutputFolders_val["NEGATIVES"] = {}

for tname in AspectRatioGroups_names:
    basePositivesTName = f"ProcessedDataset_{tname[0]}_{tname[1]}/CoreDetector"
    baseNegativesTName = f"ProcessedDataset_{tname[0]}_{tname[1]}/CoreDetector"
    OutputFolders["POSITIVES"][tname] = os.path.join(basePositivesTName, POSITIVE_FOLDER_NAME)
    OutputFolders["NEGATIVES"][tname] = os.path.join(baseNegativesTName, NEGATIVE_FOLDER_NAME)

    OutputFolders_train["POSITIVES"][tname]     = os.path.join(basePositivesTName, TRAIN_FOLDER_NAME, POSITIVE_FOLDER_NAME)
    OutputFolders_train["NEGATIVES"][tname]     = os.path.join(baseNegativesTName, TRAIN_FOLDER_NAME, NEGATIVE_FOLDER_NAME)
    OutputFolders_val["POSITIVES"][tname]       = os.path.join(basePositivesTName, VALIDATION_FOLDER_NAME, POSITIVE_FOLDER_NAME)
    OutputFolders_val["NEGATIVES"][tname]       = os.path.join(baseNegativesTName, VALIDATION_FOLDER_NAME, NEGATIVE_FOLDER_NAME)

def recreateImagesFolders(onlySplitFolders=False):
    # Remove and recreate output folder
    allOutputFolders = [OutputFolders, OutputFolders_train, OutputFolders_val]
    if onlySplitFolders:
        allOutputFolders = allOutputFolders[1:]

    for outFolder in allOutputFolders:
        for xkey, foldersCase in outFolder.items():
            for outputFolderKey, outputFolderStr in foldersCase.items():
                assert isinstance(outputFolderStr, str)
                if os.path.exists(outputFolderStr):
                    shutil.rmtree(outputFolderStr)
                os.makedirs(outputFolderStr)


# For each given row in the dataset description,
# produce in the output folder the desired number of positive/negative samples - if needsAugmentation is True, otherwise it just takes the small picture and put in the folder
def processExamples(needsAugmentation=False, onlyOutputStats=False):

    def outputExamples(debugName, datasetDescPath, needsAugmentation, onlyOutputStats):
        with open(datasetDescPath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            allBBoxesFound = []
            for row in reader:
                if row is None or len(row) == 0:
                    continue

                print(f"Processing row: {row[0]}")
                # One row = one input image data bboxes
                numEntries = len(row)
                assert numEntries > 1, "Your row seems to be almost empty..."
                if (numEntries - 1) % 4 != 0:
                    assert "I was expecting a series of 4 numbers representing the bboxes"
                numBBoxes = (numEntries - 1) // 4
                img_path = row[0]
                if len(img_path) == 0 or img_path==" ":
                    print(f"Invalid line in dataset {datasetDescPath}, row content is : {row}")
                    continue

                indexOfDot = img_path.rindex('.')
                img_name = img_path[:indexOfDot]
                rowData = row[1:]

                # Gather all bboxes
                bboxes_found = []
                for bboxIndex in range(numBBoxes):
                    bbox = rowData[bboxIndex*4 : (bboxIndex+1) * 4]
                    try:
                        bbox = [int(x) for x in bbox]
                    except:
                        print(f"INVALID BBOX here at index {index}")

                    assert len(bbox) == 4

                    if bbox[1] >= bbox[3] or bbox[0] >= bbox[2]:
                        print(f"INVALID BBOX here at index {index}")
                        continue

                    bboxes_found.append(bbox)

                    if onlyOutputStats:
                        continue

                    # Iterate and output all the bboxes
                    imgInputFullPath = os.path.join(origDatasetPath, img_path)
                    if not os.path.exists(imgInputFullPath):
                        print(f"file {imgInputFullPath} couldn't be found")
                        continue


                    img = cv2.imread(imgInputFullPath)  # , cv2.COLOR_BGR2RGB)
                    img_width = img.shape[1]
                    img_height = img.shape[0]
                    img_channels = 1 if len(img.shape) < 3 else img.shape[2]

                    for index, bbox in enumerate(bboxes_found):
                        # Find the aspect ratio group for this picture to identify where to output it
                        height = bbox[3] - bbox[1]
                        width = bbox[2] - bbox[0]
                        aspectRatio = width / height
                        aspectRatioGroup = findAspectRatioGroup(aspectRatio)
                        outputFolder = OutputFolders[debugName][aspectRatioGroup]

                        if "12288_14336_289bd60c-cc9b-4535-adf7-bae0d9679525" in row[0]:
                            a = 3
                        """
                        if width == 33 and height == 30:  # and "2048_12288_7d607c7b-8b5f-4701-bd67-17143cc39aa5" in row[0]:
                            aspectRatio = aspectRatio  # Debug breakpoint
                        if "2048_12288_7d607c7b-8b5f-4701-bd67-17143cc39aa5_9" in imgOutputFullPath:
                            a = 3 # Debug breakpoint
                        """

                        imgOutputFullPath = os.path.join(outputFolder, "{0}_{1}.jpg".format(img_name, index))

                        # Save the picture
                        img_proposed = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                        res = cv2.imwrite(imgOutputFullPath, img_proposed)
                        assert res

                allBBoxesFound.extend(bboxes_found)

        allBBoxesFound  = np.array(allBBoxesFound)
        widths  = allBBoxesFound[: , 2] - allBBoxesFound[: , 0]
        heights = allBBoxesFound[: , 3] - allBBoxesFound[: , 1]
        ratios = widths / heights


        mean_width = np.mean(widths)
        mean_height = np.mean(heights)

        BINS_WEIGHTS = [0, 30, 50, 100, 200, 300, 500]
        BINS_HEIGHTS = [0, 30, 50, 100, 200, 300, 500]
        BINS_RATIOS = [0, 0.2, 0.49, 1.0, 1.5, 2.0, 4.0]

        width_hist, widths_bin_edges = np.histogram(widths, bins=BINS_WEIGHTS, density=True)
        height_hist, heights_bin_edges = np.histogram(heights, bins=BINS_HEIGHTS, density=True)
        ratio_hist, ratio_bin_edges = np.histogram(ratios, bins=BINS_RATIOS, density=True)

        plt.hist(widths, bins=BINS_WEIGHTS, density=True)
        plt.title("Widths histogram - " + debugName)
        plt.savefig(f'Widths_hist - {debugName}.png')
        plt.show()

        plt.hist(heights, bins=BINS_HEIGHTS, density=True)
        plt.title("Heights histogram - " + debugName)
        plt.savefig(f'Heights_hist - {debugName}.png')
        plt.show()

        plt.hist(ratios, bins=BINS_RATIOS, density=True)
        plt.title("Ratios histogram - " + debugName)
        plt.savefig(f'Ratios_hist - {debugName}.png')
        plt.show()

    outputExamples("NEGATIVES", datasetDescPath_negatives, needsAugmentation=False, onlyOutputStats=onlyOutputStats)
    outputExamples("POSITIVES", datasetDescPath_positives, needsAugmentation=False, onlyOutputStats=onlyOutputStats)

def splitDataset(train_percent = 0.8):
    def saveclassFiles(allFilesInClass, train_percent, inFolder, outFolder_train, outFolder_validation):
        numFiles = len(allFilesInClass)
        trainCount = numFiles * train_percent
        for index, fileName in enumerate(allFilesInClass):
            fullInputPath = os.path.join(inFolder, fileName)
            if index < trainCount:
                fullOutPath = os.path.join(outFolder_train, fileName)
            else:
                fullOutPath = os.path.join(outFolder_validation, fileName)

            shutil.copyfile(fullInputPath, fullOutPath)


    for aspectRatioGroup in AspectRatioGroups_names:
        positiveExamplesFolder = OutputFolders["POSITIVES"][aspectRatioGroup]
        negativeExamplesFolder = OutputFolders["NEGATIVES"][aspectRatioGroup]

        # Take the files in the directories and just shuffle them
        negative_files = os.listdir(negativeExamplesFolder)
        positive_files = os.listdir(positiveExamplesFolder)
        random.shuffle(negative_files)
        random.shuffle(positive_files)

        out_negatives_train         = OutputFolders_train["NEGATIVES"][aspectRatioGroup]
        out_negatives_validation    = OutputFolders_val["NEGATIVES"][aspectRatioGroup]
        out_positives_train         = OutputFolders_train["POSITIVES"][aspectRatioGroup]
        out_positives_validation    = OutputFolders_val["POSITIVES"][aspectRatioGroup]

        saveclassFiles(negative_files, train_percent,   inFolder=negativeExamplesFolder,
                                                        outFolder_train=out_negatives_train,
                                                        outFolder_validation=out_negatives_validation)

        saveclassFiles(positive_files, train_percent,   inFolder=positiveExamplesFolder,
                                                        outFolder_train=out_positives_train,
                                                        outFolder_validation=out_positives_validation)

    # Delete the previous folders
    """
    shutil.rmtree(negativeExamplesFolder)
    shutil.rmtree(positiveExamplesFolder)
    """

if __name__ == "__main__":
    # Read dataset rows one by one and process each entry

    onlyOutputStats = False # Should I recreate folders with pictures or just draw statistics ?
    onlySplitFolders = False # True only if you want to split the already processed folders

    if onlyOutputStats is False:
        recreateImagesFolders(onlySplitFolders=onlySplitFolders)

    if not onlySplitFolders:
        processExamples(needsAugmentation=False, onlyOutputStats=onlyOutputStats)

    splitDataset()
