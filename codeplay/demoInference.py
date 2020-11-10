# TODO:

# Issue 0: Show the two functions for evaluation purposes, debug things , folder with outputs and logs in demoFigs
# Present the data preprocessing and training scripts

# Issue 1: image resize
"""
# Try to double zoom to make it work on the small image patches
# It works lot better with scales_used including [2,4, ..]. For scale 8 we need a very strong 16 or 24 GB which i Highly recommend.
Why is that  ? Many zones marked in the dataset are small but we train at a zoomed scale
# We either resize the image or use small base resolution for the core model, e.g. 112 instead of 224.

Because of the scale issue, we might just need to split the images like we do not in patches and show to doctors the positive split images one by one !

Talk about tradeoff time vs resize / scale
"""

# Issue 2: Evaluation and How do we fix things ?
# It is important to have big recall ! 100% if possible
# Confirmations on negatives are actually more expensive ! explain why !
# On the precision it is important to have it big but more important is to provide a small number of patches to the medics since they will look at each and confirm or infirm
# To fix / evaluate things, First, see the output of the evaluation function testOnFolder since it can give precious informatiotions.
#  E.g. What scale did we found and what aspect ratio ? We  then need to augment more / do pre-processing for those areas, increase the dataset size and we can fit the
#   issues one by one if needed.
# General ideas for parameters: try different threhsold values, scales, aspect ratios.
# General ideas for methods: more augmentation of the dataset ! Especially ZOOM the images and RESCALE the existing images and bounding boxes !! First thing to do NEXT
                            # use more aspect ratios some with lower res  !!! show them the parameter
							# Note that if you use the base resolution as 112 instead of 224 then it is not necessarly anymore to try scale 4 or 8 !
							# These create different tradeoffs between performance of precison vs recall vs time needed !


# Issue 3: Increase the performance of the model using the TODOs on the ipynb file, see which will provide better results !


import tensorflow as tf
import os
from coremodelmain import CoreModelMain
from coremodelcommons import *
from coremodeldatasetutils import CoreModelDatasetHelper
import numpy as np
import csv
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from codeplay.coremodelcommons import classNames_lst


class BKInference:
	def __init__(self, modelsPath):
		self.coreModelMain = CoreModelMain()
		self.coreModelMain.loadExistingModels(modelsPath)
		# Get the models loaded first !
		# NOTE: KEEP THIS LOADED ALL THE TIME FOR A CLIENT SESSIOn

		self.aspect_ratio_used = [(1,1)]#, (1,2)] # 2,1 is not reliable in this case
		self.scales_used = [1, 2, 4]#, 8] # What image scale factors to use. Put more for recall, but it will take more time
		self.modelByAspectRatio = {}
		for aspect_ratio in self.aspect_ratio_used:
			self.modelByAspectRatio[aspect_ratio] = self.coreModelMain.getModelByAspectRatio(aspect_ratio)

	def debugPlotImageSplitPatches(self, image_patches, trueLabel, probabilities, predictedLabels,
								   resWidth, resHeight, savePath=None):
		assert len(image_patches.shape) == 4 # expected as batchsize, numRows, numCols, pixels for each
		nrows, ncols = image_patches.shape[1], image_patches.shape[2]
		fig = plt.figure()
		gs = gridspec.GridSpec(nrows, ncols)
		gs.update(wspace=0.01, hspace=0.01)

		for i in range(nrows):
			for j in range(ncols):
				imgIndex = i*ncols + j
				ax = plt.subplot(gs[imgIndex])
				plt.axis('off')
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				ax.set_aspect('auto')
				patch = tf.reshape(image_patches[0, i, j,], [resHeight, resWidth, 3])
				plt.imshow(patch)

				title = f"P:{probabilities[i][j]:.2f}"
				isCorrect = classNames_lst[int(trueLabel)] == classNames_lst[int(predictedLabels[i][j])]
				correctnessColor = "green" if isCorrect else "red"
				plt.title(title, color=correctnessColor)


		if savePath:
			plt.savefig(savePath, bbox_inches='tight', dpi=120)
			plt.show()
		#plt.imshow(patch)
		return fig

	# Given a patch of images computes the results (see the call to understand the data types
	def do_inference_for_image_patches(self, model, image_patches, resWidth, resHeight):
		originalShape = image_patches.get_shape().as_list()
		assert originalShape[0] == 1 # We expect a single batch
		assert originalShape[3] == resWidth*resHeight*3 # since we want the number of pixels extracted to be the expected one

		# Convert to a full batch

		totalBatchSize = np.prod(originalShape[:-1])
		#patches_results = tf.zeros(shape=(totalBatchSize, 2)) # [batch, row patch, col path, 0] = probability ,  [..., 1] = label
		image_patches = tf.reshape(image_patches, shape=(-1, resHeight, resWidth, 3)) # Put all small splitted images in a single big batch
		#print(tf.__version__)
		#image_patches = tf.concat([image_patches, image_patches], axis=0)
		#print(image_patches.shape)

		# Do inference using the model
		probabilities, labels = self.coreModelMain.predictImageBatch(modelForSpec=model,
																	 images=image_patches)

		probabilities.reshape(originalShape[:-1])  # back in [numBatch, numRows, numCols]
		labels.reshape(originalShape[:-1])
		image_patches = tf.reshape(image_patches, shape=originalShape)
		return probabilities, labels

	def	outputPredictedImage(self, originalImages, positivePatches_Rows, positivePatches_Cols, patchHeight, patchWidth):
		batch_positivePatches_bboxes_rel = []
		batch_colors = []

		for originalImage in originalImages:
			imgHeight = originalImage.shape[0]
			imgWidth = originalImage.shape[1]

			# Computes the relative positions on positive patches bboxes
			positivePatches_yCoords = [rowIndex * patchHeight for rowIndex in positivePatches_Rows]
			positivePatches_xCoords = [colIndex * patchWidth for colIndex in positivePatches_Cols]
			positivePatches_bboxes = [(row, col, row + patchHeight, col + patchWidth) for row,col in zip(positivePatches_yCoords, positivePatches_xCoords)]
			positivePatches_bboxes_rel = [(start_row/imgHeight, start_col/imgWidth, end_row/imgHeight, end_col/imgWidth) for (start_row, start_col, end_row, end_col) in positivePatches_bboxes]

			colors = [[1.0, 0.0, 0.0]]
			batch_colors.append(colors)
			batch_positivePatches_bboxes_rel.append(positivePatches_bboxes_rel)

		# Draw the images
		outputImgs = tf.image.draw_bounding_boxes(originalImages, batch_positivePatches_bboxes_rel, colors, name=None)
		return outputImgs

	# Given an image path returns an image marked with areas that are SUSPECT, the probability of each + number of bbox found as positive
	def do_BKTest_withScale(self, img_path, scaleX, scaleY, trueLabel=None, debugPlotSplit=False, outputMarkedImagesFolderPath=None):

		# Read the image
		assert os.path.exists(img_path)
		image, _, _ = CoreModelDatasetHelper.basicProcessDatasetImg(img_path, scaleX=scaleX, scaleY=scaleY, findLabelFromPath=False)
		image = tf.expand_dims(image, 0)

		bkFound = 0
		bkFound_aspectRatio = None

		for aspect_ratio in self.aspect_ratio_used:
			print(" #Trying aspect ratio ", aspect_ratio)
			resWidth, resHeight = findResByAspectRatioSpec(aspect_ratio)
			modelToUse = self.modelByAspectRatio[aspect_ratio]

			# Split the image and prepare batches for the given patch resolution
			ksizes = [1, resHeight, resWidth, 1]
			strides = [1, resHeight, resWidth, 1]
			rates = [1, 1, 1, 1]
			padding = 'VALID'

			image_patches = tf.image.extract_patches(image, ksizes, strides, rates, padding)
			nrows, ncols = image_patches.shape[1], image_patches.shape[2]

			# Compute the probability and classification label for each item in the batch
			probabilities, labels = self.do_inference_for_image_patches(modelToUse, image_patches, resWidth=resWidth, resHeight=resHeight)
			probabilities = probabilities.reshape((nrows, ncols))
			labels = labels.reshape((nrows, ncols))

			if debugPlotSplit:
				self.debugPlotImageSplitPatches(image_patches, trueLabel, probabilities, labels,
												resWidth=resWidth, resHeight=resHeight, savePath=f"splitFig_{aspect_ratio}.png")


			# Get the coordinates of the positive patches
			positivePatches_Rows, positivePatches_Cols = np.where(labels != 0)
			if len(positivePatches_Cols) == 0 or len(positivePatches_Rows) == 0:
				continue # Great, no BK inside

			# Well, we have to output the final output with marked positive patches
			bkFound = len(positivePatches_Cols)
			bkFound_aspectRatio = aspect_ratio
			outputImg = self.outputPredictedImage(image, positivePatches_Rows, positivePatches_Cols, resHeight, resWidth)

			fileName = os.path.basename(img_path)
			extensionIndex = fileName.rfind('.')

			outputExtension = fileName[extensionIndex+1:]
			outputFileName = f"{fileName[:extensionIndex]}_pos_as:{aspect_ratio,}.s:{scaleX},{scaleY}.{outputExtension}"
			fullOutputPath = os.path.join(outputMarkedImagesFolderPath, outputFileName)
			tf.keras.preprocessing.image.save_img(fullOutputPath, outputImg[0])
			break # No reason to go with other scales, we already found something here

		return bkFound, bkFound_aspectRatio

	# Returns the number of patches marked as positive, the scale used and aspect ratio that provided the positive examples
	def do_BKTest(self, img_path, trueLabel=None, debugPlotSplit=False, outputMarkedImagesFolderPath=None, indexInDataset=None):
		if not os.path.exists(outputMarkedImagesFolderPath):
			os.makedirs(outputMarkedImagesFolderPath)

		if indexInDataset is None:
			indexInDataset = 0
		print(f"### Testing image index {indexInDataset}, path {img_path}")
		numFoundThings = None
		aspectRatioThatFoundThings = None
		scaleThatFoundThings = None
		for scale in self.scales_used:
			print(" #Trying scale ", scale)
			result, aspectRatioThatFoundThings = bkTest.do_BKTest_withScale(img_path=img_path,
									   			scaleX=scale, scaleY=scale,
												trueLabel=trueLabel, debugPlotSplit=debugPlotSplit,
												outputMarkedImagesFolderPath=outputMarkedImagesFolderPath)

			if result > 0:
				numFoundThings = result
				scaleThatFoundThings = scale
				#print("Found the BK !")
				break # Do not continue to evaluate other scales

		return numFoundThings if numFoundThings != None else 0, scaleThatFoundThings, aspectRatioThatFoundThings

def testOnHarcodedPaths(bkTest):
	# an image and its ground true label
	img_path1 = "/home/ciprian/Downloads/Zaya/codeplay/OrigDataset/9216_55296_5b7be338-c94d-46bc-9824-21d51acd3f8c.jpeg"
	img_path2 = "/home/ciprian/Downloads/Zaya/codeplay/OrigDataset/27648_18432_49ee023a-41b6-4e09-a55f-eb37988976d4.jpeg"
	img_path3 = "/home/ciprian/Downloads/Zaya/codeplay/OrigDataset/7168_5120_13e73186-727b-4b8c-a807-41ffc01181ba.jpeg"

	img_path = img_path3

	assert os.path.exists(img_path)
	trueLabel = False # If you don;t know it, put it as None

	numFoundThings, scaleThatFound, aspectRatioThatFound = bkTest.do_BKTest(img_path=img_path, trueLabel=False, debugPlotSplit=True, outputMarkedImagesFolderPath="./demoFigs/BKTests")
	print(f"Final Result is {numFoundThings > 0}: num patches found: {numFoundThings}, scale that found: {scaleThatFound}, aspect that found: {aspectRatioThatFound}")

def testOnFolder(testFolderPath):
	datasetDescPath = os.path.join(testFolderPath, "PoleDifferentiator.csv")
	assert os.path.exists(datasetDescPath), "Couldnt find the csv datapath"

	# Go through each example and output a list of tuples [(img name, ground true label, predicted label)]
	imgNames = []
	groundTrueLabels = []
	predictedLabels = []
	additionalInfoForEachTest = [] # A collection of tuples containing additional debug info

	start_time = time.time()
	with open(datasetDescPath, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')

		maxIters = None
		for rowIndex, row in enumerate(reader):
			if row is None or len(row) == 0:
				continue

			if maxIters is not None and rowIndex >= maxIters:
				break

			fileName = row[0]
			groundTrueLabel = int(row[1])
			fullInputImgFile = os.path.join(testFolderPath, fileName)

			numPatchesFound, scaleThatFoundIssues, aspectRatioThatFoundIssues = bkTest.do_BKTest(img_path=fullInputImgFile, trueLabel=False, debugPlotSplit=False,
							 outputMarkedImagesFolderPath="./demoFigs/BKTests", indexInDataset=rowIndex)

			result = numPatchesFound > 0
			imgNames.append(fileName)
			groundTrueLabels.append(groundTrueLabel)
			predictedLabels.append(int(result))
			additionalInfoForEachTest.append((rowIndex, fileName,
											   groundTrueLabel, result,
											  numPatchesFound, scaleThatFoundIssues, aspectRatioThatFoundIssues))

			print("Result is: ", result)
	end_time = time.time()

	# Let's compute and plot statistics
	Y = np.array(groundTrueLabels)
	Yh = np.array(predictedLabels)
	tp = np.count_nonzero(np.logical_and(Y == 1, Yh == 1))
	fp = np.count_nonzero(np.logical_and(Y == 0, Yh == 1))
	tn = np.count_nonzero(np.logical_and(Y == 0, Yh == 0))
	fn = np.count_nonzero(np.logical_and(Y == 1, Yh == 0))
	precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
	recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
	print("Total examples run: ", len(imgNames))
	print(f"Stats: Precision:{precision:0.6f} Recall:{recall:0.6f}.")
	print(f"TP:{tp:0.2f}")
	print(f"FP:{fp:0.2f}")
	print(f"TN:{tn:0.2f}")
	print(f"FN:{fn:0.2f}")

	# Let's output things about the missed examples
	wrongIndices = np.where(Y != Yh)[0]
	print("\n\n -------------- WRONG CLASSIFIED IMAGES----------\n")
	for wi in wrongIndices:
		we = additionalInfoForEachTest[wi]
		print(f"index: {we[0]} file:{we[1]} GroundTruth: {we[2]} Pred: {int(we[3])} #PosPatches: {we[4]} scaleFound: {we[5]} asrFound: {we[6]}")

	print("Total real execution time: ", end_time-start_time)


if __name__ == "__main__":
	bkTest = BKInference("./demoModels")

	#testOnHarcodedPaths(bkTest)

	testFolderPath = "/home/ciprian/Downloads/OrigDataset_withScans"
	testOnFolder(testFolderPath)
