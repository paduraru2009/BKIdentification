from coremodelcommons import *
from coremodeltf import *
from coremodeldatasetutils import *
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import os

class CoreModelMain():
    def __init__(self):
        pass
        self.specs = None
        self.datasets = None
        self.models = {}

    # Loads all the datasets all returns a dictionary for each type with sizes, plus a specification for sizes and ratios loaded
    def loadDatasets(self, basePath):
        datasets = {}

        specs = []
        for aspectRatioRange in AspectRatioGroups_names:
            tupleSpec = (aspectRatioRange[0], aspectRatioRange[1])
            resX, resY = findResByAspectRatioSpec(aspectRatioRange)

            print(f"## Loading spec {tupleSpec}")
            train_neg_ds, train_pos_ds, validation_neg_ds, validation_pos_ds, trainMaxSize, evalMaxSize = CoreModelDatasetHelper.loadDatasets(
                basePath, tupleSpec, resX, resY)

            datasets[tupleSpec] = {}
            datasets[tupleSpec]['train_neg_ds']         = train_neg_ds
            datasets[tupleSpec]['train_pos_ds']         = train_pos_ds
            datasets[tupleSpec]['validation_neg_ds']    = validation_neg_ds
            datasets[tupleSpec]['validation_pos_ds']    = validation_pos_ds
            datasets[tupleSpec]['trainMaxSize']         = trainMaxSize
            datasets[tupleSpec]['evalMaxSize']          = evalMaxSize

            specs.append(tupleSpec)

        self.specs = specs
        self.datasets = datasets

    def getDataset(self, res, aspect_x, aspect_y):
        return self.datasets[(res, aspect_x, aspect_y)]

    def getLoadedSpecs(self):
        return self.specs

    def plotSampleFromDataset(self, aspect_x, aspect_y):
        # Some shape sanity checks..
        tupleSpec = (aspect_x, aspect_y)
        train_neg_ds = self.datasets[tupleSpec]['train_neg_ds']
        train_pos_ds = self.datasets[tupleSpec]['train_pos_ds']
        validation_neg_ds = self.datasets[tupleSpec]['validation_neg_ds']
        validation_pos_ds = self.datasets[tupleSpec]['validation_pos_ds']
        trainMaxSize = self.datasets[tupleSpec]['trainMaxSize']
        evalMaxSize = self.datasets[tupleSpec]['evalMaxSize']

        datasetsToOutput = { 'train_neg_ds' : train_neg_ds,

                             'train_pos_ds' : train_pos_ds,
                             'validation_neg_ds' : validation_neg_ds,
                             'validation_pos_ds' : validation_pos_ds,
                            }

        print(f"##### Showing sample from dataset spec: {tupleSpec}")
        for dataset_name, dataset_data in datasetsToOutput.items():
            print(f"#### Dataset {dataset_name} \n ")
            # Print on screen a few results
            plt.figure(figsize=(10, 10))
            plt.title(dataset_name)
            for images, labels, file_path in dataset_data.take(1):
                for i in range(half_batch_size):
                    print(f"{i}, S:{images[i].shape}, L:{labels[i].numpy()}, P:{file_path[i]}")

                for i in range(half_batch_size):
                    ax = plt.subplot(4, half_batch_size/4, i + 1)
                    scaledBackImg = images[i]  # (images[i] * scaleFactor + scaleMean).numpy().astype("uint8"))
                    plt.imshow(scaledBackImg)
                    #plt.title(classNames[labels[i]].numpy())
                    plt.axis("off")
                break
            plt.show()

    def trainDatasets(self):
        # %%

        # Training the models
        for spec in self.specs:
            modelName = str(spec)
            resX, resY = findResByAspectRatioSpec(spec)

            print(f"#### Training model: {modelName} #### \n\n")
            with tf.device(DEVICE):
                model = BasicCoreModel(resX, resY)
                model.build((None, resX, resY, 3))
                model.summary()

                train_neg_ds = self.datasets[spec]['train_neg_ds']
                train_pos_ds = self.datasets[spec]['train_pos_ds']
                validation_neg_ds = self.datasets[spec]['validation_neg_ds']
                validation_pos_ds = self.datasets[spec]['validation_pos_ds']
                trainMaxSize = self.datasets[spec]['trainMaxSize']
                evalMaxSize = self.datasets[spec]['evalMaxSize']

                numStepsPerTrainingEpoch = int(math.ceil(trainMaxSize / half_batch_size))
                numStepsPerValidationEpoch = int(math.ceil(evalMaxSize / half_batch_size))

                model.trainModel(model, modelName, train_neg_ds, train_pos_ds, validation_neg_ds, validation_pos_ds,
                                 numStepsPerTrainingEpoch, numStepsPerValidationEpoch, EPOCHS)

                self.models[spec] = model

    def loadExistingModels(self, modelsBasePath):
        self.reconstructed_models = {}
        for spec in AspectRatioGroups_names:
            fullModelPath = os.path.join(modelsBasePath, str(spec))
            assert os.path.exists(fullModelPath)
            modelForSpec = tf.keras.models.load_model(fullModelPath)
            print(f"###Loaded model spec {spec}")
            modelForSpec.compile(optimizer=PARAM_OPTIMIZER, loss=PARAM_LOSS_FN)
            modelForSpec.summary()

            self.reconstructed_models[spec] = modelForSpec

    # Predicts the results for an image batch using the model sent as parameter
    # Returns for each image in the batch the predicted probability of having BK and label
    def predictImageBatch(self, modelForSpec, images):
        predictedProbs = modelForSpec(images)
        predictedProbs = tf.squeeze(predictedProbs, axis=1)
        predictedLabels = tf.cast(predictedProbs > PROB_THRESHOLD_POSITIVE, tf.float32)
        predictedProbs = predictedProbs.numpy()
        predictedLabels = predictedLabels.numpy()
        return predictedProbs, predictedLabels

    def getModelByAspectRatio(self, aspectRatio):
        return self.reconstructed_models[aspectRatio]

    # Testing first some positive sample...
    def inferenceDemo(self, aspectRatioName):
        assert isinstance(aspectRatioName, tuple)

        modelForSpec = self.getModelByAspectRatio(aspectRatioName)
        datasetsAndName = [(self.datasets[aspectRatioName]['train_pos_ds'], 'train_pos_ds'),
                           (self.datasets[aspectRatioName]['train_neg_ds'], 'train_neg_ds'),
                            ]

        for dataset, datasetName in datasetsAndName:
            titleName = f"=============== Spec: {aspectRatioName} - {datasetName}"
            print(titleName)
            fig = plt.figure(figsize=(16, 16))
            fig.suptitle(titleName)

            for images, labels, filepath in dataset.take(1):
                predictedProbs, predictedLabels = self.predictImageBatch(modelForSpec, images)
                true_labels = labels.numpy()

                for i in range(32):
                    ax = plt.subplot(8, 8, i + 1)
                    plt.imshow(images[i])

                    title = f"P:{predictedProbs[i]:.2f}"
                    isCorrect = classNames_lst[int(true_labels[i])] == classNames_lst[int(predictedLabels[i])]
                    correctnessColor = "green" if isCorrect else "red"
                    plt.title(title, color=correctnessColor)
                    plt.axis("off")

            plt.show()


if __name__ == "__main__":
    coreModelMain = CoreModelMain()
    coreModelMain.loadDatasets("") # "/Users/ciprian/Downloads/SplittedDatasets"


    #coreModelMain.trainDatasets()

    coreModelMain.loadExistingModels("./demoModels")
    coreModelMain.inferenceDemo((1,1))
    coreModelMain.inferenceDemo((1,2))
    coreModelMain.inferenceDemo((2,1))






