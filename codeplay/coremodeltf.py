

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from coremodelcommons import *

# from PIL import image

""""
with tf.device('/device:cpu:0'):
    v = tf.constant([[1,2,3],[1,2,3]])
    v = tf.reshape(v, shape=(-1,))
    s = tf.math.reduce_sum(tf.math.abs(v))
    tf.print(s)
"""


# Model definition for a piece of picture detector
# TODO: NEED TO optimize depending on the resolution
class BasicCoreModel(tf.keras.Model):
    def __init__(self, inputResX, inputResY):
        # print("Model is created...")
        super(BasicCoreModel, self).__init__(name='BasicCoreModel')
        # self.inputFormat = tf.keras.Input(shape=(img_height,img_width,3), name="inputImg")

        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                                            input_shape=(inputResX, inputResY, 3), activation="elu")
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="elu")
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="elu")
        self.conv4 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation="elu")
        self.globalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()

        self.dense01 = tf.keras.layers.Dense(64)
        self.dense1 = tf.keras.layers.Dense(1)

        lr_schedule = PARAM_LR_SCHEDULE
        self.optimizer = PARAM_OPTIMIZER
        self.loss_fn = PARAM_LOSS_FN

    def call(self, input_tensor, training=False):
        # print("Model is being called...")
        # assert self.input.shape == input_tensor.shape
        # DO NOT FORGET to use the training param correctly !

        x = input_tensor

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.globalAvgPooling(x)

        shapeL = list(x.shape[1:])
        numFlatten = np.prod(shapeL)
        x = tf.reshape(x, [-1, numFlatten])

        x = self.dense01(x)
        x = self.dense1(x)

        if not training:
            x = tf.nn.sigmoid(x)

        return x  # tf.nn.sigmoid(x)

    # @tf.function
    def train_step(self, images, labels):
        # Forward pass, keep gradients
        with tf.GradientTape() as tape:
            predictionProbs = self(images, training=True)
            loss = self.loss_fn(labels, predictionProbs)

        # Compute derivative of loss w.r.t. trainable params
        grads = tape.gradient(loss, self.trainable_weights)

        # Gradiesnts sum debugging
        # abs = tf.math.abs(grads.numpy())
        # s = tf.math.reduce_sum()
        # print(f"Gradients sum: {s}")

        predictionLabels = tf.squeeze(tf.cast(predictionProbs > PROB_THRESHOLD_POSITIVE, tf.float32), axis=1)
        # print(predictionLabels)
        # print(labels)
        acc = tf.reduce_mean(tf.cast(predictionLabels == labels, tf.float32))

        # print("train batch size ", len(images), len(labels))

        # Apply them
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss, acc

        # Returns the loss, accuracy, prediction probabilities and predicted labels (0 or 1) for these

    def evaluate(self, images, labels):
        # print("Model is being evaluated...")
        predictionProbs = self(images, training=False)
        predictionLabels = tf.squeeze(tf.cast(predictionProbs > PROB_THRESHOLD_POSITIVE, tf.float32), axis=1)
        loss = self.loss_fn(labels, predictionProbs)
        acc = tf.reduce_mean(tf.cast(predictionLabels == labels, tf.float32))

        return loss, acc, predictionProbs, predictionLabels

    @staticmethod
    def trainModel(model, modelName, train_neg_ds, train_pos_ds, validation_neg_ds, validation_pos_ds,
                   numStepsPerTrainEpoch, numStepsPerValidationEpoch, numEpochs):
        print("Number of steps per training epoch ", numStepsPerTrainEpoch)
        print("Number of steps per validation epoch ", numStepsPerValidationEpoch)

        totalSteps = (numStepsPerTrainEpoch + numStepsPerValidationEpoch)
        ratioTrainDataset = (numStepsPerTrainEpoch) / totalSteps
        ratioValidationDataset = (numStepsPerValidationEpoch) / totalSteps

        outputModelPath = os.path.join(SAVE_MODEL_PATH, modelName)

        train_loss_results = []
        train_accuracy_results = []
        eval_loss_results = []
        eval_accuracy_results = []

        # Log stuff
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir_train = TENSORBOARD_LOGS_FOLDER + "/train_" + modelName +"/" + str(time)
        train_file_writer = tf.summary.create_file_writer(logdir_train + "/metrics")
        logdir_eval = TENSORBOARD_LOGS_FOLDER + "/eval_" + modelName + "/" + str(time)
        eval_file_writer = tf.summary.create_file_writer(logdir_eval + "/metrics")

        tf.summary.trace_on()

        best_acc = None

        # Get iterator to datasets
        train_pos_ds_iter = iter(train_pos_ds)
        train_neg_ds_iter = iter(train_neg_ds)

        for epoch in range(EPOCHS):

            # Do Training
            # -------------------------------------------------------
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_acc_avg = tf.keras.metrics.Mean()
            print("Starting epoch ", epoch)

            # On each step, take the next batch (don't forget that it is repeating forever in cycle..)
            # And combine half of positive, half negatives to have good data rate
            for step in range(numStepsPerTrainEpoch):
                batch_pos = next(train_pos_ds_iter)
                batch_neg = next(train_neg_ds_iter)

                batch_images = tf.concat(values=[batch_pos[0], batch_neg[0]], axis=0)
                batch_labels = tf.concat(values=[batch_pos[1], batch_neg[1]], axis=0)
                # print(batch_labels)

                train_loss, train_acc = model.train_step(batch_images, batch_labels)

                epoch_loss_avg.update_state(train_loss)
                epoch_acc_avg.update_state(train_acc)

                if step % stepsToLog == 0:
                    # print metrics
                    print(f"Epoch {epoch} step {step}: loss = {train_loss.numpy()} acc = {train_acc.numpy()}")

            # Epoch end, add stats
            train_loss_avg = epoch_loss_avg.result().numpy()
            train_acc_avg = epoch_acc_avg.result().numpy()

            train_loss_results.append(train_loss_avg)
            train_accuracy_results.append(train_acc_avg)

            with train_file_writer.as_default():
                tf.summary.scalar('loss', data=train_loss_avg, step=epoch)
                tf.summary.scalar('acc', data=train_acc_avg, step=epoch)

            print(f"Epoch {epoch} - training ended, loss = {train_loss_avg} acc = {train_acc_avg}")
            # -------------------------------------------------------

            # Do evaluation
            # -------------------------------------------------------
            epoch_validation_loss_avg = tf.keras.metrics.Mean()
            epoch_validation_acc_avg = tf.keras.metrics.Mean()
            for step in range(numStepsPerValidationEpoch):
                validation_pos_ds_iter = iter(validation_pos_ds)
                validation_neg_ds_iter = iter(validation_neg_ds)

                # Take negative samples
                batch_eval_pos = next(validation_pos_ds_iter)
                batch_eval_neg = next(validation_neg_ds_iter)

                batch_eval_images = tf.concat(values=[batch_eval_pos[0], batch_eval_neg[0]], axis=0)
                batch_eval_labels = tf.concat(values=[batch_eval_pos[1], batch_eval_neg[1]], axis=0)

                eval_loss, eval_acc, predictedProbs, predictedLabels = model.evaluate(batch_eval_images,
                                                                                      batch_eval_labels)
                epoch_validation_loss_avg.update_state(eval_loss)
                epoch_validation_acc_avg.update_state(eval_acc)

            val_loss_avg = epoch_validation_loss_avg.result().numpy()
            val_acc_avg = epoch_validation_acc_avg.result().numpy()

            if USE_BOTH_TRAINING_AND_EVAL_ACC_STATS:
                val_loss_avg = (val_loss_avg * ratioValidationDataset) + (train_loss_avg * ratioTrainDataset)
                val_acc_avg = (val_acc_avg * ratioValidationDataset) + (train_acc_avg * ratioTrainDataset)

            print(f"Epoch {epoch} ended. evaluation: loss = {val_loss_avg} acc = {val_acc_avg}")

            # Save the best found model
            if best_acc is None or val_acc_avg > best_acc:
                print(f"Saving model with acc {val_acc_avg} because it is the new record !")
                best_acc = val_acc_avg
                model.save(outputModelPath)

            with eval_file_writer.as_default():
                tf.summary.scalar('loss', data=val_loss_avg, step=epoch)
                tf.summary.scalar('acc', data=val_acc_avg, step=epoch)
            # -------------------------------------------------------

        # Plot the training results
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training metrics - ' + modelName)
        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].plot(train_loss_results)
        axes[0].plot(eval_loss_results)
        axes[0].legend()

        axes[1].set_ylabel("Accuracy", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(train_accuracy_results)
        axes[1].plot(eval_accuracy_results)
        axes[1].legend()


