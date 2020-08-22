#setting agg as bg as this will save the imgs in the bg
import matplotlib
matplotlib.use("Agg")

#importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from learningratefinder import LearningRateFinder
from clr_callback import CyclicLR
import config
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import pickle
import cv2
import sys
import os


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int , default= 0,
                help="whether or not to find optimal learning rate")

args = vars(ap.parse_args())

print("[INFO] LOADING IMAGES...")
imagePaths = list(paths.list_images(config.DATASET_PATH_COLAB))
# print(imagePaths)
data = []
labels = []

#looping over imagePaths
for imagePath in imagePaths:
    #extract the class label
    # print(imagePath)
    label = imagePath.split(os.path.sep)[-2]
    # print(label)
    
    #loading the image, converting it to RGB channel, and resize
    #to be fixed at 224*224 
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    #update the data ans labels lists 
    data.append(image)
    labels.append(label)

#converting the data and labels tp numpy array
print("[INFO] PROCESSING DATA...")
data = np.array(data, dtype="float32")
labels = np.array(labels)

#perform one-hot encoding on the labels 
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

print(labels[0])

#partition the data into training and testing 
(trainX, testX, trainY, testY) = train_test_split(data,
                                                 labels,
                                                 test_size=config.TEST_SPLIT,
                                                 random_state = 42)
                                            

#preparing the validation split from the training split
(trainX, valX, trainY, valY) = train_test_split(trainX,
                                                trainY,
                                                test_size = config.VAL_SPLIT,
                                                random_state = 84)

#initialize the training data augmentation object 

aug = ImageDataGenerator(
            rotation_range= 30,
            zoom_range = 0.15,
            width_shift_range= 0.2,
            height_shift_range= 0.2,
            shear_range= 0.15,
            horizontal_flip= True,
            fill_mode="nearest")
    

with tf.device("/gpu:0"):
    #loading the model 
    basemodel = VGG16(weights= "imagenet",
                    include_top= False,
                    input_tensor = Input(shape=(224,224,3))
                    )

    #headmodel which will be placed on basemodel            
    headmodel = basemodel.output
    headmodel = Flatten(name="Flatten")(headmodel)
    headmodel = Dense(512, activation="relu")(headmodel)
    headmodel = Dropout(0.5)(headmodel)
    headmodel = Dense(len(config.CLASSES), activation="softmax")(headmodel)

    #placing the headmodel on top of the base model
    model = Model(inputs=basemodel.input, outputs=headmodel)

    #freezing the layers of the basemodel
    for layer in basemodel.layers:
        layer.trainable = False

    print("[INFO] COMPILING MODEL ...")

    opt = SGD(lr = config.MIN_LR, momentum= 0.9)
    model.compile(loss = "categorical_crossentropy",
                optimizer = opt,
                metrics = ["accuracy"])

    #cheacking whether we have to perform finding optimal rate
    if args["lr_find"] > 0:
        print("[INFO] finding learning rate...")
        lrf = LearningRateFinder(model)
        lrf.find(
            aug.flow(trainX, trainY, batch_size= config.BATHC_SIZE),
            1e-10, 1e+1,
            stepsPerEpoch= np.ceil((trainX.shape[0]/float(config.BATHC_SIZE))),
            epochs= 20,
            batchSize= config.BATHC_SIZE
        )

        #plotting the loss for various lrs
        lrf.plot_loss()
        plt.savefig(config.LRFIND_PLOT_PATH)

        #exiting the script so as to change the lrs in the config file
        print("[INFO] learning rate finder complete")
        print("[INFO] examine plots and adjust learning rates before training")

        #exit the script
        sys.exit(0)

    stepsize = config.STEP_SIZE *(trainX.shape[0] // config.BATHC_SIZE)
    clr =  CyclicLR(
        mode= config.CLR_METHOD,
        base_lr=config.MIN_LR,
        max_lr=config.MAX_LR,
        step_size= stepsize
    )


    print(["[INFO] training netwrok..."])
    H = model.fit_generator(
            aug.flow(trainX, trainY, batch_size= config.BATHC_SIZE),
            validation_data = (valX, valY),
            steps_per_epoch = trainX.shape[0],
            epochs = config.NUM_EPOCHS,
            verbose = 1
    )


    print("[INFO] EVALUATING THE NETWORK...")

    predictions = model.predict(testX, batch_size = config.BATHC_SIZE)

    print(classification_report(testY.argmax(axis =1),
        predictions.argmax(axis =1),
        target_names = config.CLASSES))

    print("[INFO] SERIALIZING NETWROK TO {} .... ".format(config.MODEL_PATH))
    model.save(config.MODEL_PATH_COLAB)    


    #plot that plots and saves the training history
    N = np.arange(0, config.NUM_EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label= "train_loss")
    plt.plot(N, H.history["val_loss"], label = "val_loss")
    plt.plot(N, H.history["accuracy"], label = "train_acc")
    plt.plot(N, H.history["val_accuracy"], label = "val_acc")
    plt.title("Training Loss and Accuracy")
    plt.ylabel("Loss / Accuracy")
    plt.xlabel("Epoch #")
    plt.legend(loc = "lower left")
    plt.savefig(config.TRAINING_PLOT_PATH)


    # plot the learning rate history
    N = np.arange(0, len(clr.history["lr"]))
    plt.figure()
    plt.plot(N, clr.history["lr"])
    plt.title("Cyclical Learning Rate (CLR)")
    plt.xlabel("Training Iterations")
    plt.ylabel("Learning Rate")
    plt.savefig(config.CLR_PLOT_PATH)
