import pandas as pd
import numpy as np
from  tqdm import  tqdm 
# from keras.preprocessing import image
import cv2
import sys

test  = pd.read_csv("/home/lohith/Documents/projects/Group_Classification/output/Test.csv")

# print(test.shape)
test_img = []
# image = cv2.imread("/home/lohith/Documents/projects/Group_Classification/Test Data/Img3968.jpg")

for i in tqdm(range(test.shape[0])):
    imagePath = "/home/lohith/Documents/projects/Group_Classification/Test Data/" + test["Filename"][i] 


    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))

    test_img.append(image)

test = np.array(test_img)

model.load_model("/content/drive/My Drive/Group_Classification/output/model.data-00000-of-00001")

prediction = model.predict(test) 
final_pred =  lb.inverse_transform(prediction)
test  = pd.read_csv("/content/drive/My Drive/Group_Classification/output/Test.csv")
test ["Category"] = final_pred
test.Category = test.Category.map({"group of adults": "Adults", "group of babaies": "Toddler", "group of teenagers":"Teenagers"})
test.to_csv("/content/drive/My Drive/Group_Classification/output/Submission_VGG16.csv")
