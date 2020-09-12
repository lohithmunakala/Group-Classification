# Group Classification

Claassifies Data Based on different groups : Toddler, Teenager, Adult 
The following has been trained on ResNet152 and the validation accuracy is about 92%.

## Approach
### Input Images
I collected the data from Bing as the source for the images. Using bing_image_downloader, the required files ie. pictures of toddlers, teenagers and adults were downloaded and saved into the input folder. 
### Feature Engineering 
The images were resized into 224,224,3 and were augmented so that the number of images increases. The labels were processed and labels were created for each class. 
### Training the Model
The resized images were then trained on various models such as VGG16, VGG19, ResNet50, ResNet152. They were tried out with varying learning rates (LR’s) and different batch sizes. 

Before training the data, the models were put under different LR’s to find the best suitable LR for that given case.

Out of all the models trained, the best was found to be ResNet152 with a batch size of 64.  

Next, the model was trained on ResNet152, with a batch size of 64 and a max-LR of 10^-4 and min-LR of 10^-6  as the LR style used was Cyclic in Nature. This resulted in the model having an accuracy of 90% on the validation data.   
### Tools Used:

The libraries that were used are Tensorflow, Python, Colab for training on GPU, Numpy, Pandas, Matplotlib.  


### How to run the code:

Open the inference.ipynb file and run the cells with the test image folder and the output will be displayed in the output/submission.csv.




### Here is a graph of the network on the training data:

![alt text](https://github.com/lohithmunakala/Group-Classification/blob/master/output/ResNet152%20model/TRAINING_PLOT.png?raw=true)

We can see that both the validation and training loss are close to 0.3 and the validation and training accuracy are almost 0.9 (90% accurate). 
	
### The LR finder graph can be found here:

![alt text](https://github.com/lohithmunakala/Group-Classification/blob/master/output/ResNet152%20model/LRFIND_PLOT.png?raw=true)
 
### The Cyclic - LR plot can be found here:
 ![alt text](https://github.com/lohithmunakala/Group-Classification/blob/master/output/ResNet152%20model/CLR_PLOT.png?raw=true)
