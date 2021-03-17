# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./wup/01_dataset.png "Visualization"
[image2]: ./wup/02_preprocess.png. "Preprocess"
[image3]: ./wup/03_newimages_beforeafter.png "New images, before/after"
[image4]: ./wup/04_detected_wrong.png "Wrong"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/amedveczki/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used native python in the Jupyter notebook to check what size of the various data sets are.

```
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43
```

#### 2. Include an exploratory visualization of the dataset.

Here are some examples based on the different sets (training, validation, test) datasets.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I chose not to convert to grayscale as information is lost during the conversion (however, that would be likely good enough). Based on some images which seemed to be completely black at first sight contained enough color information but was hard to see (and while it is likely that a NN can be trained to detect even those, we can easily help it.). Based on what I thought what should be done during preprocessing I've found the method what I needed: `histogram equalization`, https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html. This would stretch the most information between -1 and 1 from RGB values.

(My first idea was to just scale the image color values to -1, 1 based on its minimum/maximum values, but that was not good enough, I still saw completely dark pictures. With this, most pictures look much better, only some of them are less visible but still much better overall than without it.)



![alt text][image2]

I did not augment the data as the result was good enough.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x8 |
| ReLU				|												|
| Avg pooling	     | 2x2 stride,  outputs 15x15x8 	|
| Convolution 5x5	 | 1x1 stride, valid padding, outputs 11x11x25 |
| ReLU	|  |
| Dropout	| Probability: 0.5 |
| Flatten	| 11x11x25 => 3025 |
| Fully connected		| 3025 => 180 |
| ReLU	|         									|
| Fully connected	| 180 => 100 |
| ReLU |												|
| Fully connected | 100 => n_classes (43) |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

First I used the training from LeNet lab workspace (MNIST data), and I completely missed the fact that it prints validation data. I tried to tweak everything before using dropout since I thought it needs to be more than 93% before I try to use such techniques. Fortunately I realized the problem after too much time.

The Adam optimizer which I have used does it job, better than simple gradient descent, over the batch size of 128 and with the number of epochs 18. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.3% 
* test set accuracy of 95.3%

My first approach was LeNet from the from the LeNet lab. I tried to tweak its parameters (and using average pooling instead of max pooling) without changing any major structure. As I stated earlier I thought I was seeing training data, so I kept on trying. After realizing my problem and including a dropout to remove overfitting it was already good enough.

### Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web around the coordinates 

[]: https://goo.gl/maps/gPDa4aHPmc2fCk8i7	"53.5596623,9.9609098"

![alt text][image3]

I did not try to search for pictures which would be hard for the NN to recognize - I just tried to use Google StreetView in Germany. And I was shocked, I didn't know street view was banned from most of its territory. However there were still a few cities which were available and at least the traffic signs were visible and here they are.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).



| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 36 Go Straight Or Right |       Go Straight Or Right        |
| 25 Road Work |             Road Work             |
| 32 End of all speed and passing limits	| **6 End of speed limit (80km/h)**	|
| 2 Speed limit (50km/h)	| **1 Speed limit (30km/h)**	|
| 38 Keep Right	| 38 Keep Right |

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

- **End of all speed limits**: likely due to the slash in the sign (and due to the lack of training data - though I haven't checked this statistics) it was categorized erroneously as **end of speed limit 80km/h**. As it can be seen below, 32 (the "true" end of all speed limits) has very similar training pictures.
- **Speed limit 50km/h**: I believe that by extending the training data by rotating the contents +- a few degrees could help with this case.  

![image4][image4]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a go straight/right sign, and the image does contain a go straight/right sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.99961615e-01 | 36,Go straight or right |
| 3.69379013e-05 |          38,Keep right          |
| 7.98607687e-07	| 18,General caution	|
| 5.41308282e-07	|     40,Roundabout mandatory     |
| 8.99943320e-09	| 20,Dangerous curve to the right |

The full output for top 5 softmax is:

> ```
> TopKV2(values=array([[  9.99961615e-01,   3.69379013e-05,   7.98607687e-07,
>           5.41308282e-07,   8.99943320e-09],
>        [  9.80772853e-01,   1.88649427e-02,   3.19967134e-04,
>           3.73572329e-05,   1.51716392e-06],
>        [  7.95670092e-01,   1.14883341e-01,   8.31412897e-02,
>           3.23416782e-03,   1.03643036e-03],
>        [  9.99993443e-01,   5.78455683e-06,   6.34331798e-07,
>           8.60309015e-08,   5.15649246e-10],
>        [  1.00000000e+00,   2.10368375e-16,   1.93504739e-18,
>           8.87624044e-19,   9.64690536e-20]], dtype=float32), indices=array([[36, 38, 18, 40, 20],
>        [25,  1, 31,  5, 24],
>        [ 6, 32, 41, 38, 18],
>        [ 1,  0,  4, 40, 42],
>        [38, 36, 34, 40, 19]], dtype=int32))
> ```

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


