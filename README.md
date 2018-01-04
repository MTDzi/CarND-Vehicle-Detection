# Project 5: Vehicle Detection Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/non-car.png
[image3]: ./output_images/car_hog_and_not.png
[image4]: ./output_images/not_car_hog_and_not.png
[image5]: ./output_images/test1_out.jpg
[image6]: ./output_images/test2_out.jpg
[image7]: ./output_images/test3_out.jpg
[image8]: ./output_images/test4_out.jpg
[image9]: ./output_images/test5_out.jpg
[image10]: ./output_images/test6_out.jpg
[image11]: ./output_images/test6_boxes_out.jpg
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
### Writeup / README

#### 1. Provide a Writeup / README

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the **5th and 7th code cell** of the IPython notebook (the 5th cell contains the functions extracting HOG features, and the 7th actually applies them to create the feature matrix `X`).  

I started by reading in all the names of files for `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pix_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `R`-channel of the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(1, 1)`:

![alt text][image3]

and for a non-car:

![alt text][image4]


#### 2. Explain how you settled on your final choice of HOG parameters

I tried various combinations of parameters and color spaces, but settled for the HOG features extracted from the `LUV` and `HLS` color spaces, with `orientation`, `pix_per_cell`, and `cells_per_block` chosen to be equal: 8, 16, and 1, respectively.

The reason I chose `LUV` and `HLS` was that for both of them HOG representations of cars seemed to be more "emphasized" than, for example, in case of HOG images extracted from the `RGB` color space.

The reason I chose these parameters was that I wanted to produce very few features comming from HOG as possible, and focus more on color-based features (and see how good a model can I build using primarily colors).


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them)

I trained a linear SVM using the `sklearn.svm.LinearSVM` class, with default hyperparameters.

But also, I trained an eXtreme Gradient Boosted classifier using the `xgboost.XGBClassifier` class.

Both classifiers are trained in the **9th code cell** in the notebook.

I chose the `xgboost.XGBClassifier` because I felt uncomfortable choosing an arbitrary threshold for deciding which boxes do contain a vehicle and which don't. I preferred to work with a mean of estimated probabilities (the xgboost classifier has the `predict_proba` method, whereas the SVM classifier doesn't).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the **14th code cell**, in  the `search_windows()` function.

I made tests with varying window sizes, overlaps, and other variants of parameters in the **22nd code cell**, and after several iterations I settled for window sizes: 128, 86, 64, 48, and 32. The overlap for all these windows was 0.75, except for the smallest, 32x32 window for which the overlap was set to 0.5.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are the examples:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

I used the `xgboost.XGBClassifier` since it performed as fast as the `sklearn.svm.LinearSVM` model, it performed slightly better, and most importantly -- I was able to get predicted probabilities with the `predict_proba` method. The biggest boost in performance resulted from limiting the number of features (that's why I used the simplest HOG features possible).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps along with the output of `scipy.ndimage.measurements.label()` 

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image11]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**First,** my approach strongly relies on the images presented to the model during training. As a result, the model will fail to identify a vehicle whose appearance significantly differs from images in the training set (trucks, motorcycles, and other).

**Second,** I've constrained the portion of the view in which I'm looking for vehicles (this was done to limit resource requirements). But this might be easily remedied if I had unlimited time and/or resources.

**Third,** because the method relies on video/images, the method might omit vehicles due to glares, obstacles on the road (which might occur at turns, not in the project video), vehicles covering up other vehicles, etc.

**Fourth,** when two or more vehicles are side-by-side in the image, this method might fail in identifying them as separate vehicles. This might be very problematic, and I don't see a clean method of dealing with this problem in this set-up. It is, however, possible to use other approaches, capable of identifying instances of objects (YOLO would do the trick, if I understand correctly).

**Finally,** this method is *slow* -- I've read that YOLO combined with a GTX 1080 GPU can make several predictions per second!
