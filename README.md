#Advanced Lane Finding Project

---

###The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/distorted.png "Distorted"
[image2]: ./examples/undistorted.png "Undistorted"
[image3]: ./examples/thresholded.png "Thresholded"
[image4]: ./examples/warp.png "Warp Example"
[image5]: ./examples/findlines1.png "Find lines initial"
[image6]: ./examples/findlines2.png "Find lines continuous"
[image7]: ./examples/highlighted_lane.png "Highlighted lane image"
[video1]: ./project_video.mp4 "Video"

---

###[Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

#####Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Writeup / README

##### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

---

### Camera Calibration

##### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located at "./main_notebook.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image using `cv2.findChessboardCorners()`.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

The iteration of preparing "objpoints" and "imgpoints" is done for all of the calibration chessboard images found at "./camera_cal" directory.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

---

### Pipeline (single images)

##### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Distorted image][image1]

---

Using the camera calibration matrix and the distortion coefficients obtained from the previous step, I now define a function called `undistort()` using `cv2.undistort()` function from openCV that takes in a distorted image and returns its undistorted counterpart. Using this function on the above distorted image returns the following undistorted image. 

![Undistorted image][image2]

---

##### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of different color thresholds to generate a binary image (thresholding steps can be found in the 8th, 9th and 10th code cells of "main_notebook.ipynb" IPython notebook). I used S and L channels of the images in HLS color space while also using the G channel from the images in RGB color space. Thresholding S channel helped me select the yellow lane pixels while the G channel was used to select the white lane pixels. The L channel thresholding was used to limit the selection of shadow regions during selection using S channel. 

A helper function `thresh_inside()` was used on S channel and G channel where the pixels within the threshold range were selected. Also, another helper function `thresh_outside()` was used on the L channel where the pixels within the threshold range were discarded. This helped me to remove the shadow pixels that were selected from the S and G channel thresholding. 

These were then combined to form the `apply_thresholds()` function that takes in a color image, applies the combination of the above mentioned thresholds and returns a binary image where the lane pixels are white. 

Here's an example of my output for this step. 

![Thresholded binary image][image3]

---

##### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in 22nd code cell of the "main_notebook.ipynb" IPython notebook. The `warp()` function takes as inputs an image, and returns a warped image which is a bird's eye view of the input image.  

`warp()` contains a function called `cv2.getPerspectiveTransform()` that takes in a pair of arrays `src` and `dst` and returns a transformation matrix `M`. These arrays contain co-ordinate points such that each point in the `src` array maps to a point in the `dst` array such that the mapping defines how the transformation of the image happens. 

Here, I manually decided and hardcoded these points. Since we can't rely on our knowledge about the curve, I select a test image where the lane is straight to do this. By using the fact that lane lines that appear converging in the perspective image are actually parallel, I selected four points for each `src` and `dst` array since four points are enough to define a linear transformation function.  

Using the transformation matrix `M` obtained as above and an image as a input, `warp()` then uses `cv2.warpPerspective()` to transform the image to the defined view. 

```python
w,h = 1280,720
x,y = 0.5*w, 0.8*h
src = np.float32([[200./1280*w,720./720*h],
                  [453./1280*w,547./720*h],
                  [835./1280*w,547./720*h],
                  [1100./1280*w,720./720*h]])
dst = np.float32([[(w-x)/2.,h],
                  [(w-x)/2.,0.82*h],
                  [(w+x)/2.,0.82*h],
                  [(w+x)/2.,h]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720        | 
| 453, 547      | 320, 590.4      |
| 835, 547     | 960, 590.4      |
| 1100, 720      | 960, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart.

![Warp example][image4]

---

##### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The 14th and the 16th code cells of the "./main_notebook.ipynb" IPython notebook contains the code that was used to identify lane-line pixels and fit their position with a polynomial.

In the 14th code cell, we have a function `find_lines_initial()` which can be used to identify lane lines for the first time (i.e., image frame which doesn't have any previous frame). It works as follows:

  * First, we take in a binary image and calculate the histogram of the bottom half of the image. The bottom half is enough to find out the starting of the lane line at the bottom of the image. Then, we can identify the peaks in the histogram as positions of the lane lines. 
  * To distinguish between left and right lane lines, we divide the image into two parts vertically by calculating the midpoint and then looking for a peak in these two image parts. 
  * After we have base positions of where our lane lines are, we construct a window around these base positions with some margin (parameter `margin` set inside the function) as half the width while the height of the window is calculated after choosing the number of windows we wish to use.  
  * This window is used to look for white pixels. Once we identify a certain number of pixels (parameter `minpix` set inside the function) in the window, the identified pixels are appended to arrays that store the lane line pixels after being identified. This is done seperately for left lane line and right lane line since we have split the image vertically at the center. 
  * Then, after appending the `minpix` number of pixels, the center of the window is shifted to the mean value of the pixels that were identified in the previous window iteration. 
  * This is carried on until the whole height of the image has been scanned by our specified number of windows.
  * After identifying the pixels that form our lane lines, we use the co-ordinates of the pixel positions to fit a second order polynomial to each of our left and right lane-line.
  
An iteration of this can be seen in the below image:

![Finding lines for the first time][image5]

---
  
However, instead of doing the above process each time, we can use the information of the polynomials that were fitted to an image using the above process to search for lane line pixels in the next image (since in a video, the continuous frames are similar and so are the lane line positions). This is done in the 16th code cell of the IPython notebook using the `find_lines_continuous()` function, which works as follows:

  * The function takes in the coefficients of the polynomials that were fitted in the previous frame of the video, and searches for white pixels around a specified `margin` around the second order polynomial curve. 
  * The pixels that are found inside the marginal range around the curve are stored in an arary. 
  * The x and y positions of the pixels are sperated and a new seond order polynomial is fitted for these points. These are done for left and right lane line seperately.

![Finding lines continuously][image6]

---

##### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is done in the 18th code cell and 20th code cell of the IPython notebook named "main_notebook.ipynb". 

The radius of curvature is found out as follows:

  * The y-position at which the curvature is to be found out is decided as `image.shape[0]` which is basically the base of the `image` or the current position of the car.
  * Then using the coefficients of the polynomials obtained (in pixel space), we calculate the points in real world space by using conversion coefficients to go from pixel to meters using, ```ym_per_pix = 30/720 # meters per pixel in y dimension``` and ```xm_per_pix = 3.7/700 # meters per pixel in x dimension```.
  * Then, using the formula for radius of curvature, the measurements for left and right lane lines are found out.
  * The mean of the two radii of curvature is calculated as the lane's radius of curvature before being returned by the function.
  
The offset of the vehicle from the center of the lane is found out as follows:

  * At the base of the image (`y = image.shape[0]`), using the polynomial equations for the left and right lane lines, two x-positions are found which leaves us with two points on the image. These two points are positioned such that the distance between them is equal to the width of the lane that was detected. 
  * Using these points, the position of the center of the lane is found out.
  * Also, the center of the image is considered as the car's position. 
  * Hence, the distance between car's position and the center of the lane called offset is found out by calculating the difference.

##### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell number 22 in the function `get_final_output()`.  Here is an example of my result on a test image:

![Highlighted lane image][image7]

---

### Pipeline (video)

##### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://www.youtube.com/watch?v=Z6wAI8M-K68)

---

### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The first bottleneck was at the step where I had to retain optimum amount of lane line pixels and convert the color image to a binary image. The most critical part was choosing what methods to use. I chose to experiment with various color spaces and different color channel thresholds. I figured that S, L channels from HLS color space and G channel from RGB color space could be realiably used as discussed in the above sections. I particulary took time to figure out how to use the L channel to avoid shadows being picked up by the S channel thresholding. Also, I converged on a set of threshold ranges for each of the color channels after some experimentation before defining a function that combined these methods.   

Since my pipeline focuses on specifically detecting pixels with high saturation (such as yellow lines) and high Green channel value (such as white lines) while avoiding pixels with low lightness values (such as shadows), my pipeline would fail at any situation where the optimum amount of lane line pixels required to fit the polynomials correctly is not picked up. It can also fail at places where pixels not relating to the lane lines are picked up and including in the plotted polynomials. 

A few of these cases are as follows:

  * Very bright images. Here, the color channel values vary drastically and thus our static threshold ranges will not work.
  * Too much shadow. When there is too much shadow in the image, my pipeline might avoid selecting those pixels and enough pixels from the lane lines might not be picked up to correctly plot the curves.

A few improvements to the pipeline that can be used to avoid these failures are as follows:

  * Dynamic threshold range setting. By using a function to dynamically set the threshold ranges for our different color channels by either looking at the variations in brightness or histograms could be better at picking up the lane line pixels in videos where there is a lot of variations in brightness. This might also help ud deal with the heavy shadows issue.

---
