##**Advanced Lane Finding Project**

## CHANGES AFTER SUBMISSION
###Two main points were made during submission

1.- Distance represented in the video wasn't taken from the center. This has been updated. Has been updated and can be seen in the video. 
2.- Radius were too big to be "real". Pixels per meter wasn't taken in account in calculate radius original function. Has been updated and can be seen in the video.
3.- There were some points in the video where we lost the lanes. This has been updated, smoothness of the curve has been changed ( 10 frames are smoothed together) and method for calculating polyfit has been improved in those frames were we lost control, now we use old frame in case we don't have points well distributed to calculate polyfit. i.e If we have points concentrated in one height of the image, we cannot rely on polyfit, so we'll use last frame. Now, video is a bit better in terms of stability. 

VIDEO CAN BE SEEN HERE!!! ->  
_______________________________________________________________
ORIGINAL WRITE UP BELOW THIS POINT

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images_for_write_up/chess.png "Undistorted"
[image2]: ./images_for_write_up/undst1.png "Undistorted"
[image3]: ./images_for_write_up/thresh1.png "Undistorted"
[image4]: ./images_for_write_up/undst3.png "Undistorted"
[image5]: ./images_for_write_up/thresh2.png "Undistorted"
[image6]: ./images_for_write_up/beye1.png "Undistorted"
[image7]: ./images_for_write_up/beye5.png "Undistorted"
[image8]: ./images_for_write_up/lines1.png "Undistorted"
[image9]: ./images_for_write_up/lines2.png "Undistorted"
[image10]: ./images_for_write_up/final_image.png "Undistorted"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this example is contained in 01 - Camera_calibration

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Camera calibration is used for later, so I can run the program without calibrating the camera againg

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

I used function which loads matrix previously calculated and use it to obtain an image without the distortion effect of the lens. 

	def undistort_image(img):
    dist =  np.load('./assets/dist.npy')
    mtx =  np.load('./assets/mtx.npy')
    out = cv2.undistort(img, mtx, dist, None, mtx)
    return out
    
    
![alt text][image2]
![alt text][image4]


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of different methods described during this part of the course: 

1.- Undistort image
2.- Calculate S channel. Everything is calculated over S channel as we find that is the more steady way to detect lanes
2.- Apply sobel in X direction with thresh = (20,100)
3.- Apply sobel in Y direction with thresh = (20,100) 
4.- Apply sobel in XY direction with thresh = (20,100) 
5.- Apply a direction thresholding using thresh = (0.7,1.3)
6.- Thresholded image is the logical or, of the logical and of sobel in X and Y, and the logical and of sobel XY and direction thresholding. 
7.- Calculate a threshold over the S channel of the image using thresh= (170,255)
8.- Final thresholded image is the result of the logical or in step 6 and 8

![alt text][image3]
![alt text][image5]
####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transformer is called birds_eye, because is basically a birds eye view of the road. 

	def birds_eye(img):

     # Get image dimensions
    (h, w) = (img.shape[0], img.shape[1])
    # Define source points
    src = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
    # Define corresponding destination points
    dst = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    img_size = (img.shape[0],img.shape[1])
    
    out = cv2.warpPerspective(img, M, (w, h))
    
    return out


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 564, 460      | 100, 0        | 
| 716, 450      | 1180, 0      |
| -100, 720     | 100, 720      |
| 1380, 720      | 1180, 720        |



![alt text][image6]
![alt text][image7]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After thresholding the image, I calculate which points of the line belong to side of the road. I perform this using the sliding ROI method. This is performed in extract_lines_new under 003 - Project

1.- Given a thresholded image
2.- Separate lines in left and right
3.- In case enough points are found, use old frame. This should avoid probles
3.- Calculate polyfit of second grade of each line
4.- Smooth the polyfit averaging with the last 10 frames
5.- Calculate curvature


![alt text][image8]
![alt text][image9]

In green we can see the polyfit line. You can check a video here -> https://youtu.be/7-xgBveeoMo





![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this curverad function inside 003 - Project. 

Basically we calculate the radius of the curve using formula given during course. 

	def curverad(curve,y_eval):
    	y_eval = np.max(y_eval)
    	curvature_radius = ((1 + (2*curve[0]*y_eval + curve[1])**2)**1.5) / 				np.absolute(2*curve[0])
    	return curvature_radius
        
Curve radius is represented on the frames of final video.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is an example result of my complete pipeline

![alt text][image10]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

You can check my video here -> https://www.youtube.com/watch?v=zMekgFItxzE&feature=youtu.be

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I use a simple approach to tackle this project, basically taken in account this is my last month and I am willing to advance into the project to apply what I've learned to smart agro. 

I think that my project needs a code ordering, I recognize isn't in best shape. Moreover, some sanity checks could be entered to improve the recognition of the lines, for example: 

- Distance of lines
- Radius of both lines
- Improve curve recognition using old frames when points in the line aren't enough distributed. 

