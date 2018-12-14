# HRTFCNN
Convolutional Neural Network to Estimate HRTF for Spatial Audio

## Introduction
With the recent popularity of virtual reality, the concept of spatial audio has generated a lot more buzz in recent years. However, the problem for truly realistic spatial audio is an incredibly complex and difficult process. One of the primary methods for simulating realistic spatial audio is using HRTFs, or Head Related Transfer Functions. These functions essentially map the relationship between a sound source and how that sound propogates throughout a person's ear. Thus, HRTFs are incredibly individualized and without accurate HRTFs, spatial audio fails pretty poorly. Current state of the art methods primarily rely on measuring HRTFs using expensive and complex microphone and speaker arrays in anechoic chambers, but with the rapid advancement of machine learning in the past decade, there's likely a more efficient and cheaper alternative. 

In this project we make slight modifications on the recent paper "Personalized HRTF Modeling Based on Deep Neural Network Using Anthropometric Measurements and Images of the Ear" by Lee et. al, a few utilities for compatibility with the standardized SOFA format, as well as an interactive iPython notebook to estimate measurements.

## Utilities
One part of this project is a few tools used to load and write SOFA and Matlab files. This is mainly based on the fact that HRTFs are usually stored in these two formats. 
### CIPIC
Perhaps the most public database of HRTFs, the database created by the Center for Image Processing and Integrated Computing (CIPIC) of the University of California at Davis is incredibly simple and standardized. This project mainly focuses on the CIPIC database of HRTFs. In this dataset, we have 45 speakers with 50 elevations and 25 different azimuths. It also includes anthropomorphic measurements and images of ears which will prove to cause issues in the long run and will be explained in the Neural Network section. On the official website, the HRTFs are stored in Matlab files which allow easy storage of matricies. 

https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/
### SOFA
However, as HRTFs have gained popularity, people have felt the need to create a common convention for storage. What the entire industry has seemed to accept is the SOFA (Spatially Oriented Format for Acoustics) file format which was standardized by the AES (Audio Engineering Society). However, since this file format is relatively new, and spatial audio is somewhat of a niche focus, especially in machine learning, there isn't much support for the file format in many of the popular programming languages. Thus, in order to make this project work, we needed to develop a few utilities that can load and save SOFA files, especially since in this project we mainly used the python programming language. This choice to use python was based off a pretty recent standard for machine learning frameworks (Keras, TensorFlow, and PyTorch) for using python. It's also an incredibly simple language and in order to use Google Colab, you need python.

In these utilities you can find classes for HRTFs in which you can store elevations, azimuths, and impulses. There's is still more room for expansion to all kinds of SOFA files, but current the classes are only compatible with the CIPIC SOFA format and simple saving and storage of certain attributes. Check the `utils` folder to see the files.

https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)

## Measurement Estimation
A big part of this project is using the combination of anthropomorphic measurements and ear images to estimate HRTFs. But unfortunately, taking anthropomorphic measurements is incredibly cumbersome since you need to use a tape measure and measure about 37 various things (e.g. angles of ears). Thus I've also built a tool using an interactive step by step Jupyter notebook to estimate these measuremnts with simple pictures. In order to do so, we take 2 pictures, front and side while holding an 8.5x11in page for comparison.

From these pictures we take advantage of a technique called homography to warp the image to accurately represent measurements within the image. We can do this by selecting the four corners of the page in the image and our algorithm will flatten the page so it's actually 8.5x11 in the image and we can find the measurements of other parts. This process is the same as mobile phone page scanners. The measurements won't be super accurate but should be relatively accurate within a couple inches as long as the page is substantialy parallel to the camera plane. Essentially the notebook will prompt for the corners on the 8.5x11 page on each image and then will ask for various points (head height, neck width, etc.) and will estimate from the provided points.

Homography: https://www.wikiwand.com/en/Homography_(computer_vision)


## Convolutional Neural Network
The real meat of the project is the machine learning aspect. What's convenient about this problem and HRTFs, is that we're trying to estimate a function and neural networks, which have gained significant advancements and popularity in the past decade, are function approximators. The way HRTFs are represented, specifically in the CIPIC database, is that we measure 200 samples as impulses for various elevations and azimuths. This is really convenient since we can represent the output of the neural network as a vector of length 200.

For the input, we use the same input as given by the paper. We will use images of ears and instead of the full 37 anthropomorphic features which are too complex, we will only use 17 measurements. Thus the input is a single image, 17 measurements, the elevation, and the azimuth. We must first crop the image of the ear as well as run a Canny edge detection to essentially remove any high frequencies that don't affect the HRTF (hair, skin color, etc.). We then build this model in Keras and train the model on a Tensor Processing Unit (TPU) on Google Colab. Training takes about 5 hours. Please read the paper if you want the specific neural network architecture.

Also note that our model is different from the paper in which we maintain a single neural network for the entire dataset, whereas the paper decides to do a single neural network per combination of azimuths and elevations.

A huge issue with the CIPIC database is missing measurements for anthropomorphic measurements and images of ears. As a result the 45 subjects cut down to 32. Since our model is so reliant on images, this setback is incredibly detrimental. Future work to help this is described below.

Results: RMLSE (Room mean log square error) for our model= -24.285 RMLSE average = −19.23 and the papers model = −18.40 This seems to have better performance than the average hrtf among the CIPIC subjects but there's still much improvement necessary and a specific test with many real subjects would show subjective accuracy. 

The paper:
https://www.mdpi.com/2076-3417/8/11/2180/pdf

## Instructions
Feel free to clone the repo and remember to run `pip install -r requirements.txt` and install jupyter to run the notebooks. If you're trying to train you'll likely want to train online so the link below on Google Colab lets you train on a free TPU for 12 hours at a time.

Use the model and utils to generate a sofa file and feel free to load it in something like Max MSP. IRCAM SPAT is an incredibly helpful tool. The spat5.binaraul object lets you load in a sofa file and move audio around in various azimuths and elevations.

## Plans to improve
Since training takes 5 hours each time, there hasn't been much time to finetune the model for training. Google Colab tends to disconnect often and cut off training so in order to improve the model we could train on AWS or GCP instances. Also, in order to deal with the sparse data problem, we could use common image augmentation techniques and ZCA to preprocess the images and create more data. These techniques include random cropping in the images, rotating, brightening, etc. As a result we are only focusing on the contours of the ear, not necessarily the orientation or darkness of the image. As explained before, other future work will focus on expanding the utilties for SOFA support in python since there seems to be a lot of machine learning potential for HRTFs in the future. Finally, we can improve the anthropomorphic measurements by automatically detecting the page and automatically detecting the size of each measurement instead of manual points. This could speed up the process significantly since a lot of time is spent generating the HRTFs and training the model.

In order to walk through the entire training process follow this link
https://colab.research.google.com/drive/1YjlgEzn3wjde6VCa5mpQx4DTGrymTJgo

