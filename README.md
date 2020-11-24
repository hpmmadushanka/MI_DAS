Technologies used in - Machine Learning, Image Processing, Neural Networks

# FYP (Final Year Projet)

This is a Driver Assistance System you can use in your Vehicle with a mono dash camera. 


Requirements (Configurations)

1. Tensorflow-GPU
2. Python 



step 1 - Clone this repository (Our main repository)
step 2 - clone https://github.com/dbolya/yolact.git to an another folder ( Because we need the YOLACT segmentation to our project)

Copy the below folders and files from the second repository to our repository.
1. Layers
2. Utils
3. backbone.py
4. run_coco_eval.py
5. train.py
6. yolact.py


Then in the main directory you need to create a folder named "weights" and you should download the weights from the internet to this "weights" folder. 
The following links for the weights downloads have been given below.

|Image Size| Backbone| FPS| mAP| Weights|
|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|
|550|Resnet50-FPN|42.5|28.2|[yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view)| 	
|550|Darknet53-FPN|40.0|28.7|[yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing)| 	
|550|Resnet101-FPN|33.5|29.8|[yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)| 	
|700|Resnet101-FPN|23.6|31.2|[yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view?usp=sharing)| 

In this project we used - yolact_base_54_800000.pth as the weight from above mentioned ones

Finally you need to run - 

python object_tracker.py --video ./(Location of the input video) --output ./data/video/result.avi

We get the markings of the most dangerous vehicles using ML as shown in the below image.

![](images/001.jpg)


