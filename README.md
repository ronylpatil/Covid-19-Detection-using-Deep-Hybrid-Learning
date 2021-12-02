# Covid-19-Detection-WebApp-using-Deep-Hybrid-Learning

##### Profile Visits :
![Visitors](https://visitor-badge.glitch.me/badge?page_id=ronylpatil.Covid-19-Detection-WebApp-using-Deep-Hybrid-Learning&left_color=lightgrey&right_color=red&left_text=visitors)

<p align="center">
  <img class="center" src ="https://www.uab.edu/news/images/2018/COVIDEvolution.jpg" alt="Drawing" style="width: 1350px; height: 600px">
</p>

<b>Description : </b> Covid-19 Detection from MRI's or CT Scan Images using Deep Hybrid Learning Technique. Here in this project I am combining best of the both world's, one is traditional Machine Learning and Deep Learning to create a solution that works amazing well specially when we have limited training dataset. Here I used VGG16 pretrained network for extracting usefull features of image dataset and finally used XGBOOST for classifying the images.

 XGBoost is optimized version of gradient boosting and it is a much evolved version of random forest, actually XGBoost optimize speed 
                       and possibly accuracy. Suppose we don't have millions or thousands of images that required for deep learning then I found 
                       that the accuracy that I got with tens of images with XGBoost is far superior to what we would get with deep learning. 
                       So anytime if we work with limited data, always think about XGBoost as first option and if it doesn't work great then of 
                       course try deep learning. I am sure that sometime deep learning will not be enough if we have limited data. We can also 
                       engineer our own feature extractor but here I'm going to use VGG16 pretrained architecture which make it easy for us to extract these 
                       features without defining alot of code.
                       
At the end these fusion trained 
                       on 1000 MRI's or CT scans images (500 each COVID-19 & NORMAL) and finally achieved 98.32% accuracy and it took only 21.89 seconds to 
                       train this model. This fusion learning reduce the training time of our model, in general deep larning
                       it usually takes a lot of time for training a model, but this hybrid technique reduced the training
                       time and at the same time gave very good accuracy.

<b>Network Architecture : </b>
<p align="center">
  <img class="center" src ="/main/vgg.png" alt="Drawing" style="width: 900px; height: 500px">
</p>

<b>Confusion Matrix : </b>
<p align="center">
  <img class="center" src ="/main/confusion matrix.png" alt="Drawing" style="width: 500px; height: 400px">
</p>

<b>Classification Report : </b>
<p align="center">
  <img class="center" src ="/main/classification report.png" alt="Drawing" style="width: 400px; height: 170px">
</p>

<b>Heroku App : paste link here</b><br>
<b>Dataset Source : https://www.kaggle.com/fusicfenta/chest-xray-for-covid19-detection</b>

<b>Sample Output : </b>
<p align="center">
  <img class="center" src ="/main/Image 6.png" alt="Drawing" style="width: 1400px; height: 800px">
</p>
