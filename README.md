# Emotion Recogniser

## Abstract :
Ever wondered what emotions are ? Why do we feel the need to recognise or even know them ?
Emotions are reactions that human beings experience in response to events or situations.Emotions depict a person's present state of mind.
My project is user-friendly,responsive,easy to use web application "E-Care" which predicts a person's current emotion.It gives the user a flexible choice whether they want to depict their emotion or emotion along with name just by going through one extra step of form completion.

## Introduction :
Recently there has been a great upsurge in the demand of people's necessities.In this somewhere or the other mental health of a person gets affected and there comes this wonderful technology of Artificial Intelligence - Emotion Recognition.Emotion recognition provides benefits to many institutions and aspects of life. It is useful and important for security and healthcare purposes. Also, it is crucial for easy and simple detection of human feelings at a specific moment without actually asking them.The aim of my website "E-Care" is to build an online Emotion with Name Detector which enables user to anticipate their feeling and to give them suggestions which will lift up their mood in form of pop-up boxes. The web-app is accessible with a simple UI and a lot of features.

## Problem Statement :
Develop a browser-based application or a native mobile application to demonstrate application of Face Recognition technology.You could choose to demonstrate application of Face Recognition in any area of application of your choice.You are free to choose any Face API you are comfortable with.
## Keywords :
*Facial Recognition*-a way of identifying or confirming an individual’s identity using their face in photos , videos, or in real-time.
*Emotion Recognition*-a way of detecting human feelings using their face in photos , videos , or in real-time without actually asking them.
Further there are 2 alternatives-
1)Emotion Display
2)Emotion with Name Display

# Basic Workflow :

<img width="1302" alt="Screenshot 2022-05-27 at 5 02 01 PM" src="https://user-images.githubusercontent.com/79802180/170691430-6468820f-f4b1-4225-8e49-6bb615e89ce0.png">

# System Architecture :
Firstly the weights of the trained model are sent to the flask server.
The site provides us with 2 flexible features-Emotion button and Emotion with Name button.  
On pressing Emotion with Name button it directs us to a form page asking user to upload image and their name.This image is then saved in Flask database with name of file as name of user.  
On submitting it takes us to the Emotion with Name page which with the help of images in the flask database in flask recognizes the person and simultaneously takes the weights from flask to predict the emotion of the person.  
On pressing the Emotion button it takes the weights from flask to predict the emotion of person.

![System_Architecture](https://user-images.githubusercontent.com/79802180/170697555-56f609cf-56f3-4d5b-b489-46960c79f5ed.jpg)

# System Description

## UI UX of Web-App :
In this section you can see the UI/UX of my web app.The colours used are based on my knowledge of colour psychology.  
The primary colour is blue because "Blue" represents 'trust' , 'secure' , 'faith' , 'responsibility' .Thus this app provides trustworthy information and the users can trust this website and its E-Care's responsiblity to look after their users and make them feel secure and safe.  
The font used is "Lato" . It's a San-Serif Font signifying something minimal,friendly and modern, just the message which I wanted to convey through my website .  
The name of the website "E-Care" also holds an internal significance .The 'E' in it stands for 'Emotion' and 'Online' i.e we aim to assist the users from being distant from them.

You can see my Website UI below. You can view my low-fidelity [here](https://www.figma.com/file/1nflF1xM9e6MWWnuANtmXR/Engage?node-id=0%3A1) 

### High Fidelity of website

<img width="947" alt="Screenshot 2022-05-27 at 8 37 22 PM" src="https://user-images.githubusercontent.com/79802180/170727108-b679b476-02a5-41b5-afeb-4ee415afcdaf.png">
<img width="604" alt="Screenshot 2022-05-29 at 11 04 19 AM" src="https://user-images.githubusercontent.com/79802180/170853874-6030901c-5455-47fd-a903-6b2849139bae.png">


### Colour Palatte and Font

<img width="678" alt="Screenshot 2022-05-27 at 8 35 52 PM" src="https://user-images.githubusercontent.com/79802180/170726835-8435d7e9-11b8-4804-9aec-549b1b2ef51d.png">

### Moodboard

<img width="1193" alt="Screenshot 2022-05-27 at 7 36 01 PM" src="https://user-images.githubusercontent.com/79802180/170715965-6156a58f-34da-4e95-a198-0fe2484bd1fb.png">
<img width="1440" alt="Screenshot 2022-05-27 at 7 38 22 PM" src="https://user-images.githubusercontent.com/79802180/170715931-cbda3bcc-5593-4a05-802a-bf4af7a7a6af.png">


## Getting Started
The website contains a Home Page,a Form page,Emotion page(which displays only emotion) and Emotion with Name Page(which displays name along with emotion)
The Home Page provides you a brief introduction of what the web-app does along with 2 easy accessible buttons-Emotion,Emotion with Name.
The Emotion button will directly take you to the Emotion.html page where emotion is displayed.
The Emotion+Name button will take you to a form page which the user is required to fill up for further actions.
It also contains a brief introduction about me , my interests and qualities.
The web-app also provides a header which helps the user getting directed to a particular section and a footer through which they can contact me.
<img width="1434" alt="Screenshot 2022-05-27 at 6 11 08 PM" src="https://user-images.githubusercontent.com/79802180/170703944-9726664e-8619-4d39-82fb-15fd403900d9.png">
<img width="1432" alt="Screenshot 2022-05-27 at 6 11 58 PM" src="https://user-images.githubusercontent.com/79802180/170702026-54fd6d6a-dd1d-4bc9-8d9f-b8781222bf3f.png">
<img width="1436" alt="Screenshot 2022-05-27 at 6 12 05 PM" src="https://user-images.githubusercontent.com/79802180/170702032-9640166f-da6b-4297-9d1e-1900d143aeed.png">

## Filling up Form
The second step is to fill up the form which accepts the name and image of a user.The user then clicks on the submit button and a JS alert is generated telling that form has been filled successfully.If the user doesn't upload all the information then a alert is generated asking them to upload all the required information and if they enter the file of wrong format(*other than jpg,jpeg and png)* an error will be generated and it will redirect the user to the same page.   
  
This image is taken up by flask and uploaded in the uploads folder inside the static folder.Thereafter we change the name of the image to the name of the username using built in function of `os module of python` and it gets saved in the flask database.The server also has a feature of unlinking the files when the total number of files become greater than the number that the server can tolerate. In this case I have kept this number as 12 but it's totally flexible depending upon server strength. When the number of files in the uploads folder becomes greater than 12 then the path of files gets unlinked and all the images in the uploads folder gets deleted.

![Group 31](https://user-images.githubusercontent.com/79802180/170712434-f745696b-c2a7-4aec-8bee-8ff8396ab482.jpg)

<img width="1351" alt="Screenshot 2022-05-28 at 8 19 02 PM" src="https://user-images.githubusercontent.com/79802180/170830605-4d6433e3-0c66-45c1-84e1-680609c173d2.png">


## Emotion Detection
This page detects the emotion of the person in front of the camera be it in photo,video or real time. It also provides some buttons corresponding to their emotion which further directs them to a pop-up box corresponding to that particular emotion with appealing illustrations and messages to lift up a person's mood.  
  
This page takes the weights of trained emotional recognition model from flask server and passes it to frame generated by OpenCV to predict the emotion and to display it on screen.
 
<img width="1440" alt="Screenshot 2022-05-27 at 6 26 41 PM" src="https://user-images.githubusercontent.com/79802180/170703569-d2921df7-4ae6-4ce8-ad9b-747e403620d4.png">

## Emotion with Name Detection 
This page detects the emotion of the person in front of camera be it in photo,video or real time along with their name . It also provides some buttons corresponding to their emotion which further directs them to a pop-up box corresponding to that particular emotion with appealing illustrations and messages to lift up a person's mood.  
  
The emotion recognition process is the same as that done in the Emotion Page. For Face Recognition, First we make an array of the face encodings and the names of the users we have stored in the database by traversing the uploads folder,then we pass these encodings to the face recognition module of python. It compares the encodings of the frames of the user in front of the camera to the encodings stored in encodings array.It predicts the user which has the same encoding as that in our array  or encoding which has minimum distance to that of user's.If there is a match the name is passed  and displayed along with emotion. If the database does not have the user's image, his name will be shown "Unknown".


<img width="1436" alt="Screenshot 2022-05-27 at 6 15 06 PM" src="https://user-images.githubusercontent.com/79802180/170703171-b561a507-76cb-4b42-888c-a278f63b6dda.png">
<img width="1431" alt="Screenshot 2022-05-27 at 6 15 23 PM" src="https://user-images.githubusercontent.com/79802180/170703694-fccb52f0-401b-4c7f-bc94-eb4c69cd63be.png">

## Model :
The Emotional Recognition Model is a 6-7 layered deep convolutional neural network trained with FER Dataset. The link to the dataset can be found [here](https://www.kaggle.com/deadskull7/fer2013). Initially I decided to train the model on all the 7 emotions but due to a very low test accuracy of 17% on test data,
We had to cut down to 4 emotions-`happy`, `sad`, `angry` , `neutral`. The final accuracy improved to 41%. The metrices choosen for the model is accuracy and the loss function is the Categorical Cross Entropy Function.

# Tech Stacks :
<img width="685" alt="Screenshot 2022-05-27 at 8 35 12 PM" src="https://user-images.githubusercontent.com/79802180/170726697-93249744-3f3a-4888-953d-7a7c9bec6aff.png">

# Setup :
* Clone the repository
* Go in the project repository and create a virtual environment using `python 3.7` in terminal
* Install the required dependencies using `pip3 install -r requirements.txt`
* All set! Run `python app.py`/`python3 app.py` and open the `server link` generated in the terminal to test the server!

# References :
* [Face Recognition](https://pypi.org/project/face-recognition/)
* [OpenCV documentation](https://docs.opencv.org/4.x/)
* [Flask Documentation](https://flask.palletsprojects.com/en/2.1.x/)
* [Tensorflow](https://www.tensorflow.org/api_docs)
* [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
