#Backend code for accepting input from user and giving output

#Importing all the required libraries for the backend and face recognition
from flask import Flask,redirect,url_for,render_template,request,Response,flash
import cv2
import numpy as np
import face_recognition
import os

#importing tensorflow to load the weights of model.py file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import model_from_json  
from tensorflow.keras.preprocessing import image  


#flask object
app=Flask(__name__) 

#setting secret key for keeping client sessions secure
app.secret_key = "secret key"

#configuring the upload folder to upload the file the user fills in the form
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#loading the model weights
model = model_from_json(open("fer.json", "r").read())  #load model  
model.load_weights('fer.h5')                            #load weights

camera = cv2.VideoCapture(0) #open cv feature to open camera
# detector = cv2.CascadeClassifier('/Users/ishitagarg/Desktop/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


#max file count the server could witheld for proper functioning of our site
MAX_FILE_COUNT=12 #I kept it 12 as an example. The parameter is set depending on server load
    
#function to let the form only take '.jpg','.jpeg','.png'
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#setting the path of where all uploaded files are kept
image_path=app.config['UPLOAD_FOLDER']



# Initialize variables for storing facial features and names
known_face_encodings = []
known_face_names = []
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True



# function which recognizes you and your emotion
def face_emo():       
    #loads all the images in flask database(uploads folder) to the face recognition module
    for img in os.listdir(image_path):
        image_person = face_recognition.load_image_file(os.path.join(image_path,img))
        image_face_encoding = face_recognition.face_encodings(image_person)[0]
        known_face_encodings.append(image_face_encoding)
        known_face_names.append(os.path.splitext(img)[0])
    while True:
        # Capture frame by frame
        success, frame = camera.read()
        labels=[]
        if not success:
            break
        else:
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            
            #looping through all face encodings
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
                # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    #puts text below rectangle
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            #converts frame to gray image for better processing of the model
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            #detects faces in the frame
            faces_detected = detector.detectMultiScale(gray_img, 1.32, 5)  
                
            #looping through coordinates of all face detected
            for (x,y,w,h) in faces_detected:
                    
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
                roi_gray=gray_img[y:y+w,x:x+h]          #cropping region of interest i.e. face area from  image  
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                 
                
                if(np.sum([roi_gray])!=0): 
                    #some preprocessing commands before going into the model
                    roi=roi_gray.astype('float')/255.0 
                    roi = image.img_to_array(roi)  
                    roi = np.expand_dims(roi, axis = 0)  
                    #getting the predictions from the model we trained
                    predictions = model.predict(roi)[0]  
                    #taking out the index with the max probability  from the predicted labels
                    max_index = np.argmax(predictions)
                
                    emotions = ["Angry",  "Happy", "Sad", "Neutral"]  
                    predicted_emotion = emotions[max_index]  
                        
                    cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 5)  
               
                    resized_img = cv2.resize(frame, (1000, 700))  
                
            ret, buffer = cv2.imencode('.jpg', frame)
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            
# Emotion detection of person
def emo():
    #this page recognises only the emotion
    while True:
        # Capture frame by frame
        success, frame = camera.read()
        labels=[]
        if not success:
            break
        else:
          
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            
            faces_detected = detector.detectMultiScale(gray_img, 1.32, 5)  
                
            
            for (x,y,w,h) in faces_detected:
                    
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
                    roi_gray=gray_img[y:y+w,x:x+h]          #cropping region of interest i.e. face area from  image  
                    roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                    
                    if(np.sum([roi_gray])!=0): 
                        roi=roi_gray.astype('float')/255.0 
                        roi = image.img_to_array(roi)  
                        roi = np.expand_dims(roi, axis = 0)  
                        
                
                        predictions = model.predict(roi)[0]  
                
                        max_index = np.argmax(predictions)   #find max indexed array
                
                        emotions = ["Angry",  "Happy", "Sad", "Neutral"]  
                        predicted_emotion = emotions[max_index]  
                        
                        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,0), 5)  
                
            resized_img = cv2.resize(frame, (1000, 700))  
                
            ret, buffer = cv2.imencode('.jpg', frame)
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    
#routing the flask app to our home page
@app.route('/')
def index():
        return render_template('index.html')

#Function which renders form page to seek data from it
@app.route('/form',methods=['POST','GET'])
def upload():
    '''
    First we check if the number of files in the flask database
    are less than the max files it can hold
    If it is more we clear the database so that the server doesn't lead to 
    a lot of files getting loaded thus detoriating the server and the frames
    '''
    count=0
    for path in os.listdir(image_path):
    # check if current path is a file
        if os.path.isfile(os.path.join(image_path, path)):
            count += 1
    if(count>MAX_FILE_COUNT):
        for img in os.listdir(image_path):
            os.unlink(os.path.join(image_path,img))
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        name=request.form['input_name']
        file = request.files['file']
        # check if the user has even uploaded the file or not
        print(file.filename)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        #check if the filename is uploaded or not
        if file and allowed_file(file.filename):
            filename = file.filename
            filename=name+os.path.splitext(filename)[1]
            #saves the file in upload folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('Emotion_Name'))
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
            
    return render_template('form.html')





#routes to different pages
@app.route('/Emotion')
def Emotion():
    return render_template('Emotion.html')
@app.route('/video_feed_emotion')
def video_feed_emotion():
    return Response(emo(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feed_name')
def video_feed_name():
    return Response(face_emo(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/Emotion_Name')
def Emotion_Name():
    return render_template('Emotion_Name.html')


if __name__=='__main__':
    app.run(debug=True)