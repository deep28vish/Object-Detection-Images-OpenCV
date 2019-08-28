"""object detection on pics using OPENCV """
"""Dependencies are only cv2, the general idea is to call a CASCADE CLASSIFIER
which is stored in a very specific loaction mentioned in the code, the cascade is then
given a name for which it is meant for, 
IMAGE(3 channel) stored in working directory is read and resized to (1000,1000), then converted to 
GRAY IMAGE(1 channel), called cascades are applied to the GRAY_IMG,
faces, eyes, upperbody...etc returns 4 coordinate values to form a rect which is done using for loop
for multiple rect(if found),in the same for loop cv2.rectangle,cv2.putText is 
used to form a rectangle and text associated with it in the image
cv2.imshow("name of image", img) displays image for the user,
cv2.waitKey(20000) hold the image for 20 sec,
cv2.destroyAllWindows close all the open tabs of imgs. """
path = 'C:\\Users\\kuk\\OBJECT_DETECTION\\OBJECT_DETECTION_PICS\\haar_cascades'
import cv2
##############################################################################
# allocating the cascade in the system, '\\' is used instead of '\' there are mode 
# face_cascade holds the face cascade classifier and so on 
face_cascade = cv2.CascadeClassifier(path + "\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(path +"\\haarcascade_eye.xml")
upperbody_cascade = cv2.CascadeClassifier(path + "\\haarcascade_upperbody.xml")
smile_cascade = cv2.CascadeClassifier(path+"\\haarcascade_smile.xml")
profileface_cascade = cv2.CascadeClassifier(path+"\\haarcascade_profileface.xml")
lowerbody_cascade = cv2.CascadeClassifier(path+"\\haarcascade_lowerbody.xml")
fullbody_cascade = cv2.CascadeClassifier(path+"\\haarcascade_fullbody.xml")
frontalface_alt_cascade = cv2.CascadeClassifier(path+"\\haarcascade_frontalface_alt.xml")
#############################################################################
# COLOR_img is read with name in working direcory
img = cv2.imread("cr02.jpg")
# COLOR_img is resized to desired value
img = cv2.resize(img,(1000,1000))
#COLOR_IMG is converted to GRAY_IMG
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cascades are usd to get coordinate values for each time they succeed to find there objective
faces = face_cascade.detectMultiScale(gray_img, 1.05, 5)
eyes = eye_cascade.detectMultiScale(gray_img, 1.05, 5)
#upperbody = upperbody_cascade.detectMultiScale(gray_img, 1.05, 5)
#smile = smile_cascade.detectMultiScale(gray_img, 1.05, 5)
#profileface = profileface_cascade.detectMultiScale(gray_img, 1.05, 5)
#lowerbody = lowerbody_cascade.detectMultiScale(gray_img, 1.05, 5)
#fullbody = fullbody_cascade.detectMultiScale(gray_img, 1.05, 5)
#frontalface_alt = frontalface_alt_cascade.detectMultiScale(gray_img, 1.05, 5)
# printing all coordinate values
print('number of faces detected:: ',faces.shape[0])

# construcuting rectangle and having texts on the same object

for x,y,w,h in faces:
    face_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
    face_text = cv2.putText(img, "FACE",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(0,255,0))
  
    
for x,y,w,h in eyes:
    eyes_img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
    eyes_text = cv2.putText(img, "eyes",(x, y+10),cv2.FONT_HERSHEY_COMPLEX_SMALL,.7,(225,0,0))
    
#for x,y,w,h in upperbody:
#    upperbody_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
#    
#for x,y,w,h in smile:
#    smile_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,100,0), 3)
#    
#for x,y,w,h in profileface:
#    profile_img = cv2.rectangle(img, (x,y), (x+w, y+h), (100,0,0), 3)
#    
#for x,y,w,h in lowerbody:
#    lowerbody_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,100), 3)
#    
#for x,y,w,h in fullbody:
#    fullbody_img = cv2.rectangle(img, (x,y), (x+w, y+h), (150,0,0), 3)
#    
#for x,y,w,h in frontalface_alt:
#    frontalface_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,150,0), 3)
    
cv2.imshow("output",img)
cv2.waitKey(20000)

cv2.destroyAllWindows()
