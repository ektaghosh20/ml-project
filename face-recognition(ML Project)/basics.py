import face_recognition
import cv2


# Load images
imgDeepti = face_recognition.load_image_file('ImagesBasic/deepti.jpg')
imgDeepti = cv2.cvtColor(imgDeepti, cv2.COLOR_BGR2RGB)

imgGarima = face_recognition.load_image_file('ImagesBasic/garima.jpg')
imgGarima = cv2.cvtColor(imgGarima, cv2.COLOR_BGR2RGB)

imgEkta = face_recognition.load_image_file('ImagesBasic/ekta.jpg')
imgEkta = cv2.cvtColor(imgEkta, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgEkta)[0]
encodeEkta = face_recognition.face_encodings(imgEkta)[0]
cv2.rectangle(imgEkta,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocGarima = face_recognition.face_locations(imgGarima)[0]
encodeGarima = face_recognition.face_encodings(imgGarima)[0]
cv2.rectangle(imgGarima,(faceLocGarima[3],faceLocGarima[0]),(faceLocGarima[1],faceLocGarima[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodeEkta],encodeGarima)
faceDis = face_recognition.face_distance([encodeEkta],encodeGarima)
print(results,faceDis)
cv2.putText(imgGarima,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Ekta',imgEkta)
cv2.imshow('Garima',imgGarima)
cv2.waitKey(0)




cv2.waitKey(0)
cv2.destroyAllWindows()

