import os 
import cv2
import cv2.data

face_Detecttor = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

folder_names = ['Angelina Jolie', 'Brad Pitt']

path = r"C:\Users\17nru\Downloads\images dataset\Celebrity Faces Dataset"

for celebrity in folder_names:
    celebrity_images = os.path.join(path, celebrity)
    print(celebrity_images)

    for img in os.listdir(celebrity_images):
        c_img = os.path.join(celebrity_images, img)

        im_cv = cv2.imread(c_img)

        roi = face_Detecttor.detectMultiScale(im_cv, 1.4, 3)

        for x, y, w, h in roi:
            cv2.rectangle(im_cv, (x, y), (x+w, y+h), (10, 255, 20), 3)

        cv2.imshow(c_img, im_cv)

        cv2.waitKey(0)

