import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('fake_currrecy_detection.h5')

def preprocess_image_for_model(image):
    resized = cv2.resize(image, (128, 128))
    normalized = resized / 255.0
    expanded = np.expand_dims(normalized, axis=0)
    return expanded


template = cv2.imread('genuine_currency.jpeg', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp_template, des_template = orb.detectAndCompute(template, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)


MIN_MATCH_COUNT = 15  
GOOD_MATCH_PERCENT = 0.25  

while True:
    ret, frame = cap.read()

    if not ret:
        break

    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    
    matches = bf.match(des_template, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    if len(matches) > MIN_MATCH_COUNT:
        
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

       
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
           
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

           
            min_x, min_y = np.min(dst, axis=0).ravel()
            max_x, max_y = np.max(dst, axis=0).ravel()
            cropped_note = frame[int(min_y):int(max_y), int(min_x):int(max_x)]

            if cropped_note.size != 0:
               
                preprocessed_note = preprocess_image_for_model(cropped_note)

                predictions = model.predict(preprocessed_note)
                is_genuine = predictions[0][0] > 0.5
                confidence = predictions[0][0]

               
                if is_genuine:
                    cv2.putText(frame, f"Genuine Currency ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Fake Currency ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

 
    cv2.imshow('Currency Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
