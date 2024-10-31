import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('fake_currrecy_detection.h5')

cap = cv2.VideoCapture(0)


class_labels = ['Fake', 'Real']

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    processed_frame = cv2.resize(frame, (224, 224)) 
    processed_frame = processed_frame.astype('float32') / 255.0 
    processed_frame = np.expand_dims(processed_frame, axis=0)  

    
    predictions = model.predict(processed_frame)

    
    predicted_class_index = np.argmax(predictions[0])
    currency_name = class_labels[predicted_class_index %2]  

   
    cv2.putText(frame, currency_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()