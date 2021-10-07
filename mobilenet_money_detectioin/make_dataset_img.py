import cv2
import os

# Define the input size of the model
input_size = (224, 224)

# Open the web cam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)

# Set the save directory
n_try = 1
save_path = "C:/make_dataset({})".format(n_try)

# Make sub directories if not exists
os.makedirs("{}/50$".format(save_path), exist_ok=True)
os.makedirs("{}/10$".format(save_path), exist_ok=True)
os.makedirs("{}/1$".format(save_path), exist_ok=True)

# Counting the number of collected images for each class.
index_50 = 0
index_10 = 0
index_1 = 0

# Money status variable
status = 0

while cap.isOpened():
    # Reading frames from the camera
    ret, original_frame = cap.read()
    if not ret:
        break

    # Copy the original frame
    frame_to_show = cv2.copyTo(original_frame, None)

    # Add Information on screen
    msg_mask = "Money "

    if(status == 0): msg_mask += "50$"
    elif(status == 1): msg_mask += "10$"
    elif(status == 2): msg_mask += "1$"

    cv2.putText(frame_to_show, msg_mask, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    cv2.putText(frame_to_show, "50$: {:03d}".format(index_50), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
    cv2.putText(frame_to_show, "10$: {:03d}".format(index_10), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
    cv2.putText(frame_to_show, " 1$: {:03d}".format(index_1), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)

    # Show the frame
    cv2.imshow('Capture Money', frame_to_show)

    # Press Q on keyboard to  exit
    in_key = cv2.waitKey(25)
    if in_key & 0xFF == ord('q'):
        break
    
    elif in_key & 0xFF == ord('m'):
        # Changing the money status
        status += 1
        status %= 3
        
    elif in_key & 0xFF == ord('s'):
        # Save the current frame
        path = save_path
        if(status == 0): 
            path += "/50$/t{}_{:03d}.jpg".format(n_try, index_50)
            index_50 += 1

        elif(status == 1): 
            path += "/10$/t{}_{:03d}.jpg".format(n_try, index_10)
            index_10 += 1

        elif(status == 2): 
            path += "/1$/t{}_{:03d}.jpg".format(n_try, index_1)
            index_1 += 1

        cv2.imwrite(path, original_frame)





