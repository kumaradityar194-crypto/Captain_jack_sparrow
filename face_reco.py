import cv2

#Face_cap=cv2.CascadeClassifier(r"c:\Users\kumar\Downloads\face_recognition_dataset (1).csv")
video_cap=cv2.VideoCapture(0)

while True:
    ret,video_data=video_cap.read()
    if ret is not None:
        
        cv2.imshow("Video_live",video_data)
        if cv2.waitKey(0)==ord("q"):
           break
    else:
        print("Something error")
    
video_cap.release()
cv2.destroyAllWindows()
    