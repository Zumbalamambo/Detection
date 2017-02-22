import YOLO_small_tf as yolo_tf
import cv2

# # y=h=1080,x=w=1920, 511 frames in 59 secs
# cap = cv2.VideoCapture('../datasets/TongYing/20170220/10/11.mp4')   # no customer
# y=h=1080,x=w=1920, 527 frames in 60 secs
cap = cv2.VideoCapture('../datasets/TongYing/20170220/18/35.mp4')   # with customer

yolo_detector = yolo_tf.YOLO_TF()
yolo_detector.disp_console = True
yolo_detector.imshow = False
yolo_detector.tofile_img = 'outputs/people/person.jpg'
yolo_detector.tofile_txt = 'outputs/people/person.txt'
yolo_detector.filewrite_img = False
yolo_detector.filewrite_txt = False

counter = 1
while (cap.isOpened()):
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    if ret:
        # cp_frame = frame[300:, 600:1580]    # 780x780 cropped, for no customer
        cp_frame = frame[300:, 1140:1920]    # 780x780 cropped, with customer
        yolo_detector.detect_from_cvmat(cp_frame)
        yolo_detector.extract_person_from_img(img=cp_frame,
                                              output_person_filename='outputs/people/person_{:05d}'.format(counter))
    else:   # no more frame
        break
    counter += 1

cap.release()
cv2.destroyAllWindows()