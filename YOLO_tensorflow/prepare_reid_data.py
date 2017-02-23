import YOLO_small_tf as yolo_tf
import cv2
import os


if __name__ == '__main__':
    date = '20170220'
    hour_train = [12,13,19,20]
    hour_valid = [18]

    # prepare YOLO detector
    yolo_detector = yolo_tf.YOLO_TF()
    yolo_detector.disp_console = True
    yolo_detector.imshow = False
    yolo_detector.tofile_img = 'outputs/people/person.jpg'
    yolo_detector.tofile_txt = 'outputs/people/person.txt'
    yolo_detector.filewrite_img = False
    yolo_detector.filewrite_txt = False

    # prepare training images first
    for hour in hour_train:
        for min_file in sorted(os.listdir('../datasets/TongYing/{}/{}/'.format(date, hour))):
            # confirm mp4 file
            min_idx, ext = os.path.splitext(min_file)   # min + idx to avoid confusion with min()
            if ext == '.mp4':
                cap = cv2.VideoCapture('../datasets/TongYing/{}/{}/{}'.format(date, hour, min_file))
                frame_idx = 0
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        # detect salesperson first
                        cp_frame = frame[300:, 600:1580]    # 780x780 cropped
                        yolo_detector.detect_from_cvmat(cp_frame)
                        output_filename = '../datasets/Detected/person_train/0_positive/ty_{}_{}_{}_{:05d}'\
                            .format(date, hour, min_idx, frame_idx)   # extra file ext for duplicate person and .jpg
                        yolo_detector.extract_person_from_img(img=cp_frame,
                                                              output_person_filename=output_filename,
                                                              nb_person_w_confid=1,
                                                              person_min_dimen=120)
                        # detect customer
                        cp_frame = frame[300:, 1140:1920]  # 780x780 cropped
                        yolo_detector.detect_from_cvmat(cp_frame)
                        output_filename = '../datasets/Detected/person_train/1_negative/ty_{}_{}_{}_{:05d}'\
                            .format(date, hour, min_idx, frame_idx)  # extra file ext for duplicate person and .jpg
                        yolo_detector.extract_person_from_img(img=cp_frame,
                                                              output_person_filename=output_filename,
                                                              nb_person_w_confid=1,
                                                              person_min_dimen=120)

                    else:  # no more frame, go for next min_idx
                        break
                    frame_idx += 1

    # prepare validation images
    for hour in hour_valid:
        for min_file in sorted(os.listdir('../datasets/TongYing/{}/{}/'.format(date, hour))):
            # confirm mp4 file
            min_idx, ext = os.path.splitext(min_file)
            if ext == '.mp4':
                cap = cv2.VideoCapture('../datasets/TongYing/{}/{}/{}'.format(date, hour, min_file))
                frame_idx = 0
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        # detect salesperson first
                        cp_frame = frame[300:, 600:1580]  # 780x780 cropped
                        yolo_detector.detect_from_cvmat(cp_frame)
                        output_filename = '../datasets/Detected/person_valid/0_positive/ty_{}_{}_{}_{:05d}' \
                            .format(date, hour, min_idx, frame_idx)  # extra file ext for duplicate person and .jpg
                        yolo_detector.extract_person_from_img(img=cp_frame,
                                                              output_person_filename=output_filename,
                                                              nb_person_w_confid=0,
                                                              person_min_dimen=120)
                        # detect customer
                        cp_frame = frame[300:, 1140:1920]  # 780x780 cropped
                        yolo_detector.detect_from_cvmat(cp_frame)
                        output_filename = '../datasets/Detected/person_valid/1_negative/ty_{}_{}_{}_{:05d}' \
                            .format(date, hour, min_idx, frame_idx)  # extra file ext for duplicate person and .jpg
                        yolo_detector.extract_person_from_img(img=cp_frame,
                                                              output_person_filename=output_filename,
                                                              nb_person_w_confid=0,
                                                              person_min_dimen=120)

                    else:  # no more frame, go for next min_idx
                        break
                    frame_idx += 1

    cap.release()
