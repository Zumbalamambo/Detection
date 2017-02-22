# import YOLO_small_tf as yolo_tf
import YOLO_tiny_tf as yolo_tf

yolo_detector = yolo_tf.YOLO_TF()
yolo_detector.disp_console = True
yolo_detector.imshow = False
yolo_detector.tofile_img = 'outputs/people/person.jpg'
yolo_detector.tofile_txt = 'outputs/people/person.txt'
yolo_detector.filewrite_img = False
yolo_detector.filewrite_txt = False

yolo_detector.detect_from_file('test/person.jpg')
yolo_detector.extract_person_from_file(filename='test/person.jpg',
                                       output_person_filename='outputs/people/person_{:05d}.jpg'.format(2))