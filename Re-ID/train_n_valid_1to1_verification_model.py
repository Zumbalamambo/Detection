from build_1to1_verification_model import build_compiled_1to1_verification_model

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from utils.lr_annealing import LearningRateAnnealing

import numpy as np

if __name__ == '__main__':
    # parameters
    img_height = 224
    img_width = 224
    batch_size = 32
    nb_node = 256
    nb_layer = 1
    is_bn = True
    is_do = True
    l2_regular = 1e+0
    loss_func = 'binary_crossentropy'
    learning_rate = 1e-3
    momentum = 0.9
    optimizer = SGD(lr=learning_rate, momentum=momentum)
    metrics = ['accuracy']  # loss is the default metric

    nb_ep = 36      # ~14 hours
    nb_anneal_ep = 12   # 3 changes for 36, 1e-3 -4 and -5
    annealing_factor = 0.1

    nb_train_sample = 112831 + 42235    # salesperson + customer
    nb_valid_sample = 33872 + 8084


    train_img_gen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       height_shift_range=0.2,
                                       width_shift_range=0.2,
                                       horizontal_flip=True)
    valid_img_gen = ImageDataGenerator(rescale=1./255)

    train_generator = train_img_gen.flow_from_directory('../datasets/Detected/person_train',
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='binary')
    valid_generator = valid_img_gen.flow_from_directory('../datasets/Detected/person_valid',
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

    # build model
    model = build_compiled_1to1_verification_model(nb_node_hidden_layer = nb_node,
                                                   nb_hidden_layer = nb_layer,
                                                   is_bn_applied = is_bn,
                                                   is_do_applied = is_do,
                                                   l2_regularizer = l2_regular,
                                                   loss_function = loss_func,
                                                   optimizer = optimizer,
                                                   metric_list = metrics)
    anneal_schedule = LearningRateAnnealing(nb_anneal_ep, annealing_factor)

    # train
    history = model.fit_generator(train_generator,
                                  samples_per_epoch=nb_train_sample,
                                  nb_epoch=nb_ep,
                                  validation_data=valid_generator,
                                  nb_val_samples=nb_valid_sample,
                                  callbacks=[anneal_schedule])

    # save
    record = np.column_stack((np.array(history.epoch)+1,
                              history.history['loss'],
                              history.history['val_loss'],
                              history.history['accuracy'],
                              history.history['val_accuracy']))
    np.savetxt('saver/convergence.csv', record, delimiter=',')
    model.save_weights('saver/weights.h5', overwrite=False)