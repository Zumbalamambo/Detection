from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.regularizers import l2
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.optimizers import SGD


def build_1to1_verification_model(nb_node_hidden_layer = 256,
                                  nb_hidden_layer = 1,
                                  is_bn_applied = True,
                                  is_do_applied = True,
                                  l2_regularizer = 1e-1):
    # default image size
    img_height = 224
    img_width = 224

    img_size = (img_height, img_width, 3)   # Tensorflow dim ordering
    input_tensor = Input(batch_shape=(None, ) + img_size)

    print 'build vgg16 with imagenet weights'
    vgg_notop = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
    # make all layers un-trainable
    for layer in vgg_notop.layers:
        layer.trainable = False
    model_output = vgg_notop.output
    model_output = Flatten()(model_output)
    for i in range(nb_hidden_layer):
        model_output = Dense(nb_node_hidden_layer,
                             name='fc_dense_{}'.format(i+1),
                             W_regularizer=l2(l2_regularizer),
                             b_regularizer=None,
                             activation='linear')(model_output)
        if is_bn_applied:
            model_output = BatchNormalization(name='fc_bn_{}'.format(i+1))(model_output)
        model_output = Activation('relu', name='fc_act_{}'.format(i+1))(model_output)
        if is_do_applied:
            model_output = Dropout(0.5, name='fc_do_{}'.format(i+1))(model_output)  # default drop half
    # final output
    model_output = Dense(1, name='fc_dense_out',
                         W_regularizer=l2(l2_regularizer),
                         b_regularizer=None,
                         activation='sigmoid')(model_output)
    inputs = get_source_inputs(input_tensor)
    model = Model(inputs, model_output, name='vgg16_imagenet_binary')

    return model

def build_compiled_1to1_verification_model(nb_node_hidden_layer = 256,
                                           nb_hidden_layer = 1,
                                           is_bn_applied = True,
                                           is_do_applied = True,
                                           l2_regularizer = 1e-1,
                                           loss_function = 'binary_crossentropy',
                                           optimizer = 'adadelta',
                                           metric_list = ['accuracy']):
    model = build_1to1_verification_model(nb_node_hidden_layer = nb_node_hidden_layer,
                                          nb_hidden_layer = nb_hidden_layer,
                                          is_bn_applied = is_bn_applied,
                                          is_do_applied = is_do_applied,
                                          l2_regularizer = l2_regularizer)
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=metric_list)
    return model

if __name__ == '__main__':
    nb_node = 256
    nb_layer = 1
    is_bn = True
    is_do = True
    l2_regular = 1e+0
    loss_func = 'binary_crossentropy'
    learning_rate = 1e-4
    momentum = 0.9
    optimizer = SGD(lr=learning_rate, momentum=momentum)
    metrics = ['accuracy']   # loss is the default metric
    model = build_compiled_1to1_verification_model(nb_node_hidden_layer = nb_node,
                                                   nb_hidden_layer = nb_layer,
                                                   is_bn_applied = is_bn,
                                                   is_do_applied = is_do,
                                                   l2_regularizer = l2_regular,
                                                   loss_function = loss_func,
                                                   optimizer = optimizer,
                                                   metric_list = metrics)
    model.summary()
    print model.layers.__len__()
    print model.layers[15].trainable
    print model.layers[28].trainable
    print model.layers[28].get_config()