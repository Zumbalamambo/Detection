__author__ = 'bsl'

from keras.callbacks import Callback
import keras.backend as K

class LearningRateAnnealing(Callback):
    '''
        nb_epoch = no. of epochs when decay should happen.
        decay = decay value
        check for more details: https://github.com/fchollet/keras/issues/898
    '''
    def __init__(self, nb_epoch, annealing_factor):
        super(LearningRateAnnealing, self).__init__()
        self.nb_epoch = nb_epoch
        self.annealing_factor = annealing_factor

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.get_value()
        if epoch > 1 and (epoch % self.nb_epoch == 0):
            new_lr = self.annealing_factor * old_lr
            print 'lr={}'.format(new_lr)    # show how lr evolves
            K.set_value(self.model.optimizer.lr, new_lr)
        else:
            K.set_value(self.model.optimizer.lr, old_lr)

if __name__ == '__main__':
    # example for usage
    annealingSchedule = LearningRateAnnealing(5, 0.5)
    # or
    annealingSchedule = LearningRateAnnealing(20, 0.1)
