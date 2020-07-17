import tensorflow as tf
from data_loader import ImgLoader
from model import MultiTaskCNN
import tensorflow.keras.backend as kb

def null_loss(y_true, y_pred):
    return kb.zeros_like(y_true)



# switch_callbacks = tf.keras.callbacks.EarlyStopping(
#         #Stop when 'val_loss' is no longer improving
#         monitor = 'val_loss',
#         # "no longer improving" being defined as "no better than 1e-2 less"
#         min_delta = 1e-2,
#         # "no longer improving" being further defined as "for at least 2 epochs"
#         patience=2,
#         verbose=1)
#
#
batch_size = 2
img_path = "/home/prak12-2/CrossScene/cropped_images"
den_path = "/home/prak12-2/CrossScene/density_maps"
checkpoints_path = '/home/prak12-2/CrossScene/checkpoints'
data_loader = ImgLoader(img_path, den_path, batch_size)

# store weights during training
ckpt_callbacks = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoints_path,
        save_weights_only = True,
        verbose = 1)



my_model = MultiTaskCNN(batch_size)
my_model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
                 loss= ['mse', 'mse'])
my_model.fit_generator( data_loader.data_generator_both(),
                        steps_per_epoch = 3,
                        epochs=1,
                        callbacks = [ckpt_callbacks])


my_model_2 = MultiTaskCNN(batch_size)
my_model_2.load_weights(checkpoints_path)
my_model_2.compile(optimizer = tf.keras.optimizers.RMSprop(1e-3),
                   loss= [null_loss, 'mse'])
my_model_2.fit_generator(data_loader.data_generator_density(),
                         steps_per_epoch = 10,
                         epochs=3,
                         callbacks =[ckpt_callbacks])
