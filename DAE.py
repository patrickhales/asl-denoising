# Denoising autoencoder (DAE) model for simultaneous denoising and artefact suppression of ASL diff images

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout
from keras.layers.merge import concatenate
from keras.models import Model

def conv2d_block(input_tensor, n_filters=64, kernel_size=3, add_BatchNormalization=False,
                 dropout_rate=0, weight_regularization=0, activation=True):

    if weight_regularization != 0:
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same",
                   kernel_regularizer=regularizers.l2(weight_regularization))(input_tensor)
    else:
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding="same")(input_tensor)

    if add_BatchNormalization:
        x = BatchNormalization()(x)

    if activation:
        x = Activation("relu")(x)

    if dropout_rate != 0:
        x = Dropout(dropout_rate)(x)

    return x

def autoencoder(n_filters=64, inChannel=1):

    input_img = Input(shape=(None, None, inChannel))

    # ---------- encoder ----------------------
    conv1 = conv2d_block(input_img, n_filters=n_filters, kernel_size=3)

    p1 = MaxPooling2D((2, 2))(conv1)    # 50% original resolution

    conv2 = conv2d_block(p1, n_filters=n_filters, kernel_size=3)

    p2 = MaxPooling2D((2, 2))(conv2)    # 25% original resolution

    conv3 = conv2d_block(p2, n_filters=n_filters, kernel_size=3)

    # ------- decoder --------------------

    u1 = UpSampling2D((2, 2))(conv3)       # 50% original resolution

    merge1 = concatenate([conv2, u1])

    conv4 = conv2d_block(merge1, n_filters=n_filters, kernel_size=3)

    u2 = UpSampling2D((2, 2))(conv4)       # 100% original resolution

    merge2 = concatenate([conv1, u2])

    conv5 = conv2d_block(merge2, n_filters=1, kernel_size=3, activation=False)

    model = Model(inputs=input_img, outputs=conv5)

    return model
