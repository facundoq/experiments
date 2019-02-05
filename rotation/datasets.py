from keras.datasets import mnist,fashion_mnist,cifar10
from data import lsa16,cluttered_mnist,pugeault,mnist_rot

from keras import backend as K
import keras
import numpy as np

names=["mnist","fashion_mnist","cifar10","mnist_rot","cluttered_mnist","lsa16","pugeault"]

def get_data(dataset="mnist",dataformat="NHWC"):
    # the data, shuffled and split between train and test sets
    if dataset=="mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train,x_test=np.expand_dims(x_train,axis=3),np.expand_dims(x_test, axis=3)
        img_channels,img_rows, img_cols = 1,28, 28
        labels=["0","1","2","3","4","5","6","7","8","9"]
        num_classes = 10
    elif dataset=="fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
        img_channels,img_rows, img_cols = 1,28, 28
        labels=["tshirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle_boot"]
        num_classes = 10
    elif dataset=="cifar10":
        img_channels,img_rows, img_cols = 3,32,32
        labels=['dog', 'horse', 'frog', 'airplane', 'cat', 'ship', 'deer', 'bird', 'truck', 'automobile']
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    elif dataset == "mnist_rot":
        x_train, x_test,y_train, y_test, img_channels, img_rows, img_cols = mnist_rot.load_data()
        labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        num_classes = 10
    elif dataset=="cluttered_mnist":
        (x_train, y_train), (x_test, y_test), (x_val, y_val),img_channels,img_rows, img_cols = cluttered_mnist.load_data()
        x_train, x_test = np.expand_dims(x_train, axis=3), np.expand_dims(x_test, axis=3)
        x_val= np.expand_dims(x_val, axis=3)
        labels=["0","1","2","3","4","5","6","7","8","9"]
        num_classes = 10
    elif dataset== "lsa16":
        labels = ["five","four","horns","curve","fingers together","double","hook","index","l","flat hand","mitten",
                  "beak","thumb","fist","telephone","V"]
        x_train, x_test, y_train, y_test,img_channels,img_rows, img_cols \
            = lsa16.load_data(version="lsa32x32_nr_rgb_black_background",test_subjects=[9])
        num_classes = 16
    elif dataset == "pugeault":
        x_train, x_test, y_train, y_test, img_channels, img_rows, img_cols = pugeault.load_data()
        num_classes = 25
        import string
        labels=string.ascii_lowercase[:25]
    else:
        raise ValueError("Unknown dataset: %s" % dataset)

    input_shape = (img_rows, img_cols, img_channels)
    if dataformat == 'NCHW':
        x_train,x_test=x_train.transpose([0,3,1,2]),x_test.transpose([0,3,1,2])
    elif dataformat == "NHWC":
        pass #already in this format
    else:
        raise ValueError("Invalid channel format %s" % dataformat)

    
    # x_train /= x_train.max()
    # x_train -= x_train.mean(axis= (1,2), keepdims= 1)
    # x_test /= x_test.max()
    # x_test -= x_test.mean(axis= (1,2), keepdims= 1)

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test  = to_categorical(y_test, num_classes)

    # x_train=x_train.astype("float32")
    # x_test = x_test.astype("float32")
    # y_train = y_train.astype("float32")
    # y_test = y_test.astype("float32")

    return (x_train, y_train), (x_test, y_test), input_shape,num_classes,labels



def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical