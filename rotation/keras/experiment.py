from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LambdaCallback
from keras import backend as K
import numpy as np



def get_callbacks(model, no_improv_epochs=20, min_delta=1e-10):
    # Early stopping - End training early if we don't improve on the loss function
    # by a certain minimum threshold
    es = EarlyStopping('val_loss', patience=no_improv_epochs,
                       mode='min', min_delta=min_delta)
    weight_history = []

    def print_gconv(self, batch):
        layers=model.layers[0]
        weights=layers.get_weights()[0]
        weights = np.array(weights)
        weight_history.append(weights)
        if len(weight_history) > 1:
            old_weights = weight_history.pop(0)
            diff = np.abs(old_weights - weights).sum()
            print(" Weight size %s, Weight variation: %.15f" % (str(weights.shape),diff))

    print_gconv_callback = LambdaCallback(on_batch_begin=print_gconv)

    return []

def get_channel_index():
    return -1 if K.image_data_format()=='channels_last' else 1
def base_data_generator():
    # assumes channel is dimension 3
    channel_index = get_channel_index()
    def image_channel_standardize(image):
        indices=range(len(image.shape))
        spatial_indices=tuple(filter(lambda i: i != channel_index,indices))
        mean=image.mean(axis=spatial_indices,keepdims=True)
        std=image.std(axis=spatial_indices,keepdims=True)
        return (image-mean)/(std+K.epsilon())


    return ImageDataGenerator(preprocessing_function=image_channel_standardize)#samplewise_center=True,
    # samplewise_std_normalization=True)
                              #featurewise_center=True,featurewise_std_normalization=True)
def get_data_generators(x):
    data_generator = base_data_generator()
    rotated_data_generator = base_data_generator()
    rotated_data_generator.rotation_range=180

    data_generator.fit(x)
    rotated_data_generator.fit(x)
    return data_generator,rotated_data_generator

def train_rotated(model, rotated_model, x_train, y_train, x_test, y_test, classes, input_shape, batch_size, epochs,
                  rotated_epochs, plot_accuracy=False):

    data_generator, rotated_data_generator=get_data_generators(x_train)
    train_dataset = data_generator.flow(x_train, y_train, shuffle=True, batch_size=batch_size)
    test_dataset = data_generator.flow(x_test, y_test, batch_size=batch_size)
    model_callback_list = get_callbacks(model, no_improv_epochs=max(epochs // 4, 3))
    if epochs == 0:
        print("Not training model with unrotated dataset")
    else:
        print("Training model with unrotated dataset...")
        history = model.fit_generator(train_dataset,
                                      steps_per_epoch=len(x_train) / batch_size,
                                      epochs=epochs,
                                      verbose=1,
                                      validation_data=test_dataset,
                                      callbacks=model_callback_list)

        if plot_accuracy:
            plot_history(history)
    rotated_model_callback_list = get_callbacks(rotated_model, no_improv_epochs=max(epochs // 4, 3))

    rotated_train_dataset = rotated_data_generator.flow(x_train, y_train, shuffle=True, batch_size=batch_size)
    rotated_test_dataset = rotated_data_generator.flow(x_test, y_test, batch_size=batch_size)
    if rotated_epochs == 0:
        print("Not training model with rotated dataset")
    else:
        print("Training rotated model with rotated dataset...")
        rotated_history = rotated_model.fit_generator(rotated_train_dataset,
                                                      steps_per_epoch=len(x_train) / batch_size,
                                                      epochs=rotated_epochs,
                                                      verbose=1, callbacks=rotated_model_callback_list,
                                                      validation_data=rotated_test_dataset)
        if plot_accuracy:
            plot_history(rotated_history)
    print("Testing both models on both datasets...")
    scores = {}
    models = {"rotated_model": rotated_model, "model": model}
    datasets = {"test_dataset": test_dataset, "rotated_test_dataset": rotated_test_dataset,
                "train_dataset": train_dataset, "rotated_train_dataset": rotated_train_dataset}
    for model_name in sorted(models):
        m = models[model_name]
        for dataset_name in sorted(datasets):
            dataset = datasets[dataset_name]
            scores[model_name + '_' + dataset_name] = m.evaluate_generator(dataset)

    return scores


def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


import matplotlib.pyplot as plt
import math

import warnings
def visualize(images, labels, n_show, rotated=True):
    n, h, w, c = images.shape
    normal_data_generator,rotated_data_generator=get_data_generators(images)
    if rotated:
        data_generator = rotated_data_generator
    else:
        data_generator = normal_data_generator
    batch = data_generator.flow(images, labels, batch_size=n_show,shuffle=False)
    images_batch, labels_batch = batch[0]

    plots = math.ceil(math.sqrt(n_show))
    rows,columns=plots,plots
    rows=1
    columns=n_show
    f, axarr = plt.subplots(rows,columns, figsize=(50, 50), sharex=True)
    print(axarr.shape)
    for i in range(n_show):
        j, k = i // rows, i % columns
        image = images_batch[i, :, :, :]
        image=image-image.min()
        image=image/(image.max()+K.epsilon())
        if (c == 1):
            image = image.squeeze(axis=-1)
            axarr[ k].imshow(image,vmin=0,vmax=1)
        else:
            with warnings.catch_warnings():
                axarr[ k].imshow(image)
        #axarr[j, k].set_title("class %d" % labels_batch[i, :].argmax())
        axarr[ k].axis("off")
    plt.show()
