import matplotlib.pyplot as plt
from collections import namedtuple

from pytorch.training import test,train,eval_scores
import numpy as np
import os

from pytorch.dataset import get_data_generator
import torch


from utils import autolabel


TrainRotatedConfig = namedtuple('TrainRotatedConfig', 'dataset_name batch_size epochs pre_rotated_epochs rotated_epochs optimizer rotated_optimizer use_cuda')

def run(config,model, rotated_model, x_train, y_train, x_test, y_test,
                  plot_accuracy=False,loss_function=torch.nn.NLLLoss()):

    os.makedirs(experiment_plot_path(model.name, config.dataset_name),exist_ok=True)
    train_dataset,rotated_train_dataset=get_data_generator(x_train,y_train, config.batch_size)
    test_dataset, rotated_test_dataset = get_data_generator(x_test, y_test, config.batch_size)

    # UNROTATED DATASET
    if config.epochs == 0:
        print("Skipping training model with unrotated dataset")
        history={}
    else:
        print("Training model with unrotated dataset...",flush=True)
        history = train(model,config.epochs,config.optimizer,config.use_cuda,train_dataset,test_dataset,loss_function)
        if plot_accuracy:
            accuracy_plot_path =plot_history(history,"unrotated",model.name,config.dataset_name)

    # ROTATED MODEL, UNROTATED DATASET
    if config.pre_rotated_epochs == 0:
        print("Skipping pretraining rotated model with unrotated dataset")
    else:
        print("Pretraining rotated model with unrotated dataset...",flush=True)
        pre_rotated_history = train(rotated_model, config.rotated_epochs, config.rotated_optimizer, config.use_cuda,
                                train_dataset,test_dataset,loss_function)
        if plot_accuracy:
            plot_history(pre_rotated_history,"pre_rotated",model.name,config.dataset_name)


        # ROTATED DATASET
    if config.rotated_epochs == 0:
        print("Skipping pre-training of rotated model with unrotated dataset")
    else:
        print("Training rotated model with rotated dataset...",flush=True)
        rotated_history = train(rotated_model, config.rotated_epochs, config.rotated_optimizer, config.use_cuda,
                                rotated_train_dataset,rotated_test_dataset,loss_function)
        if plot_accuracy:
            rotated_accuracy_plot_path=plot_history(rotated_history,"rotated",rotated_model.name,config.dataset_name)

    print("Testing both models on both datasets...",flush=True)

    models = {"rotated_model": rotated_model, "model": model}
    datasets = {"test_dataset": test_dataset, "rotated_test_dataset": rotated_test_dataset,
                 "train_dataset": train_dataset, "rotated_train_dataset": rotated_train_dataset}
    scores=eval_scores(models,datasets,config,loss_function)
    train_test_path=train_test_accuracy_barchart2(scores,model,rotated_model,config)
    experiment_plot = os.path.join("plots",f"{model.name}_{config.dataset_name}_train_rotated.png")

    os.system(f"convert {accuracy_plot_path} {rotated_accuracy_plot_path} {train_test_path} +append {experiment_plot}")
    return scores







def write_scores(scores,output_file,general_message,config=None):
    with open(output_file, "a+") as f:
        f.write(general_message)
        print(general_message)
        for k, v in scores.items():
            message = '%s score: loss=%f, accuracy=%f\n' % (k, v[0], v[1])
            print(message)
            f.write(message)
        if config:
            config_message="Config: "+str(config)
            print(config_message)
            f.write(config_message)
        f.write("\n\n")


def train_test_accuracy_barchart2(scores,model,rotated_model,config):
    test_dataset_scores = [scores["model_test_dataset"][1], scores["rotated_model_test_dataset"][1]]
    rotated_test_dataset_scores = [scores["model_rotated_test_dataset"][1], scores["rotated_model_rotated_test_dataset"][1]]
    accuracies = np.array([test_dataset_scores, rotated_test_dataset_scores])
    return train_test_accuracy_barchart(model.name, config.dataset_name, accuracies)

def experiment_plot_path(model,dataset):
    return f"plots/{model}/{dataset}"

def train_test_accuracy_barchart(model, dataset, accuracies,savefig=True):
    import os
    # Accuracies:    |   Train unrotated   |   Train rotated
    # Test unrotated |                     |
    # Test rotated   |                     |
    #
    assert (accuracies.shape == (2, 2))

    fig, ax = plt.subplots()

    index = np.arange(2)
    bar_width = 0.3

    opacity = 0.4

    rects1 = ax.bar(index, accuracies[0, :], bar_width,
                    alpha=opacity, color='b',
                    label="Test unrotated")

    rects2 = ax.bar(index + bar_width, accuracies[1, :], bar_width,
                    alpha=opacity, color='r',
                    label="Test rotated")

    ax.set_ylim(0, 1.19)
    # ax.set_xlabel('Training scheme')
    ax.set_ylabel('Test accuracy')
    ax.set_title(f'Final accuracy on test sets.')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(("Train unrotated", "Train rotated"))
    ax.legend(loc="upper center")
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    fig.tight_layout()
    path=os.path.join(experiment_plot_path(model,dataset), f"train_test.png")
    if savefig:
        plt.savefig(path)
    plt.show()
    return path


def plot_history(history,name,model_name,dataset_name):
    from time import gmtime, strftime
    t=strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    import os
    f, (a1,a2) = plt.subplots(1,2)
    path= experiment_plot_path(model_name, dataset_name)
    path=os.path.join(path,f"{name}.png")
    # accuracy
    a1.plot(history['acc'])
    a1.plot(history['acc_val'])
    #a1.set_title('Accuracy')
    a1.set_ylabel('accuracy')
    a1.set_xlabel('epoch')
    a1.set_ylim(0,1.1)
    a1.legend(['train', 'test'], loc='lower right')
    # loss
    a2.plot(history['loss'])
    a2.plot(history['loss_val'])
    #a2.set_title('Loss')
    a2.set_ylabel('loss')
    a2.set_xlabel('epoch')
    a2.legend(['train', 'test'], loc='upper right')
    f.suptitle(f"{model_name} trained with {name} {dataset_name}")
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(path)
    plt.show()
    return path


