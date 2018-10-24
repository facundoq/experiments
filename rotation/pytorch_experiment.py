import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import os
import progressbar


import torch.nn.functional as F
import copy

class ClassificationDataset:
    def __init__(self,name,x_train,x_test,y_train,y_test,num_classes,input_shape):
        self.name=name
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.num_classes=num_classes
        self.input_shape=input_shape

class ImageDataset(Dataset):


    def __init__(self, x,y,rotation=None):

        self.x=x
        self.y=y
        mu = x.mean(axis=(0, 1, 2))/255
        std = x.std(axis=(0, 1, 2))/255
        transformations=[transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(mu, std),

                         ]

        if rotation:
            transformations.insert(1,transforms.RandomRotation(180))
        else:
            pass

        self.transform=transforms.Compose(transformations)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        image=self.x[idx,:,:,:]
        image = self.transform(image)
        target=self.y[idx,:].argmax()
        return (image,target)

def get_data_generator(x,y,batch_size):
    image_dataset=ImageDataset(x,y)
    dataset=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=1)
    image_rotated_dataset = ImageDataset(x, y, rotation=180)
    rotated_dataset = DataLoader(image_rotated_dataset , batch_size=batch_size, shuffle=True, num_workers=1)

    return dataset,rotated_dataset

def print_results(dataset,loss,accuracy,correct,n):
    print('{} => Loss: {:.4f}, Accuracy: {:.2f}% ({}/{})'.format(dataset,
        loss, 100. * accuracy, correct, n),flush=True)

def train(model,epochs,optimizer,use_cuda,train_dataset,test_dataset,loss_function):
    history={"acc":[],"acc_val":[],"loss":[],"loss_val":[]}
    model.train()
    for epoch in range(1, epochs + 1):
        loss,accuracy,correct,n=train_epoch(model,epoch,optimizer,use_cuda,train_dataset,loss_function)

        #train_results = test(model, train_dataset, use_cuda)
        #print_results("Train",*train_results)

        #loss, accuracy, correct,n= test(model,train_dataset, use_cuda, loss_function)
        test_results = test(model,test_dataset,use_cuda,loss_function)
        print_results("Test", *test_results)
        history["loss"].append(loss)
        history["loss_val"].append(test_results[0])
        history["acc"].append(accuracy)
        history["acc_val"].append(test_results[1])
    return history


def test(model, dataset, use_cuda,loss_function):
    with torch.no_grad():
        model.eval()
        loss = 0
        correct = 0

        for data, target in dataset:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.float()
            # data, target = Variable(data,), Variable(target)

            output = model(data)

            loss += loss_function(output,target).item()*data.shape[0]
            #loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    n=len(dataset.dataset)
    loss /= n
    accuracy = float(correct) / float(n)
    return loss,accuracy,correct,n

def train_epoch(model,epoch,optimizer,use_cuda,train_dataset,loss_function):
    widgets = ["Epoch {}: ".format(epoch), progressbar.Percentage()
               ,progressbar.FormatLabel(' (batch %(value)d/%(max_value)d) ')
               ,' ==stats==> ', progressbar.DynamicMessage("loss")
               ,', ',progressbar.DynamicMessage("accuracy")
               ,', ',progressbar.ETA()
               ]
    progress_bar = progressbar.ProgressBar(widgets=widgets, max_value=len(train_dataset)).start()
    batches=len(train_dataset)
    losses=np.zeros(batches)
    accuracies=np.zeros(batches)
    correct=0

    for batch_idx, (data, target) in enumerate(train_dataset):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        #MODEL OUTPUT
        output = model(data)

        loss = loss_function(output, target)
        # loss = F.nll_loss(output, target)

        # UPDATE PARAMETERS
        loss.backward()
        optimizer.step()


        # ESTIMATE BATCH LOSS AND ACCURACY
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        matches = pred.eq(target.data.view_as(pred)).cpu()
        correct += matches.sum()
        accuracies[batch_idx] = matches.float().mean().item()
        losses[batch_idx] = loss.cpu().item()

        # UPDATE UI
        if batch_idx % 20 == 0:
            progress_bar.update(batch_idx+1,loss=losses[:batch_idx+1].mean(),accuracy=accuracies[:batch_idx+1].mean())

    progress_bar.finish()
    return losses.mean(),accuracies.mean(),correct,len(train_dataset.dataset)


TrainRotatedConfig = namedtuple('TrainRotatedConfig', 'dataset_name batch_size epochs pre_rotated_epochs rotated_epochs optimizer rotated_optimizer use_cuda')

def train_rotated(config,model, rotated_model, x_train, y_train, x_test, y_test,
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

def eval_scores(models,datasets,config,loss_function):
    scores = {}
    for model_name in sorted(models):
        m = models[model_name]
        for dataset_name in sorted(datasets):
            dataset = datasets[dataset_name]
            key = model_name + '_' + dataset_name
            #print(f"Evaluating {key}:")
            loss,accuracy,correct,n=test(m,dataset,config.use_cuda,loss_function)

            scores[key] = (loss,accuracy)

    return scores






def add_weight_decay(parameters, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in parameters:
        if not param.requires_grad: continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]



def freeze_layers_except(layers,layer_names,layers_to_train):
    for i in range(len(layers)):
        name=layer_names[i]
        layer=layers[i]
        requires_grad=name in layers_to_train
        #print(f"Layer {name}: setting requires_grad to {requires_grad}.")
        for param in layer.parameters():
            param.requires_grad=requires_grad


RetrainConfig = namedtuple('RetrainConfig', 'batch_size initial_epochs retrain_epochs use_cuda '
                                            'loss_function')

def retraining(model_optimizer_generator,retrained_layers_schemes,config,dataset):
    train_dataset, rotated_train_dataset = get_data_generator(dataset.x_train, dataset.y_train, config.batch_size)
    test_dataset, rotated_test_dataset = get_data_generator(dataset.x_test, dataset.y_test, config.batch_size)

    model,optimizer=model_optimizer_generator()

    print("Training vanilla network with unrotated dataset..")
    history = train(model, config.initial_epochs, optimizer, config.use_cuda, train_dataset, test_dataset,
                    config.loss_function)

    _, accuracy, _, _ = test(model, test_dataset, config.use_cuda, config.loss_function)
    _, rotated_accuracy, _, _ = test(model, rotated_test_dataset, config.use_cuda, config.loss_function)
    unrotated_accuracies=[accuracy]
    rotated_accuracies = [rotated_accuracy]



    models={"None":model}
    for retrained_layers in retrained_layers_schemes:
        retrained_model,retrained_model_optimizer=model_optimizer_generator(previous_model=model,trainable_layers=retrained_layers)

        # freeze_layers_except(retrained_model.layers(),retrained_model.layer_names(),retrained_layers)

        #for name, val in retrained_model.named_parameters():
         #   print(name, val.requires_grad)

        retrained_layers_id="_".join(retrained_layers)
        print(f"Retraining {retrained_layers} with rotated dataset:")
        history=train(retrained_model,config.retrain_epochs,retrained_model_optimizer,config.use_cuda,rotated_train_dataset,
              rotated_test_dataset,config.loss_function)
        models["retrained_"+retrained_layers_id]=retrained_model
        _, accuracy, _, _ = test(retrained_model, test_dataset, config.use_cuda, config.loss_function)
        _, rotated_accuracy, _, _ = test(retrained_model, rotated_test_dataset, config.use_cuda, config.loss_function)
        unrotated_accuracies.append(accuracy)
        rotated_accuracies.append(rotated_accuracy)


    datasets = {"test_dataset": test_dataset, "rotated_test_dataset": rotated_test_dataset,
                "train_dataset": train_dataset, "rotated_train_dataset": rotated_train_dataset}
    print("Evaluating accuracy for all models/datasets:")
    scores = eval_scores(models, datasets, config, config.loss_function)


    return scores,models,unrotated_accuracies,rotated_accuracies


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


def autolabel(rects, ax,fontsize=16):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                f"{height:0.3}",
                ha='center', va='bottom', fontsize=fontsize)


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


def retraining_accuracy_barchart(model, dataset, unrotated_accuracies,rotated_accuracies,labels,savefig=True):
    import os

    assert len(unrotated_accuracies) == len(rotated_accuracies) == len(labels)
    unrotated_accuracies=np.array(unrotated_accuracies)
    rotated_accuracies = np.array(rotated_accuracies)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(20,8),dpi=150)


    bar_width = 0.2
    index = np.arange(n) - np.arange(n)*bar_width*2.5


    opacity = 0.4
    rects1 = ax.bar(index, unrotated_accuracies, bar_width,
                    alpha=opacity, color='b',
                    label="Test unrotated")

    rects2 = ax.bar(index + bar_width, rotated_accuracies, bar_width,
                    alpha=opacity, color='r',
                    label="Test rotated")
    fontsize = 15
    ax.set_ylim(0, 1.19)
    ax.set_xlabel('Layers retrained',fontsize=fontsize+2)
    ax.set_ylabel('Test accuracy',fontsize=fontsize+2)
    ax.set_title(f'Accuracy on test sets after retraining for {model} on {dataset}.',fontsize=fontsize+4)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels,fontsize=fontsize)
    ax.legend(loc="upper center",fontsize=fontsize+2)

    autolabel(rects1, ax,fontsize=fontsize)
    autolabel(rects2, ax,fontsize=fontsize)
    fig.tight_layout()
    path=os.path.join("plots/", f"retraining_{model}_{dataset}.png")
    if savefig:
        plt.savefig(path)
    plt.show()
    return path
