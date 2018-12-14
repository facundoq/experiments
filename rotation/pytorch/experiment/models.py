from pytorch import training

import pytorch_models
from torch import optim

class ExperimentModel:
    def __init__(self, model, parameters, optimizer):
        self.model = model
        self.parameters = parameters
        self.optimizer = optimizer

def get_model(name,dataset,use_cuda):
    def setup_model(model,lr,wd):
        if use_cuda:
            model = model.cuda()
        parameters = training.add_weight_decay(model.named_parameters(), wd)
        optimizer = optim.Adam(parameters, lr=lr)
        return optimizer

    def simple_conv():
        conv_filters = {"mnist": 32, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 64, "lsa16": 32}
        fc_filters = {"mnist": 64, "mnist_rot": 64, "cifar10": 128, "fashion_mnist": 128, "lsa16": 64}
        model = pytorch_models.SimpleConv(dataset.input_shape, dataset.num_classes,
                                          conv_filters=conv_filters[dataset.name], fc_filters=fc_filters[dataset.name])
        optimizer=setup_model(model,0.001,1e-9)
        rotated_model = pytorch_models.SimpleConv(dataset.input_shape, dataset.num_classes,
                                                  conv_filters=conv_filters[dataset.name],
                                                  fc_filters=fc_filters[dataset.name])
        rotated_optimizer = setup_model(rotated_model, 0.001, 1e-9)
        return model, optimizer, rotated_model, rotated_optimizer

    def all_convolutional():
        filters = {"mnist": 16, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 32, "lsa16": 16}
        model = pytorch_models.AllConvolutional(dataset.input_shape, dataset.num_classes,
                                                filters=filters[dataset.name])
        optimizer=setup_model(model,0.00001,1e-13)
        rotated_model = pytorch_models.AllConvolutional(dataset.input_shape, dataset.num_classes,
                                                        filters=filters[dataset.name])
        rotated_optimizer = setup_model(rotated_model, 0.00001, 1e-13)

        return model, optimizer, rotated_model, rotated_optimizer
    def all_conv():
        filters = {"mnist": 16, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 32, "lsa16": 16}
        model = pytorch_models.AllConv(dataset.input_shape, dataset.num_classes,
                                                filters=filters[dataset.name])
        optimizer=setup_model(model,0.00001,1e-13)
        rotated_model = pytorch_models.AllConvolutional(dataset.input_shape, dataset.num_classes,
                                                        filters=filters[dataset.name])
        rotated_optimizer = setup_model(rotated_model, 0.00001, 1e-13)

        return model, optimizer, rotated_model, rotated_optimizer


    models={ pytorch_models.SimpleConv.__name__:simple_conv,
             pytorch_models.AllConvolutional.__name__:all_convolutional,
             pytorch_models.AllConv.__name__: all_conv,
            }
    return models[name]()

def get_epochs(dataset,model):
    if model==pytorch_models.SimpleConv.__name__:
        epochs={'cifar10':70,'mnist':5,'fashion_mnist':12,'cluttered_mnist':10,'lsa16':50,'mnist_rot':5,'pugeault':15}
        rotated_epochs={'cifar10':120,'mnist':15,'fashion_mnist':60,'cluttered_mnist':30,'lsa16':100,'mnist_rot':5,'pugeault':40}
    elif model==pytorch_models.AllConvolutional.__name__:
        epochs={'cifar10':70,'mnist':15,'fashion_mnist':12,'cluttered_mnist':10,'lsa16':50,'mnist_rot':5,'pugeault':15}
        rotated_epochs={'cifar10':150,'mnist':50,'fashion_mnist':60,'cluttered_mnist':30,'lsa16':100,'mnist_rot':5,
                        'pugeault':40}
    elif model==pytorch_models.AllConv.__name__:
        epochs={'cifar10':70,'mnist':15,'fashion_mnist':12,'cluttered_mnist':10,'lsa16':50,'mnist_rot':5,'pugeault':15}
        rotated_epochs={'cifar10':150,'mnist':50,'fashion_mnist':60,'cluttered_mnist':30,'lsa16':100,'mnist_rot':5,
                        'pugeault':40}
    else:
        raise ValueError(f"Invalid model name: {model}")
    return epochs[dataset],rotated_epochs[dataset]
