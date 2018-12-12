from pytorch import training

import pytorch_models
from torch import optim

class ExperimentModel:
    def __init__(self, model, parameters, optimizer):
        self.model = model
        self.parameters = parameters
        self.optimizer = optimizer

def get_model(name,dataset,use_cuda):

    def simple_conv():
        conv_filters = {"mnist": 32, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 64, "lsa16": 32}
        fc_filters = {"mnist": 64, "mnist_rot": 64, "cifar10": 128, "fashion_mnist": 128, "lsa16": 64}
        model = pytorch_models.SimpleConv(dataset.input_shape, dataset.num_classes,
                                          conv_filters=conv_filters[dataset.name], fc_filters=fc_filters[dataset.name])
        # model= pytorch_models.FFNet(input_shape,num_classes)x
        if use_cuda:
            model = model.cuda()
        parameters = training.add_weight_decay(model.named_parameters(), 1e-9)
        optimizer = optim.Adam(parameters, lr=0.001)

        rotated_model = pytorch_models.SimpleConv(dataset.input_shape, dataset.num_classes,
                                                  conv_filters=conv_filters[dataset.name],
                                                  fc_filters=fc_filters[dataset.name])
        if use_cuda:
            rotated_model = rotated_model.cuda()
        rotated_parameters = training.add_weight_decay(rotated_model.named_parameters(), 1e-9)
        rotated_optimizer = optim.Adam(rotated_parameters, lr=0.001)

        return model, optimizer, rotated_model, rotated_optimizer

    def all_convolutional():

        filters = {"mnist": 16, "mnist_rot": 32, "cifar10": 64, "fashion_mnist": 32, "lsa16": 16}

        model = pytorch_models.AllConvolutional(dataset.input_shape, dataset.num_classes,
                                                filters=filters[dataset.name])
        # model= pytorch_models.FFNet(input_shape,num_classes)x
        if use_cuda:
            model = model.cuda()
        parameters = training.add_weight_decay(model.named_parameters(), 1e-13)
        optimizer = optim.Adam(parameters, lr=0.0001)
        print(model)

        rotated_model = pytorch_models.AllConvolutional(dataset.input_shape, dataset.num_classes,
                                                        filters=filters[dataset.name])

        if use_cuda:
            rotated_model = rotated_model.cuda()
        rotated_parameters = training.add_weight_decay(rotated_model.named_parameters(), 1e-13)
        rotated_optimizer = optim.Adam(rotated_parameters, lr=0.0001)

        return model, optimizer, rotated_model, rotated_optimizer

    models={"SimpleConv":simple_conv,"AllConv":all_convolutional}

    return models[name]()