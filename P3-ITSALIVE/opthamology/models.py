import torch
import torch.nn as nn
import datetime

class VoltronEncodingModel(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.log_dataset_info(dataset)
        self.train_start_time = datetime.datetime.now()
 
        self.layers = nn.ModuleDict()
 
    def log_dataset_info(self, dataset):
        example_x, example_y = dataset.get_example()
        self.x_shape = example_x.shape
        self.y_shape = example_y.shape
        self.num_cells = self.y_shape[0]
        self.window = self.x_shape[0]
        self.height= self.x_shape[1]
        self.width = self.x_shape[2]
        self.h5_filepath = dataset.h5_filepath
        self.series = dataset.series
        self.clusters = dataset.clusters
        self.num_before = dataset.num_before
        self.num_after = dataset.num_after
        self.stimulus_key = dataset.stimulus_key
        self.data_streams = dataset.data_streams
        self.z_stats = dataset.z_stats
        self.start_idx = dataset.start_idx
        self.end_idx = dataset.end_idx

class CNN(VoltronEncodingModel):
    def __init__(self, num_layers, kernel_sizes, num_channels, padding, batchnorm, **kwargs):
        super().__init__(**kwargs)

        self.padding = padding
        self.batchnorm = batchnorm

        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * num_layers
        if type(num_channels) == int:
            num_channels = [num_channels] * num_layers

        for i in range(num_layers):
            if i == 0:
                self.layers['conv'+str(i)] = nn.Conv2d(self.window, num_channels[i], kernel_size=kernel_sizes[i], stride=1, padding=padding)
            else:
                self.layers['conv'+str(i)] = nn.Conv2d(num_channels[i-1], num_channels[i], kernel_size=kernel_sizes[i], stride=1, padding=padding)

            self.layers['dropout'+str(i)] = nn.Dropout2d(p=0.2)

            if batchnorm:
                self.layers['bn'+str(i)] = nn.BatchNorm2d(num_channels[i], track_running_stats=False)

            self.layers['nl'+str(i)] = nn.Softplus()

        self.layers['flatten'] = nn.Flatten()

        temp = torch.randn(1, self.window, self.height, self.width)
        for name, layer in self.layers.items():
            temp = layer(temp)

        self.layers['fc'] = nn.Linear(temp.shape[1], self.num_cells, bias=True)
        self.layers['output'] = nn.Softplus()

    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x

class DatasetInfo:
    """
    Class to store information about a dataset
    that is used to construct and train
    subsequent models.
    """

    def __init__(self, dataset):
        ex = dataset.get_example()
        self.x_shape = ex[0].shape
        self.y_shape = ex[1].shape
        self.num_cells = self.y_shape[0]
        self.window = self.x_shape[0]
        self.height = self.x_shape[1]
        self.width = self.x_shape[2]
        self.h5_filepath = dataset.h5_filepath
        self.series = dataset.series
        self.num_before = dataset.num_before
        self.num_after = dataset.num_after
        self.start_idx = dataset.start_idx
        self.end_idx = dataset.end_idx
        self.which_clusters = dataset.cluster_ids


class BoptEncodingModel(nn.Module):
    """
    Base class for encoding models. Contains
    a DatasetInfo object that stores information
    about the dataset used to construct and reconstruct the model.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = DatasetInfo(dataset)
        self.dataset.h5_filepath = None
        self.train_start_time = datetime.datetime.now()

        self.layers = nn.ModuleDict()

class CNNComponent(BoptEncodingModel):
    """The CNN component of the model that processes visual input."""
    
    def __init__(self, num_layers, kernel_sizes, num_channels, padding, input_layernorm, **kwargs):
        super().__init__(**kwargs)
        
        self.padding = padding
        self.input_layernorm = input_layernorm
        
        og_temp = torch.randn(1, self.dataset.window, self.dataset.height, self.dataset.width)
        
        if type(kernel_sizes) == int:
            kernel_sizes = [kernel_sizes] * num_layers
        if type(num_channels) == int:
            num_channels = [num_channels] * num_layers
        
        if input_layernorm:
            self.layers['input_layernorm'] = nn.LayerNorm(
                [self.dataset.window, self.dataset.height, self.dataset.width],
                elementwise_affine=False)
        
        for i in range(num_layers):
            if i == 0:
                self.layers['conv' + str(i)] = nn.Conv2d(
                    self.dataset.window,
                    num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=padding)
            else:
                self.layers['conv' + str(i)] = nn.Conv2d(
                    num_channels[i - 1],
                    num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=1,
                    padding=padding)
            
            temp = og_temp.clone()
            for name, layer in self.layers.items():
                temp = layer(temp)
            
            self.layers['layernorm' + str(i)] = nn.LayerNorm(
                [temp.shape[1], temp.shape[2], temp.shape[3]],
                elementwise_affine=False)
            self.layers['dropout' + str(i)] = nn.Dropout(0.1)
            if i == num_layers - 1:
                pass
            else:
                self.layers['nl' + str(i)] = nn.Softplus()
        
        self.layers['flatten'] = nn.Flatten()
        
        temp = og_temp.clone()
        for name, layer in self.layers.items():
            temp = layer(temp)
        
        self.layers['fc'] = nn.Linear(temp.shape[1], self.dataset.num_cells, bias=True)
        self.layers['output'] = nn.Softplus()
    
    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x
