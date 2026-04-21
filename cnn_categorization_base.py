from torch import nn


def cnn_categorization_base(netspec_opts):
    """
    Constructs a network for the base categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the network's architecture. It has the keys
                 'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                 Each key holds a list containing the values for the
                corresponding parameter for each layer.
    Returns
    ------
     net: (nn.Sequential), the base categorization model
    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()

    # add layers as specified in netspec_opts to the network
    kernel_sizes = netspec_opts['kernel_size']
    num_filters = netspec_opts['num_filters']
    strides = netspec_opts['stride']
    layer_types = netspec_opts['layer_type']
    
    in_channels = 3
    conv_count = 1
    bn_count = 1
    relu_count = 1
    pool_count = 1
    
    for i in range(len(layer_types)):
        layer_type = layer_types[i]
        k_size = kernel_sizes[i]
        n_filt = num_filters[i]
        stride = strides[i]
        
        if layer_type == 'conv':
            if isinstance(k_size, int):
                padding = (k_size - 1) // 2 if k_size > 1 else 0
            else:
                padding = ((k_size[0] - 1) // 2, (k_size[1] - 1) // 2)
            net.add_module(f'conv_{conv_count}', nn.Conv2d(in_channels, n_filt, kernel_size=k_size, stride=stride, padding=padding))
            in_channels = n_filt
            conv_count += 1
            
        elif layer_type == 'bn':
            net.add_module(f'bn_{bn_count}', nn.BatchNorm2d(n_filt))
            bn_count += 1
            
        elif layer_type == 'relu':
            net.add_module(f'relu_{relu_count}', nn.ReLU())
            relu_count += 1
            
        elif layer_type == 'pool':
            if isinstance(k_size, int):
                padding = (k_size - 1) // 2 if k_size > 1 else 0
            else:
                padding = ((k_size[0] - 1) // 2, (k_size[1] - 1) // 2)
            net.add_module(f'pool_{pool_count}', nn.AvgPool2d(kernel_size=k_size, stride=stride, padding=padding))
            pool_count += 1
            
        elif layer_type == 'pred':
            # Add adaptive pooling and flatten before the linear layer
            net.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
            net.add_module('flatten', nn.Flatten())
            net.add_module('pred', nn.Linear(in_channels, 16))

    return net