from mxnet import nd, random
from mxnet.gluon import nn


class GluonCrash:
    random.seed(128)

    def testNdArray(self):
        x = nd.array(((1, 2, 3), (5, 6, 7)))
        print("Shape {}".format(x.shape))
        print("Size {}".format(x.size))
        print("DType {}".format(x.dtype))

        # Operation
        y = nd.random.uniform(-1, 1, (2, 3))
        print(x * y)
        print(y.exp())
        print(nd.dot(x, y.T))

        # Converting between MXNet NDArray and NumPy
        a = x.asnumpy()
        print((type(a), a))
        b = nd.array(a)
        print((type(b), b))

    def testNNDense(self):
        # `Dense` implements the operation:
        #     `output = activation(dot(input, weight) + bias)`
        #     where `activation` is the element-wise activation function
        #     passed as the `activation` argument, `weight` is a weights matrix
        #     created by the layer, and `bias` is a bias vector created by the layer
        #     (only applicable if `use_bias` is `True`).
        layer = nn.Dense(2)
        print("declaration")
        print(layer)
        layer.initialize()
        print("initialization")
        print(layer)
        x = nd.random.uniform(-1, 1, (3, 4))
        print(x)
        layer(x)
        print("feed")
        print(layer)
        print(layer.weight.data())
        print(layer.bias.data())

    def testNNChain(self):
        net = nn.Sequential()
        # Add a sequence of layers.
        net.add(
            # Similar to Dense, it is not necessary to specify the input channels
            # by the argument `in_channels`, which will be  automatically inferred
            # in the first forward pass. Also, we apply a relu activation on the
            # output. In addition, we can use a tuple to specify a  non-square
            # kernel size, such as `kernel_size=(2,4)`
            nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            # One can also use a tuple to specify non-symmetric pool and stride sizes
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
            nn.MaxPool2D(pool_size=2, strides=2),
            # The dense layer will automatically reshape the 4-D output of last
            # max pooling layer into the 2-D shape: (x.shape[0], x.size/x.shape[0])
            nn.Dense(120, activation="relu"),
            nn.Dense(84, activation="relu"),
            nn.Dense(10))
        # Sequential(
        #   (0): Conv2D(None -> 6, kernel_size=(5, 5), stride=(1, 1), Activation(relu))
        #   (1): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
        #   (2): Conv2D(None -> 16, kernel_size=(3, 3), stride=(1, 1), Activation(relu))
        #   (3): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
        #   (4): Dense(None -> 120, Activation(relu))
        #   (5): Dense(None -> 84, Activation(relu))
        #   (6): Dense(None -> 10, linear)
        # )
        print(net)

        net.initialize()
        # Input shape is (batch_size, color_channels, height, width)
        x = nd.random.uniform(shape=(4, 1, 28, 28))
        y = net(x)
        print("x shape {}".format(x.shape))
        [print("layer {} shape {}".format(i, (net[i].weight.data().shape, net[i].bias.data().shape))) for i in
         [0, 2, 4, 5, 6]]
        print("y shape {}".format(y.shape))
        print((net[0].weight.data(), net[0].bias.data()))

    def testNNConv2D(self):
        print("test testNNConv2D")
        # r"""2D convolution layer (e.g. spatial convolution over images).
        #
        #     This layer creates a convolution kernel that is convolved
        #     with the layer input to produce a tensor of
        #     outputs. If `use_bias` is True,
        #     a bias vector is created and added to the outputs. Finally, if
        #     `activation` is not `None`, it is applied to the outputs as well.
        #
        #     If `in_channels` is not specified, `Parameter` initialization will be
        #     deferred to the first time `forward` is called and `in_channels` will be
        #     inferred from the shape of input data.
        #
        # Parameters
        #     ----------
        #     channels : int
        #         The dimensionality of the output space, i.e. the number of output
        #         channels (filters) in the convolution.
        #     kernel_size :int or tuple/list of 2 int
        #         Specifies the dimensions of the convolution window.
        #     strides : int or tuple/list of 2 int,
        #         Specify the strides of the convolution.
        #     padding : int or a tuple/list of 2 int,
        #         If padding is non-zero, then the input is implicitly zero-padded
        #         on both sides for padding number of points
        #
        #     Inputs:
        #         - **data**: 4D input tensor with shape
        #           `(batch_size, in_channels, height, width)` when `layout` is `NCHW`.
        #           For other layouts shape is permuted accordingly.
        #
        #     Outputs:
        #         - **out**: 4D output tensor with shape
        #           `(batch_size, channels, out_height, out_width)` when `layout` is `NCHW`.
        #           out_height and out_width are calculated as::
        #
        #               out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
        #               out_width = floor((width+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1])+1
        layer = nn.Conv2D(channels=6, kernel_size=5, padding=(1, 1), activation='relu')
        print(layer)
        layer.initialize()
        x = nd.random.uniform(shape=(4, 1, 28, 28))
        print("x shape {}".format(x.shape))
        y = layer(x)
        print("layer {} shape {}".format(0, (layer.weight.data().shape, layer.bias.data().shape)))
        print("y shape {}".format(y.shape))
        print(layer)
        # print(layer.weight.data().shape)
        # print(layer.bias.data())

    def testNNConv2DWithMaxPooling(self):
        print("test testNNConv2DWithMaxPooling")
        net = nn.Sequential()
        # Add a sequence of layers.
        net.add(
            # Similar to Dense, it is not necessary to specify the input channels
            # by the argument `in_channels`, which will be  automatically inferred
            # in the first forward pass. Also, we apply a relu activation on the
            # output. In addition, we can use a tuple to specify a  non-square
            # kernel size, such as `kernel_size=(2,4)`
            nn.Conv2D(channels=6, kernel_size=5, padding=(1, 1), activation='relu'),
            # One can also use a tuple to specify non-symmetric pool and stride sizes
            nn.MaxPool2D(pool_size=2, strides=2))

        print(net)

        net.initialize()
        # Input shape is (batch_size, color_channels, height, width)
        x = nd.random.uniform(shape=(4, 1, 28, 28))
        y = net(x)
        print("x shape {}".format(x.shape))
        [print("layer {} shape {}".format(i, (net[i].weight.data().shape, net[i].bias.data().shape))) for i in [0]]
        print("y shape {}".format(y.shape))
        print(net)
        # print((net[0].weight.data(), net[0].bias.data()))

    def testNNChainFlexibly(self):
        class MixMLP(nn.Block):
            def __init__(self, **kwargs):
                # Run `nn.Block`'s init method
                super(MixMLP, self).__init__(**kwargs)
                self.blk = nn.Sequential()
                self.blk.add(nn.Dense(3, activation='relu'),
                             nn.Dense(4, activation='relu'))
                self.dense = nn.Dense(5)

            def forward(self, x):
                y = nd.relu(self.blk(x))
                print("after bulk {}".format(y))
                return self.dense(y)

        net = MixMLP()
        print(net)

        net.initialize()
        x = nd.random.uniform(shape=(2, 2))
        print("x {}".format(x))
        y = net(x)
        print("after dense {}".format(y))
        print(net.blk[0].weight.data().shape)
        print(net.blk[1].weight.data().shape)
        print(net.dense.weight.data().shape)


if __name__ == "__main__":
    gluonCrash: GluonCrash = GluonCrash()
    # gluonCrash.testNdArray()
    # gluonCrash.testNNDense()
    gluonCrash.testNNConv2D()
    gluonCrash.testNNConv2DWithMaxPooling()
    # gluonCrash.testNNChain()
    # gluonCrash.testNNChainFlexibly()
