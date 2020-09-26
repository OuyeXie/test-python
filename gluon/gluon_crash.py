from mxnet import nd, random, autograd, gluon, init, image, gpu
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.utils import download

from IPython import display
import matplotlib.pyplot as plt
import time


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
        # Parameters
        #     ----------
        #     units : int
        #         Dimensionality of the output space.
        #     in_units : int, optional
        #         Size of the input data. If not specified, initialization will be
        #         deferred to the first time `forward` is called and `in_units`
        #         will be inferred from the shape of input data.
        # Inputs:
        #         - **data**: if `flatten` is True, `data` should be a tensor with shape
        #           `(batch_size, x1, x2, ..., xn)`, where x1 * x2 * ... * xn is equal to
        #           `in_units`. If `flatten` is False, `data` should have shape
        #           `(x1, x2, ..., xn, in_units)`.
        #
        # Outputs:
        #         - **out**: if `flatten` is True, `out` will be a tensor with shape
        #           `(batch_size, units)`. If `flatten` is False, `out` will have shape
        #           `(x1, x2, ..., xn, units)`.
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

    # MixMLP(
    #   (blk0): Sequential(
    #     (0): Conv2D(None -> 6, kernel_size=(5, 5), stride=(1, 1), Activation(relu))
    #     (1): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    #   )
    #   (blk1): Sequential(
    #     (0): Conv2D(None -> 16, kernel_size=(3, 3), stride=(1, 1), Activation(relu))
    #     (1): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    #   )
    #   (dense0): Dense(None -> 120, Activation(relu))
    #   (dense1): Dense(None -> 84, Activation(relu))
    #   (dense2): Dense(None -> 10, linear)
    # )
    # x shape (4, 1, 28, 28)
    # blk 0-0 shape ((6, 1, 5, 5), (6,))
    # blk 0 result shape (4, 6, 12, 12)
    # blk 1-0 shape ((16, 6, 3, 3), (16,))
    # blk 0 result shape (4, 16, 5, 5)
    # dense 0 shape ((120, 400), (120,))
    # dense 0 result shape (4, 120)
    # dense 1 shape ((84, 120), (84,))
    # dense 1 result shape (4, 84)
    # dense 2 shape ((10, 84), (10,))
    # dense 2 result shape (4, 10)
    # y shape (4, 10)
    # y
    # [[-0.00525905 -0.00061266 -0.00537567 -0.00111964  0.00321069 -0.00222969
    #   -0.00345072  0.00083281 -0.00227492 -0.00129272]
    #  [-0.00469876 -0.00015404 -0.00548729 -0.00192122  0.00372589 -0.00208581
    #   -0.00302593  0.00175308 -0.00210345 -0.00071133]
    #  [-0.00470831 -0.00064738 -0.00581248 -0.00120577  0.00372367 -0.0019953
    #   -0.00276755  0.00097445 -0.00215606 -0.00058186]
    #  [-0.0049872  -0.00120375 -0.00555822 -0.00121179  0.00334217 -0.00201681
    #   -0.00269017  0.00139254 -0.00257135 -0.00025123]]
    # <NDArray 4x10 @cpu(0)>
    def testNNChainFlexiblyMore(self):
        class MixMLP(nn.Block):
            def __init__(self, **kwargs):
                # Run `nn.Block`'s init method
                super(MixMLP, self).__init__(**kwargs)
                self.blk0 = nn.Sequential()
                self.blk0.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                              nn.MaxPool2D(pool_size=2, strides=2))
                self.blk1 = nn.Sequential()
                self.blk1.add(nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                              nn.MaxPool2D(pool_size=2, strides=2))
                self.dense0 = nn.Dense(120, activation="relu")
                self.dense1 = nn.Dense(84, activation="relu")
                self.dense2 = nn.Dense(10)

            def forward(self, x):
                blk0Result = self.blk0(x)
                print("blk {} shape {}".format("0-0",
                                               (self.blk0[0].weight.data().shape, self.blk0[0].bias.data().shape)))
                print("blk {} result shape {}".format(0, blk0Result.shape))

                blk1Result = self.blk1(blk0Result)
                print("blk {} shape {}".format("1-0",
                                               (self.blk1[0].weight.data().shape, self.blk1[0].bias.data().shape)))
                print("blk {} result shape {}".format(0, blk1Result.shape))

                dense0Result = self.dense0(blk1Result)
                print("dense {} shape {}".format(0,
                                                 (self.dense0.weight.data().shape, self.dense0.bias.data().shape)))
                print("dense {} result shape {}".format(0, dense0Result.shape))

                dense1Result = self.dense1(dense0Result)
                print("dense {} shape {}".format(1,
                                                 (self.dense1.weight.data().shape, self.dense1.bias.data().shape)))
                print("dense {} result shape {}".format(1, dense1Result.shape))

                dense2Result = self.dense2(dense1Result)
                print("dense {} shape {}".format(2,
                                                 (self.dense2.weight.data().shape, self.dense2.bias.data().shape)))
                print("dense {} result shape {}".format(2, dense2Result.shape))

                return dense2Result

        net = MixMLP()
        print(net)

        net.initialize()
        x = nd.random.uniform(shape=(4, 1, 28, 28))
        print("x shape {}".format(x.shape))
        y = net(x)
        print("y shape {}".format(y.shape))
        print("y {}".format(y))

    def testAutoGrad(self):
        # simple
        x = nd.array([[1, 2], [3, 4]])
        x.attach_grad()
        with autograd.record():
            y = 2 * x * x
        y.backward()  # When y has more than one entry, y.backward() is equivalent to y.sum().backward().
        print(x.grad)

        def f(a):
            b = a * 2
            while b.norm().asscalar() < 1000:
                b = b * 2
            if b.sum().asscalar() >= 0:
                c = b[0]
            else:
                c = b[1]
            return c

        # dynamic
        a = nd.random.uniform(shape=2)
        print(a)
        a.attach_grad()
        with autograd.record():
            c = f(a)
        c.backward()
        print(c)
        print((a.grad, c / a))

    # run on linux
    def testTrainNNWithFashionMNIST(self):

        # auxiliary function to calculate the model accuracy
        def acc(output, label):
            # output: (batch, num_output) float32 ndarray
            # label: (batch, ) int32 ndarray
            return (output.argmax(axis=1) ==
                    label.astype('float32')).mean().asscalar()

        mnist_train = datasets.FashionMNIST(train=True)
        X, y = mnist_train[0]
        # (height, width, channel)
        # ('X shape: ', (28, 28, 1), 'X dtype', <class 'numpy.uint8'>, 'y:', 2)
        print(('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y))

        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        X, y = mnist_train[0:10]
        print(('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y))
        # plot images
        # display.set_matplotlib_formats('svg')
        # _, figs = plt.subplots(1, X.shape[0], figsize=(15, 15))
        # for f, x, yi in zip(figs, X, y):
        #     # 3D->2D by removing the last channel dim
        #     f.imshow(x.reshape((28, 28)).asnumpy())
        #     ax = f.axes
        #     ax.set_title(text_labels[int(yi)])
        #     ax.title.set_fontsize(14)
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        # plt.show()

        # In order to feed data into a Gluon model, we need to convert the images to the (channel, height, width) format with a floating point data type
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.13, 0.31)])
        mnist_train = mnist_train.transform_first(transformer)
        X, y = mnist_train[0:10]
        print(('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y))

        # get a (randomized) batch of examples
        batch_size = 256
        train_data = gluon.data.DataLoader(
            mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
        for data, label in train_data:
            print(data.shape, label.shape)
            break

        # create a validation dataset and data loader
        mnist_valid = gluon.data.vision.FashionMNIST(train=False)
        # This works on Linux with Conda env test-python
        valid_data = gluon.data.DataLoader(
            mnist_valid.transform_first(transformer),
            batch_size=batch_size, num_workers=4)

        # Define the mode
        net = nn.Sequential()
        net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Flatten(),
                nn.Dense(120, activation="relu"),
                nn.Dense(84, activation="relu"),
                nn.Dense(10))
        # changed the weight initialization method to Xavier
        net.initialize(init=init.Xavier())

        # use standard softmax cross entropy loss for classification problems
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        # standard stochastic gradient descent with constant learning rate of 0.1
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

        # training loop
        for epoch in range(10):
            train_loss, train_acc, valid_acc = 0., 0., 0.
            tic = time.time()
            for data, label in train_data:
                # forward + backward
                with autograd.record():
                    output = net(data)
                    loss = softmax_cross_entropy(output, label)
                loss.backward()
                # update parameters
                trainer.step(batch_size)
                # calculate training metrics
                train_loss += loss.mean().asscalar()
                train_acc += acc(output, label)
            # calculate validation accuracy
            for data, label in valid_data:
                valid_acc += acc(net(data), label)
            print("Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec" % (
                epoch, train_loss / len(train_data), train_acc / len(train_data),
                valid_acc / len(valid_data), time.time() - tic))

        # save the trained parameters onto disk
        net.save_parameters('net.params')

    # run on linux
    def testPredictNNWithSavedParams(self):
        # copy a simple model’s definition
        net = nn.Sequential()
        net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Flatten(),
                nn.Dense(120, activation="relu"),
                nn.Dense(84, activation="relu"),
                nn.Dense(10))

        # load params back
        net.load_parameters('net.params')

        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.13, 0.31)])

        mnist_valid = datasets.FashionMNIST(train=False)
        X, Y = mnist_valid[:10]
        preds = []
        for x, y in zip(X, Y):
            x = transformer(x).expand_dims(axis=0)
            pred = net(x).argmax(axis=1)
            preds.append((pred.astype('int32').asscalar(), y))
        print(preds)

        # _, figs = plt.subplots(1, 10, figsize=(15, 15))
        # text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
        #                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        # display.set_matplotlib_formats('svg')
        # for f, x, yi, pyi in zip(figs, X, y, preds):
        #     f.imshow(x.reshape((28, 28)).asnumpy())
        #     ax = f.axes
        #     ax.set_title(text_labels[yi] + '\n' + text_labels[pyi])
        #     ax.title.set_fontsize(14)
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        # plt.show()

    def testPredictNNWithResNet50V2(self):
        # ResNet-50 V2 model was trained on the ImageNet dataset
        net = models.resnet50_v2(pretrained=True)

        url = 'http://data.mxnet.io/models/imagenet/synset.txt'
        fname = download(url)
        with open(fname, 'r') as f:
            text_labels = [' '.join(l.split()[1:]) for l in f]

        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/" + \
              "Golden_Retriever_medium-to-light-coat.jpg/" + \
              "365px-Golden_Retriever_medium-to-light-coat.jpg"
        fname = download(url)
        x = image.imread(fname)

        x = image.resize_short(x, 256)
        x, _ = image.center_crop(x, (224, 224))

        # plt.imshow(x.asnumpy())
        # plt.show()

        def transform(data):
            data = data.transpose((2, 0, 1)).expand_dims(axis=0)
            rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
            rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
            return (data.astype('float32') / 255 - rgb_mean) / rgb_std

        prob = net(transform(x)).softmax()
        idx = prob.topk(k=5)[0]
        for i in idx:
            i = int(i.asscalar())
            print('With prob = %.5f, it contains %s' % (
                prob[0, i].asscalar(), text_labels[i]))

    # TODO: test GPU with hoverboard cluster?
    # def testGpuBasics(self):
    #     x = nd.ones((3, 4), ctx=gpu())
    #     print(x)



if __name__ == "__main__":
    gluonCrash: GluonCrash = GluonCrash()

    # gluonCrash.testNdArray()

    # gluonCrash.testNNDense()
    # gluonCrash.testNNChain()
    # gluonCrash.testNNConv2D()
    # gluonCrash.testNNConv2DWithMaxPooling()
    # gluonCrash.testNNChainFlexibly()
    # gluonCrash.testNNChainFlexiblyMore()

    # gluonCrash.testAutoGrad()

    # gluonCrash.testTrainNNWithFashionMNIST()

    # gluonCrash.testPredictNNWithSavedParams()
    gluonCrash.testPredictNNWithResNet50V2()

    # gluonCrash.testGpuBasics()