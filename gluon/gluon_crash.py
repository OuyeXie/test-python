from mxnet import nd

class GluonCrash:

    def testNdArray(self):
        array = nd.array(((1, 2, 3), (5, 6, 7)))
        print("Shape {}".format(array.shape))
        print("Size {}".format(array.size))
        print("DType {}".format(array.dtype))

if __name__ == "__main__":
    gluonCrash: GluonCrash = GluonCrash()
    gluonCrash.testNdArray()

