__author__ = 'ouyexie'
__version__ = "0.1"

import gluonnlp
import numpy as np


class SamplerTest(object):
    np.random.seed(100)

    # https://gluon-nlp.mxnet.io/api/notes/data_api.html
    #   if minibatcheas are formed using uniform sampling, it can cause a large amount of padding
    #   consider constructing a sampler using bucketing, which defines how the samples in a dataset will be iterated in a more economic way
    def testFixedBucketSampler(self):
        lengths = [np.random.randint(1, 100) for _ in range(1000)]
        sampler = gluonnlp.data.FixedBucketSampler(lengths, 8, ratio=0)
        # FixedBucketSampler:
        #   sample_num=1000, batch_num=130
        #   key=[18, 27, 36, 45, 54, 63, 72, 81, 90, 99]
        #   cnt=[177, 86, 86, 91, 97, 91, 92, 82, 102, 96]
        #   batch_size=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        #
        # batch_num: how many batches in total
        # import math
        #     from functools import reduce
        #     values = list(map(lambda x: math.ceil(x/8.0), [177, 86, 86, 91, 97, 91, 92, 82, 102, 96]))
        #     sum = reduce(lambda x, y: x+y, values)
        #     print(values)
        #     print(sum)
        # [23, 11, 11, 12, 13, 12, 12, 11, 13, 12]
        # 130
        #
        # key: bucket_keys, details can be found here from ConstWidthBucket (lengths in range [1, 100))
        #
        # cnt shows the number of samples belonging to each bucket
        print(sampler.stats())

        sampler = gluonnlp.data.FixedBucketSampler(lengths, 8, ratio=0.5)
        # FixedBucketSampler:
        #   sample_num=1000, batch_num=109
        #   key=[18, 27, 36, 45, 54, 63, 72, 81, 90, 99]
        #   cnt=[177, 86, 86, 91, 97, 91, 92, 82, 102, 96]
        #   batch_size=[22, 14, 11, 8, 8, 8, 8, 8, 8, 8]
        #
        # consider scaling up the batch size of smaller buckets by setting ratio gt 0
        print(sampler.stats())


if __name__ == "__main__":
    samplerTest: SamplerTest = SamplerTest()
    samplerTest.testFixedBucketSampler()
