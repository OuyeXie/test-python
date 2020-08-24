# Description : a test class to test sum of a fraction
# Auther      : Ouye Xie
# Created on  : 20/05/2014

__author__ = 'ouyexie'
__version__ = "0.1"

import random


class SumTest(object):
    def __init__(self):
        random.seed(10)
        num = 100
        self.a = [random.randint(1, 10) * 10 for i in range(num)]
        self.b = [random.randint(0, 10) for i in range(num)]

    def compute(self):
        print("a: " + str(self.a))
        print("b: " + str(self.b))
        exact = 0.0
        for i in range(0, len(self.a)):
            if self.b[i] != 0:
                exact += self.a[i] / self.b[i]
        print("exact value: %f" % exact)

        print("#########################################")

        sum_a = sum(self.a)
        print("sum_a: %f" % sum_a)

        sum_b = sum(self.b)
        print("sum_b: %f" % sum_b)

        approximate = ((sum_a * len(self.a)) / sum_b)
        print("approxiate value: %f" % approximate)

        gap = exact - approximate
        print("gap value; %f" % gap)

        gap_percentage = gap / exact
        print("gap percentage value: %f%%" % (gap_percentage * 100.0))

        print("#########################################")

        sum_a = sum(self.a)
        print("sum_a: %f" % sum_a)

        sum_b = sum([(1.0 / item) for item in self.b if item != 0])
        print("sum_b: %f" % sum_b)

        approximate = (sum_a * sum_b) / len(self.a)
        print("approxiate value: %f" % approximate)

        gap = exact - approximate
        print("gap value; %f" % gap)

        gap_percentage = gap / exact
        print("gap percentage value: %f%%" % (gap_percentage * 100.0))


if __name__ == "__main__":
    print("start computing")
    sum_test = Sum_test()
    sum_test.compute()
    print("finish computing")
