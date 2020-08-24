# Description : a test class to test sum of a fraction
# Auther      : Ouye Xie
# Created on  : 20/05/2014
# Copyright   : Copyright (c) 2013-2014  Patsnap,
#               All Rights Reserved.
#
#               Use and copying of this software and preparation of derivative works
#               based upon this software are not permitted.
#
#               All information contained herein is, and remains the property of
#               Patsnap and its suppliers, if any. The intellectual and technical
#               concepts contained herein are proprietary to Patsnap and its suppliers
#               and may be covered by U.S. and Foreign Patents, patents in process,
#               and are protected by trade secret or copyright law.
#               Dissemination of this information or reproduction of this material
#               is strictly forbidden unless prior written permission is obtained
#               from Patsnap.
#
#               This software is made available AS IS, and COPYRIGHT OWNERS DISCLAIMS
#               ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE
#               IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#               PURPOSE, AND NOTWITHSTANDING ANY OTHER PROVISION CONTAINED HEREIN, ANY
#               LIABILITY FOR DAMAGES RESULTING FROM THE SOFTWARE OR ITS USE IS
#               EXPRESSLY DISCLAIMED, WHETHER ARISING IN CONTRACT, TORT (INCLUDING
#               NEGLIGENCE) OR STRICT LIABILITY, EVEN IF COPYRIGHT OWNERS ARE ADVISED
#               OF THE POSSIBILITY OF SUCH DAMAGES.

__author__ = 'ouyexie'
__version__ = "0.1"

import sys
import math
import random

class Sum_test( object ):
    def __init__( self ):
        random.seed(10)
        num = 100
        self.a = [random.randint(1,10) * 10 for i in range(num)]
        self.b = [random.randint(0,10) for i in range(num)]


    def compute( self ):
        print("a: " + str(self.a))
        print("b: " + str(self.b))
        exact = 0.0
        for i in range(0, len(self.a)):
            if self.b[i] != 0:
                exact += self.a[i]/self.b[i]
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
        
        gap_percentage = gap/exact
        print("gap percentage value: %f%%" % (gap_percentage*100.0))
        
        print("#########################################")
        
        sum_a = sum(self.a)
        print("sum_a: %f" % sum_a)
        
        sum_b = sum([(1.0/item) for item in self.b if item != 0])
        print("sum_b: %f" % sum_b)
        
        approximate = (sum_a * sum_b) / len(self.a)
        print("approxiate value: %f" % approximate)
        
        gap = exact - approximate
        print("gap value; %f" % gap)
        
        gap_percentage = gap/exact
        print("gap percentage value: %f%%" % (gap_percentage*100.0))

if __name__ == "__main__":

    print("start computing")
    sum_test = Sum_test()
    sum_test.compute()
    print("finish computing")
    


