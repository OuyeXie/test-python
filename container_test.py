__author__ = 'ouyexie'
__version__ = "0.1"

from collections import ChainMap


class ContainerTest(object):

    def test_chain_map(self):
        baseline: dict = {'music': 'bach', 'art': 'rembrandt'}
        adjustments: dict = {'art': 'van gogh', 'opera': 'carmen'}
        chainMap: ChainMap = ChainMap(baseline, adjustments)

        print(list(baseline))
        print(list(adjustments))

        print(chainMap.keys())
        print(chainMap.maps)

        print(list(chainMap))
        print(list(chainMap.maps))

        combined: dict= baseline.copy()
        combined.update(adjustments)
        print(list(combined))


if __name__ == "__main__":
    containerTest: ContainerTest = ContainerTest()
    containerTest.test_chain_map()
