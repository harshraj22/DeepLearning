from collections import defaultdict
from typing import List, AnyStr
import unittest
import pandas as pd
import enum
import logging
from copy import copy
from pprint import pprint

logging.basicConfig(level=logging.NOTSET)

class Phase(enum.Enum):
    Train = enum.auto()
    Val = enum.auto()


class CategoryEncoder:
    """Given a list of set of categories, build up indexes corresponding to
    each category. Each category in each column is mapped to a unique value
    (here called indexes). """
    def __init__(self, all_categories:List[List[AnyStr]]):
        """
        Args:
            all_categories (List[List[Str]]): List of set of categories. eg.
            [
                ['A', 'B', 'C'],
                ['D', 'E'],
                ['F', 'G', 'H', 'I', 'J']
            ]
            The sets need not be of the same size
        """
        self.all_categories = all_categories

        # a list containing category's mapping to an index.
        # self.index[i] represents the mapping corresponding to i'th column in dataset(csv)
        self.index, self.offset, offset = [], 0, 0

        for column_index, categories in enumerate(all_categories):
            categories = set(categories)
            # read here for strange lambda fuction: https://stackoverflow.com/q/70539432/10127204
            self.index.append(defaultdict(lambda offset=offset: offset))

            for index, category in enumerate(categories):
                self.index[column_index][category] = index + 1 + offset
            
            offset += len(categories) + 1
        
        self.offset = offset

    def get_index(self, column_index, word):
        return self.index[column_index][word]

    def __len__(self):
        return self.offset


class TestCategoryEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all_categories =[
                ['A', 'B', 'C', 'A'],
                ['D', 'E'],
                ['F', 'G', 'H', 'I', 'J']
            ] 
        cls.encoder = CategoryEncoder(cls.all_categories)

    def test_len_of_categories(self):
        self.assertEqual(13, len(self.encoder))

    def test_index_of_missing_value(self):
        self.assertEqual(0, self.encoder.get_index(0, 'D'))


if __name__ == '__main__':
    unittest.main()