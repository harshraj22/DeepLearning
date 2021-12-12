from collections import defaultdict
from typing import List, AnyStr
import unittest
import pandas as pd

class CategoryEncoder:
    """Given a list of set of categories, build up indexes corresponding to
    each category.  """
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
        self.index, self.offset = [], 0

        for column_index, categories in enumerate(all_categories):
            categories = set(categories)
            self.index.append(defaultdict(lambda: self.offset))

            for index, category in enumerate(categories):
                self.index[column_index][category] = index + 1 + self.offset
            
            self.offset += len(category) + 1

    def get_index(self, column_index, word):
        return self.index[column_index][word]

    def __len__(self):
        return self.offset


# ToDo: Create a dummy data frame, and check the correctness of returned indexes
# from the Category Encoder.
class TestCategoryEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
