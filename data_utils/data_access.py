import logging
import numpy as np
import sys
sys.path.append("/home/sunzehui/GeneNet/data_utils/")
from data_processor.data_reader import DataClass 

class Data():
    """Data Class
    """
    def __init__(self, id, type, params, test_size=0.3, stratify=True):

        self.test_size = test_size
        self.stratify = stratify
        self.data_type = type
        self.data_params = params
        if self.data_type == 'data_processor':
            self.data_reader = DataClass(**params)
        else:
            logging.error('unsupported GeneNet_data type')
            raise ValueError('unsupported GeneNet_data type')

    def get_train_validate_test(self):
        return self.data_reader.get_train_validate_test()

    def get_train_test(self):
        x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = self.data_reader.get_train_validate_test()
        x_train = np.concatenate((x_train, x_validate))
        y_train = np.concatenate((y_train, y_validate))
        info_train = list(info_train) + list(info_validate)
        return x_train, x_test, y_train, y_test, info_train, info_test, columns

    def get_data(self):
        x = self.data_reader.x
        y = self.data_reader.y
        info = self.data_reader.info
        columns = self.data_reader.columns
        return x, y, info, columns

    # def get_relevant_features(self):
    #     if hasattr(self.data_reader, 'relevant_features'):
    #         return self.data_reader.get_relevant_features()
    #     else:
    #         return None
