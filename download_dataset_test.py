'''
Download Dataset Test
'''

import pandas as pd
import d2l_save

try:
    train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size=256)
    print('The Fashion-MNIST Dataset has been downloaded.')
except Exception as e:
    print(f'Failed to download the Fashion-MNIST Dataset: {e}')

try:
    train_data = pd.read_csv(d2l_save.download('kaggle_house_train'))
    test_data = pd.read_csv(d2l_save.download('kaggle_house_test'))
    print('The Kaggle House Price Prediction Dataset has been downloaded.')
except Exception as e:
    print(f'Failed to download the Kaggle House Price Prediction Dataset: {e}')


try:
    lines = d2l_save.read_time_machine()
    print(f'The Time Machine Dataset has been downloaded: {lines[0]}')
except Exception as e:
    print(f'Failed to download the Time Machine Dataset: {e}')
