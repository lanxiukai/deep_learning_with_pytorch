'''
Download Dataset Test
'''

import pandas as pd
import d2l_save

try:
    train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size=256)
    print('Fashion-MNIST Dataset Downloaded')
except Exception as e:
    print(f'Fashion-MNIST Dataset Download Failed: {e}')

try:
    train_data = pd.read_csv(d2l_save.download('kaggle_house_train'))
    test_data = pd.read_csv(d2l_save.download('kaggle_house_test'))
    print('Kaggle House Price Prediction Dataset Downloaded')
except Exception as e:
    print(f'Kaggle House Price Prediction Dataset Download Failed: {e}')
