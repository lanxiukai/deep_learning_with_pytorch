'''
Download Dataset Test
'''

import pandas as pd
import d2l_save

try:
    train_iter, test_iter = d2l_save.vision_loaders(
        dataset="mnist", data_dir="./data", batch_size=256, num_workers=16, pin_memory=True)
    print('The MNIST Dataset has been downloaded.')
except Exception as e:
    print(f'Failed to download the MNIST Dataset: {e}')


try:
    train_iter, test_iter = d2l_save.vision_loaders(
        dataset="fashion_mnist", data_dir="./data", batch_size=256, num_workers=16, pin_memory=True)
    print('The Fashion-MNIST Dataset has been downloaded.')
except Exception as e:
    print(f'Failed to download the Fashion-MNIST Dataset: {e}')


try:
    train_iter, test_iter = d2l_save.vision_loaders(
        dataset="cifar10", data_dir="./data", batch_size=256, num_workers=16, pin_memory=True)
    print('The CIFAR-10 Dataset has been downloaded.')
except Exception as e:
    print(f'Failed to download the CIFAR-10 Dataset: {e}')


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
