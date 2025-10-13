'''
Load Fashion-MNIST dataset test using PyTorch
'''

from pathlib import Path
import d2l_save

def _infer_project_root() -> Path:
    if '__file__' in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()

bash_path = _infer_project_root()

train_iter, test_iter = d2l_save.load_data_fashion_mnist(batch_size=256)

if __name__ == '__main__':
    timer = d2l_save.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.3f} sec')
