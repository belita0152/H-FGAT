# 1. Train/Test Dataset Split

split_ratio = 0.8

def load_data(x, y, split_ratio=0.8):
    split_idx = int(len(x) * split_ratio)
    train_x, train_y = x[:split_idx], y[:split_idx]
    test_x, test_y = x[split_idx:], y[split_idx:]

    return (train_x, train_y), (test_x, test_y)
