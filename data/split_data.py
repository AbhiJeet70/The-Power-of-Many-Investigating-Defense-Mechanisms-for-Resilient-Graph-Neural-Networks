import torch
import numpy as np

def split_dataset(data, test_size=0.2, val_size=0.1, device=torch.device('cpu')):
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    num_test = int(test_size * num_nodes)
    num_val = int(val_size * num_nodes)
    num_train = num_nodes - num_test - num_val

    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)

    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train + num_val]] = True
    test_mask[indices[num_train + num_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Target and clean test masks
    num_target = int(0.1 * num_nodes)
    target_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    clean_test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    target_mask[indices[num_train + num_val:num_train + num_val + num_target]] = True
    clean_test_mask[indices[num_train + num_val + num_target:]] = True

    data.target_mask = target_mask
    data.clean_test_mask = clean_test_mask

    return data
