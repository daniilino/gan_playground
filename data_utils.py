import os
import torch
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, random_split

def collect_only_files(root):
    full_paths = [os.path.join(root, entry) for entry in os.listdir(root) if os.path.isfile(os.path.join(root, entry))]
    names = [os.path.split(path)[1] for path in full_paths]

    names, full_paths = list(names), list(full_paths)
    names, full_paths = zip(*sorted(zip(names, full_paths)))

    return full_paths, names

def collect_only_dirs(root):
    full_paths = [os.path.join(root, entry)for entry in os.listdir(root) if os.path.isdir(os.path.join(root, entry))]
    names = [os.path.split(path)[1] for path in full_paths]

    names, full_paths = zip(*sorted(zip(names, full_paths)))
    names, full_paths = list(names), list(full_paths)

    return full_paths, names

def update_dict(log_dict, new_data, label):
    for k, v in new_data.items():
        log_dict[f"{label} {k}"] = v

    return log_dict

def calculate_mean_for_list_of_dicts(list_of_dicts):
    keys = list(list_of_dicts[0].keys())
    values = [list(d.values()) for d in list_of_dicts]
    values = torch.tensor(values)
    avg = values.mean(0).tolist()

    new_dict = {k: a for k, a in zip(keys, avg)}

    return new_dict

def calculate_mean_std(raw_data, batch_size=200):
    raw_loader = DataLoader(raw_data, batch_size=batch_size, num_workers=4, shuffle=False)

    num_samples = len(raw_loader.dataset)
    num_batches = num_samples // batch_size

    data_mean = 0.
    data_std = 0.
    for i, (images, _) in enumerate(raw_loader):
        print(f"Calculating MEAN and STD: batch {i+1:8} / {num_batches}", end="\r")
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        data_mean += images.mean(2).sum(0)
        data_std += images.std(2).sum(0)

    data_mean /= num_samples
    data_std /= num_samples

    print("Please save data_mean and data_std values if you don't wanna calculate them once again:")
    print(f"DATA_MEAN: {data_mean}")
    print(f"DATA_STD : {data_std}")
    
    return data_mean, data_std

def split_train_val_test(dataset, val=0.1, test=0.1, batch_size=256, num_workers=None):

    n = len(dataset)  # total number of examples
    i_val  = int(val * n)  # take ~10% for test
    i_test = int(test * n)  # take ~10% for test

    data_train, data_val, data_test = random_split(dataset, [n-i_val-i_test, i_val, i_test])

    # the more cores you take, the longer it takes to get the iterator, 
    # however next() works faster then
    if num_workers is None:
        available_cores = mp.cpu_count() - 1
    else:
        available_cores = num_workers

    loader_train = DataLoader(data_train, batch_size=batch_size, num_workers=available_cores, shuffle=True)
    loader_val   = DataLoader(data_val,   batch_size=batch_size, num_workers=available_cores, shuffle=False)
    loader_test  = DataLoader(data_test,  batch_size=batch_size, num_workers=available_cores, shuffle=False)
    data = {"train": loader_train, "val": loader_val, "test": loader_test}

    return data