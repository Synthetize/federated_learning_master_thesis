from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from flwr_datasets.utils import divide_dataset
from collections import Counter
import pandas as pd
from datasets import load_dataset

fd = None

train_transform = transforms.Compose([
    transforms.ToTensor(),
])
test_validation_transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_transform_fn(transform):
    def apply_transforms(batch):
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch
    return apply_transforms

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    global fd
    if fd is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, 
            partition_by="label",
            alpha=0.1, 
            seed=42)
        fd = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})

    partition_train_val = fd.load_partition(partition_id)
    train, valid = divide_dataset(partition_train_val, [0.8, 0.2])

    train = train.with_transform(get_transform_fn(train_transform))
    valid = valid.with_transform(get_transform_fn(test_validation_transform))
    
    return DataLoader(train, batch_size=batch_size), DataLoader(valid, batch_size=batch_size)

def load_centralized_dataset(batch_size):
    test_dataset = load_dataset("cifar10", split="test")
    test_dataset = test_dataset.with_transform(get_transform_fn(test_validation_transform))
    return DataLoader(test_dataset, batch_size=batch_size)
    

# def compare_all_distributions(num_partitions: int):

#     all_counts = {}
#     for i in range(num_partitions):
#         train_loader, val_loader = load_data(partition_id=i, num_partitions=num_partitions, batch_size=32)
#         train_labels = train_loader.dataset["label"]
#         val_labels = val_loader.dataset["label"]
#         all_counts[f"Client {i}"] = Counter(train_labels) + Counter(val_labels)

#     df = pd.DataFrame(all_counts).fillna(0).astype(int)
#     df.index.name = "Classe"
#     df = df.sort_index()
    
#     print(df.to_string())
#     return df

# if __name__ == "__main__":
#     N_CLIENTS = 10
#     dist_df = compare_all_distributions(num_partitions=N_CLIENTS)
    
#     print(f"\Sum of all samples: {dist_df.sum().sum()}")