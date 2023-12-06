"""Client side data loader."""
from typing import Tuple

from data_partitioner import load_ground_truth, is_valid_frame
from zod_dataset import ZodDataset
from torch import Generator
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision import transforms
from zod import ZodFrames


def load_datasets(
    partitioned_frame_ids: list, 
    zod_frames: ZodFrames
) -> Tuple[DataLoader, DataLoader]:
    """Get train and validation dataloaer."""
   

    
    seed = 42
    transform = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )
    
    ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")


    print("Before valid check:", len(partitioned_frame_ids))


    partitioned_frame_ids = [idx for idx in partitioned_frame_ids if is_valid_frame(idx, ground_truth)]

    print("After valid check:", len(partitioned_frame_ids))

    trainset = ZodDataset(
        zod_frames=zod_frames,
        frames_id_set=partitioned_frame_ids,
        stored_ground_truth=ground_truth,
        transform=transform,
    )

    # Split each partition into train/val and create DataLoader
    len_test = int(len(trainset) * 0.1)
    len_train = int(len(trainset) - len_test)
    print("number of test frames:", len_test)
    print("number of train frames:", len_train)
    print("total frames in client:", (len(trainset))) #Här
    print(trainset)

    lengths = [len_train, len_test]
    ds_train, ds_test = random_split(trainset, lengths, Generator().manual_seed(seed))
    if len(ds_train) > 0:
        train_sampler = RandomSampler(ds_train)
        # Rest of your code using train_sampler
    else:
        print("Training dataset is empty. Check your data splitting process.")

    trainloader = DataLoader(
        ds_train,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        # sampler=train_sampler,

    )
    testloader = DataLoader(ds_test, batch_size=32, num_workers=0)

    return trainloader, testloader

