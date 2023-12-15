"""Partition data and distribute to clients."""
import random
from enum import Enum

from zod import constants
from zod import ZodFrames
import numpy as np
import json
import pandas as pd
df = pd.read_csv("metadata.csv")
random.seed(42)

class PartitionStrategy(Enum):
    """Partition Strategy enum."""

    RANDOM = "random"
    LOCATION = "location"
    SPECIFIC = "specific"
    SPLIT = "split"
    

# load data based on cid and strategy
def partition_train_data(
    strat: PartitionStrategy,
    no_clients: int,
    zod_frames: ZodFrames,
    percentage_of_data: int,
) -> dict:
    """Partition train data.

    Data partition will be saved as a dictionary client_number -> [frames_id's] and this
    dict is downloaded by the client that loads the correct elements by the idx list
    in the dictionary.

    Args:
        strat (PartitionStrategy): partition strategy
        no_clients (int): number of clients
        zod_importer (ZODImporter): dataset importer
        percentage_of_data (int): percentage of data to partition

    Returns:
        dict: client_number -> frames_id map
    """
    training_frames_all = zod_frames.get_split(constants.TRAIN)

    ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")
    print("loaded stored ground truth")

    training_frames_all = [
        idx for idx in training_frames_all if is_valid_frame(idx, ground_truth)
    ]

    # randomly sample by percentage of data
    sampled_training_frames = random.sample(
        training_frames_all, int(len(training_frames_all) * percentage_of_data)
    )

# ==================================================RANDOM==================================================
    
    if strat == PartitionStrategy.RANDOM:
        cid_partitions = {}
        random.shuffle(sampled_training_frames)
        sublist_size = len(sampled_training_frames) // no_clients
        for i in range(no_clients):
            cid_partitions[str(i)] = sampled_training_frames[
                i * sublist_size : (i + 1) * sublist_size
            ]


# ==================================================LOCATION==================================================

    if strat == PartitionStrategy.LOCATION:
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans
        cid_partitions = {}
        
        df = pd.read_csv("metadata.csv")

        if 'latitude' in df.columns:
            # Proceed with the operation on 'latitude' column
            df['latitude_rad'] = np.radians(df['latitude'])
            df['longitude_rad'] = np.radians(df['longitude'])
        else:
            print("Latitude column not found in the DataFrame.")

        # Convert the latitude and longitude columns to radians
        df['latitude_rad'] = np.radians(df['latitude'])
        df['longitude_rad'] = np.radians(df['longitude'])

        # Create a new column with tuples of (latitude, longitude) in radians
        df['Coordinates'] = list(zip(df['latitude_rad'], df['longitude_rad']))


        #K-Means clustering model:
        kmeans = KMeans(n_clusters=no_clients, random_state=1)

        # Fit the K-Means mode
        df['Cluster'] = kmeans.fit_predict(df[['latitude_rad', 'longitude_rad']])

        df = df.sort_values(by="Cluster")
        df["frame_id"] = df["frame_id"].astype(str)


        for i in range(0, no_clients):
        
            # Filter the DataFrame for the current cluster
            cluster_data = df[df['Cluster'] == i]

            #how much data to keep in every cluster
            num_rows_to_keep = int(len(cluster_data) * percentage_of_data)
            sampled_cluster = cluster_data.sample(n=num_rows_to_keep)

            
            # Extract the 'frame_id' values as a list
            frame_ids = sampled_cluster['frame_id'].tolist()
            # frame_ids = [str(x).zfill(6) for x in list(set(sampled_training_frames_int) & set(road_indices))]

            # Add the frame_ids to the dictionary with the cluster as the key
            cid_partitions[i] = frame_ids

        cid_partitions = {str(key): [str(x).zfill(6) for x in value] for key, value in cid_partitions.items()}
        # cid_partitions = cid_partitions.
 

# ===================================================SPLIT===================================================
    
    if strat == PartitionStrategy.SPLIT:
        import pandas as pd 
        df = pd.read_csv("DFRemainingWOGB.csv")

        snow_records = df[df["road_condition"] == "snow"]["frame_id"].tolist()
        wet_records = df[df["road_condition"] == "wet"]["frame_id"].tolist()
        normal_records = df[df["road_condition"] == "normal"]["frame_id"].tolist()

        lens = (len(snow_records)//no_clients) * percentage_of_data
        lenw = (len(wet_records)//no_clients) * percentage_of_data
        lenn = (len(normal_records)//no_clients) * percentage_of_data

        cid_partitions = {}

        for i in range(no_clients):
            key = f'{i}'
            cid_partitions[key] = (snow_records[lens * i : lens + lens * i]+
                            wet_records[lenw * i : lenw + lenw * i]+
                            normal_records[lenn * i : lenn + lenn * i])

        cid_partitions = {str(key): [str(x).zfill(6) for x in value] for key, value in cid_partitions.items()}


# ==================================================SPECIFIC==================================================
        
    if strat == PartitionStrategy.SPECIFIC:
        import pandas as pd
        df = pd.read_csv("metadata.csv")
        cid_partitions = {}

        conditions = ['wet', 'normal', 'snow']
        
        for i, condition in enumerate(conditions):

            road_indices = list(df.loc[df['road_condition'] == condition].index)
            sampled_training_frames_int = [int(x) for x in sampled_training_frames]

            overlaps = [str(x).zfill(6) for x in list(set(sampled_training_frames_int) & set(road_indices))]
            cid_partitions[str(i)] = overlaps

        for i in range(no_clients-(len(conditions))):
            cid_partitions = split_longest_list(cid_partitions)

    print(cid_partitions)
    return cid_partitions


def split_longest_list(input_dict):
    # Find the longest list and its key
    max_key = max(input_dict, key=lambda key: len(input_dict[key]))
    max_length = len(input_dict[max_key])

    # Create a new key for the split list
    highest_key = (max(input_dict.keys(),key=int))
    new_key = str(int(highest_key) + 1)

    # Calculate the splitting point
    split_point = max_length // 2

    # Split the longest list
    split_list = input_dict[max_key][:split_point]
    input_dict[str(new_key)] = input_dict[max_key][split_point:]
    input_dict[str(max_key)] = split_list

    return input_dict

def is_valid_frame(frame_id: str, ground_truth: dict) -> bool:
    """Check if frame is valid."""
    if frame_id == "005350":
        return False
    
    return frame_id in ground_truth


def load_ground_truth(path: str) -> dict:
    """Load ground truth from file."""
    with open(path) as json_file:
        gt = json.load(json_file)

    for f in gt:
        gt[f] = np.array(gt[f])

    return gt