import sys

sys.path.append('../fleet-learning-young-talents')

from zod import ZodFrames
from zod import constants
from data_partitioner import partition_train_data, PartitionStrategy
from data_loader import load_datasets
import matplotlib.pyplot as plt
import pandas as pd
import json

NO_CLIENTS = 40


def main() -> None:
    data = {
    'frame_id': [],
    'time': [],
    'country_code': [],
    'scraped_weather': [],
    'collection_car': [],
    'road_type': [],
    'road_condition': [],
    'time_of_day': [],
    'num_lane_instances': [],
    'num_vehicles': [],
    'num_vulnerable_vehicles': [],
    'num_pedestrians': [],
    'num_traffic_lights': [],
    'num_traffic_signs': [],
    'longitude': [],
    'latitude': [],
    'solar_angle_elevation': []
    }

    df = pd.DataFrame(data)

    zod_frames = ZodFrames("/mnt/ZOD", version="full")

    for i in range(100000): #25 min

        frame = zod_frames[str(i)]
        file = open(frame.info.metadata_path)
        metadata = json.load(file)

        new_row = pd.DataFrame(metadata, index=[0])  
        df = pd.concat([df, new_row], ignore_index=True) 
        
    df.to_csv("metadata1.csv",index=False)
    exit()


if __name__ == '__main__':
    main()