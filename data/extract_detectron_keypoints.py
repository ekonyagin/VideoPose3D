import logging
import os

import numpy as np


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)

    dataset = np.load('./data_2d_custom_videos.npz', allow_pickle=True)

    for key, value in dataset['positions_2d'].item().items():
        for i, keypoints in enumerate(value['custom']):
            metadata = dataset['metadata'].item()['video_metadata'][key]
            logging.info(f'Saving keypoints from {key} with shape {keypoints.shape}')
            np.save(f'./detectron_keypoints/{key}_{i}', {
                'keypoints': keypoints,
                'metadata': metadata
            })
