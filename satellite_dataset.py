"""
Author: JudePark <judepark@kookmin.ac.kr>
"""
import multiprocessing

from torch.utils.data import DataLoader, Dataset
from preprocess_image import image2tensor, process2dataset



class SatelliteDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.src = dataset

    def __getitem__(self, index):
        return self.src[index]

    def __len__(self) -> int:
        return len(self.src)

    def get_num_class(self) -> int: return 4


def collate_fn(batch):
    src, info = zip(*batch)

    assert len(info[0][0]) == len(info[0][1])
    # image tensor, bbox informations /w theta, labels
    return src[0], info[0][0], info[0][1]


def get_data_loader(dataset: Dataset, collate_fn:object=None, bs: int=32, shuffle:bool=True) -> DataLoader: return DataLoader(dataset,
                                                                                                      batch_size=bs,
                                                                                                      shuffle=shuffle,
                                                                                                      num_workers=multiprocessing.cpu_count(),
                                                                                                      collate_fn=collate_fn)

if __name__ == '__main__':
    """
    Example for use.
    """
    dataset = process2dataset(image_folder_path='./rsc/sample_images/', json_path='./rsc/sample_images/labels.json')
    dataset = SatelliteDataset(dataset)
    sample_loader = get_data_loader(dataset, bs=2, collate_fn=collate_fn)

    for i, (src, bbox, label) in enumerate(sample_loader):
        print(src)
        print(bbox)
        print(label)
        break


    pass