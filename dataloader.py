import os

import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class DynTexTrainDataset(Dataset):
    def __init__(self, data_root, effect, aug=False):
        self.data_root = os.path.join(data_root, effect)
        self.effect = effect
        self.data_list = sorted(os.listdir(self.data_root), key=lambda x: int(x.replace('.png', '')))

    def __len__(self):
        return len(self.data_list) - 1

    def __getitem__(self, item):
        path1 = self.data_list[item]
        path2 = self.data_list[item + 1]
        source = cv2.imread(os.path.join(self.data_root, path1),
                            cv2.IMREAD_UNCHANGED).transpose([2, 0, 1]).astype('double') / 255.0
        target = cv2.imread(os.path.join(self.data_root, path2),
                            cv2.IMREAD_UNCHANGED).transpose([2, 0, 1]).astype('double') / 255.0
        return source, target


def test():
    from options import TrainOptions
    from tqdm import tqdm
    opt = TrainOptions().parse()
    data_root = '/Users/tony/PycharmProjects/DynTexTrans/data/processed'
    dataset = DynTexTrainDataset(data_root, 'flame')
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize, num_workers=opt.num_workers)
    for data in tqdm(dataloader):
        pass


if __name__ == '__main__':
    test()
