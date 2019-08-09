import os

import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import albumentations as albu


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


class DynTexNNFTrainDataset(Dataset):
    def __init__(self, data_root, effect):
        self.data_root = os.path.join(data_root, effect)
        self.effect = effect
        self.data_list = sorted(os.listdir(self.data_root), key=lambda x: int(x.replace('.png', '')))
        self.train_shape = cv2.imread(os.path.join(self.data_root, self.data_list[0])).shape[:2]
        self.scaled_shape = (int(self.train_shape[0] * 0.5), int(self.train_shape[1] * 0.8))

        self.target_transforms = albu.Compose([
            albu.RandomSizedCrop(self.scaled_shape, height=self.train_shape[1], width=self.train_shape[0]),
            albu.ElasticTransform(),
            albu.Blur(blur_limit=11, p=0.5),
            albu.RandomBrightnessContrast(p=0.5),
            albu.HorizontalFlip(p=0.5),
        ])
        self.final_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.source_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list) - 1

    def __getitem__(self, item):
        path1 = self.data_list[item]
        path2 = self.data_list[item + 1]
        source_t = cv2.imread(os.path.join(self.data_root, path1), cv2.IMREAD_UNCHANGED)
        source_t1 = cv2.imread(os.path.join(self.data_root, path2), cv2.IMREAD_UNCHANGED)
        
        augmented = self.target_transforms(image=source_t, mask=source_t1)
        target_t, target_t1 = augmented['image'], augmented['mask']
        target_t = self.final_transforms(target_t)
        target_t1 = self.final_transforms(target_t1)
        source_t = self.source_transforms(source_t)
        source_t1 = self.source_transforms(source_t1)
        return source_t, target_t, source_t1, target_t1


class DynTexFigureTrainDataset(Dataset):
    def __init__(self, data_root, effect):
        self.data_root = os.path.join(data_root, effect)
        self.effect = effect
        self.data_list = sorted(os.listdir(self.data_root), key=lambda x: int(x.replace('.png', '')))
        self.train_shape = cv2.imread(os.path.join(self.data_root, self.data_list[0])).shape[:2]
        self.scaled_shape = (int(self.train_shape[0] * 0.5), int(self.train_shape[1] * 0.8))

        self.target_transforms = albu.Compose([
            albu.RandomSizedCrop(self.scaled_shape, height=self.train_shape[1], width=self.train_shape[0]),
            albu.ElasticTransform(),
            albu.Blur(blur_limit=11, p=0.5),
            albu.RandomBrightnessContrast(p=0.5),
            albu.HorizontalFlip(p=0.5),
        ])
        self.final_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.source_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data_list) - 1

    def __getitem__(self, item):
        path1 = self.data_list[item]
        path2 = self.data_list[item + 1]
        source_t = cv2.imread(os.path.join(self.data_root, path1), cv2.IMREAD_UNCHANGED)
        source_t1 = cv2.imread(os.path.join(self.data_root, path2), cv2.IMREAD_UNCHANGED)

        augmented = self.target_transforms(image=source_t, mask=source_t1)
        target_t, target_t1 = augmented['image'], augmented['mask']
        target_t = self.final_transforms(target_t)
        target_t1 = self.final_transforms(target_t1)
        source_t = self.source_transforms(source_t)
        source_t1 = self.source_transforms(source_t1)
        return source_t, target_t, source_t1, target_t1




def test():
    from options import TrainOptions
    from nnf import NNFPredictor, Synthesiser3D
    opt = TrainOptions().parse()
    data_root = '/Users/tony/PycharmProjects/DynTexTrans/data/processed'
    dataset = DynTexNNFTrainDataset(data_root, 'flame')
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize, num_workers=opt.num_workers)
    syner = Synthesiser3D()
    nnfer = NNFPredictor()
    for i, (source_t, target_t, source_t1, target_t1) in enumerate(dataloader):
        source_t.requires_grad_()
        target_t.requires_grad_()
        source_t1.requires_grad_()
        target_t1.requires_grad_()

        nnf = nnfer(source_t, target_t)
        name = os.path.join(data_root, '..', '{}.png'.format(str(i)))
        target_predict = syner(source_t1, nnf)
        cv2.imwrite(name, (target_predict.detach().numpy()[0].transpose(1, 2, 0) * 255).astype('int'))


if __name__ == '__main__':
    test()
