from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import config


class KeyDataset(Dataset):
    def __init__(self, transform, mode):
        self.train_imgs = []
        self.valid_imgs = []
        self.test_imgs = []
        self.train_labels = {}
        self.valid_labels = {}
        self.test_labels = {}
        self.transform = transform
        self.mode = mode
        with open(config.data_dir + config.train_dir, 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = config.data_dir + l[1:]
            self.train_imgs.append(img_path)

        with open(config.data_dir + 'clean_test_key_list.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = config.data_dir + 'images/' + l + '.jpg'
            self.test_imgs.append(img_path)

        with open(config.data_dir + 'clean_val_key_list.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = config.data_dir + l[1:]
            self.valid_imgs.append(img_path)

        with open(config.data_dir + config.train_labels_file, 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = config.data_dir + entry[0][1:]
            self.train_labels[img_path] = int(entry[1])

        with open(config.data_dir + config.test_validation_labels_file, 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = config.data_dir + entry[0][1:]
            self.test_labels[img_path] = int(entry[1])

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
        elif self.mode == 'valid':
            img_path = self.valid_imgs[index]
            target = self.test_labels[img_path]
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
        else:
            raise Exception('%s not allowed'.format(self.mode))
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)

        return img, target, img_path

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_imgs)
        elif self.mode == 'valid':
            return len(self.valid_imgs)
        elif self.mode == 'test':
            return len(self.test_imgs)


class KeyDataLoader(object):
    def __init__(self):

        self.transform_train = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size-config.crop),
            # transforms.RandomHorizontalFlip(),

            ##FLIPPING PREPROCESSING
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(45),
            # transforms.RandomAffine(45),
            # transforms.ColorJitter(),

            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),

        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size-config.crop),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

    def run(self):

        train_loader = DataLoader(
            dataset=KeyDataset(
                transform=self.transform_train, mode='train'),
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers)
        valid_loader = DataLoader(
            dataset=KeyDataset(
                transform=self.transform_test, mode='valid'),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers)
        test_loader = DataLoader(
            dataset=KeyDataset(
                transform=self.transform_test, mode='test'),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers)

        return train_loader, valid_loader, test_loader
