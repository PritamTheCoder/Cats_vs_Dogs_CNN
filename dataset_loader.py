from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

dataset = ImageFolder("data/train")
trainData, testData, trainLabel, testLabel = train_test_split(
    dataset.imgs, dataset.targets, test_size=0.2, random_state=0)

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = self.checkChannel(dataset)  # filter only RGB
        self.transform = transform

    def checkChannel(self, dataset):
        datasetRGB = []
        for path, label in dataset:
            if Image.open(path).getbands() == ('R', 'G', 'B'):
                datasetRGB.append((path, label))
        return datasetRGB

    def getResizedImage(self, item):
        image = Image.open(self.dataset[item][0])
        width, height = image.size
        if width > height:
            left = (width - height) // 2
            top = 0
            right = left + height
            bottom = height
        elif height > width:
            left = 0
            top = (height - width) // 2
            right = width
            bottom = top + width
        else:
            left, top, right, bottom = 0, 0, width, height

        return image.crop((left, top, right, bottom))

    def __getitem__(self, item):
        image = self.getResizedImage(item)  # now using cropped image
        if transform is not None:
            return self.transform(image), self.dataset[item][1]
        return image, self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)

