from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch
from PIL import Image

class ClassificationDataset:
    def __init__(self,name,x_train,x_test,y_train,y_test,num_classes,input_shape):
        self.name=name
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.num_classes=num_classes
        self.input_shape=input_shape

class ImageDataset(Dataset):


    def __init__(self, x,y,rotation=None):

        self.x=x
        self.y=y
        mu = x.mean(axis=(0, 1, 2))/255
        std = x.std(axis=(0, 1, 2))/255
        transformations=[transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(mu, std),

                         ]

        if rotation:
            transformations.insert(1,transforms.RandomRotation(180,resample=Image.BILINEAR))
        else:
            pass

        self.transform=transforms.Compose(transformations)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        image =self.x[idx,:,:,:]
        image = self.transform(image)
        target=self.y[idx,:].argmax()
        return (image,target)

    def get_batch(self,idx):
        if isinstance(idx,int):
            idx=[idx]
        images = []
        for i in idx:
            image= self.transform(self.x[i, :, :, :])
            images.append(image)
        y = self.y[idx, :].argmax(axis=1)
        x= torch.stack(images,0)
        return x,y

def get_data_generator(x,y,batch_size):
    image_dataset=ImageDataset(x,y)
    dataset=DataLoader(image_dataset,batch_size=batch_size,shuffle=True,num_workers=1)
    image_rotated_dataset = ImageDataset(x, y, rotation=180)
    rotated_dataset = DataLoader(image_rotated_dataset , batch_size=batch_size, shuffle=True, num_workers=1)

    return dataset,rotated_dataset