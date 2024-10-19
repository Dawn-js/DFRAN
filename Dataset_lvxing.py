import os
from torch.utils import data
from torchvision.transforms import InterpolationMode
from PIL import Image
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,ToTensor)

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class MVSA_Single():
    def __init__(self, txt_path):
        fh = open(txt_path, 'r', encoding='utf-8')
        self.imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()  # 以空格进行split
            name = words[1]
            label = int(words[0])
            text = ''
            for p in words[2:]:
                text += p
                text += ' '  # 先加上空格还原最初的文本，放到bert中会被处理掉
            text = text.rstrip()  # 把末尾的空格去掉
            self.imgs.append((name, label, text))

    def __getitem__(self, index):
        name_path, emo_label, text = self.imgs[index]
        # emo_label = emo_label - 1
        if emo_label<3:
            emo_label = 0
        elif emo_label>3:
            emo_label = 2
        else:
            emo_label = 1

            
        name_path = name_path.strip('\'"')
        # 按逗号分隔每个网址
        image_urls = [url.strip() for url in name_path.split(',')]
        # 只保留第一张图片的URL
        image_url = image_urls[0]
        base_path = '/data_home/home/sunbo1/BIT-Assistant_Network-UnComplete_Data/lvxing_dataset/img'
        image1 = image_url.split('/')[-1]
        name_path = os.path.join(base_path, image1)
        image = _transform(n_px=224)(Image.open(name_path))
        return image, text, emo_label

    def __len__(self):
        return len(self.imgs)

train_path = '/data_home/home/sunbo1/BIT-Assistant_Network-UnComplete_Data/lvxing_dataset/train.txt'
valid_path = '/data_home/home/sunbo1/BIT-Assistant_Network-UnComplete_Data/lvxing_dataset/val.txt'
test_path = '/data_home/home/sunbo1/BIT-Assistant_Network-UnComplete_Data/lvxing_dataset/test.txt'
            

train_d = MVSA_Single(train_path)
test_d = MVSA_Single(test_path)


def get_dataset(batch_size=16):
     train_data = data.DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=4)
     test_data = data.DataLoader(test_d, batch_size=batch_size, shuffle=False, num_workers=4)
     return train_data,test_data

if __name__ == '__main__':
    # 设置相关参数
    batch_size = 32
    num_workers = 4
    pin_memory = True
    drop_last = False

    # 获取数据集
    train_data, valid_data,test_data = get_dataset(batch_size)

    for epoch in range(1):
        for i, data in enumerate(valid_data):
            images, texts,label = data
            print(images.shape)
