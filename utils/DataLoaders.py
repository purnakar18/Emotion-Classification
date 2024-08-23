import torch
import torchvision
from PIL import Image
import torch.utils.data as data
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, BertForMultipleChoice

class Hist(data.Dataset):
    def __init__(self, root, files, target, text, transforms=None):
        self.root = root
        self.files = files
        self.target = target
        self.text = text
        self.transforms = transforms

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root, self.files[item])).convert(mode="RGB")
        target_ = self.target[item]
        text_ = self.text[item]
        if self.transforms:
            image = self.transforms(image)
        return image, target_, text_

    def __len__(self):
        return len(self.files)

class CustomDataset(Dataset):
    def __init__(self, root, files, texts, labels, tokenizer, max_len, transforms=None):
        self.root = root
        self.files = files
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transforms = transforms

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.files[idx])).convert(mode="RGB")
        text = str(self.texts[idx])
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float),
            'images':  image
        }
def get_dataloader(train_files, train_emo, train_text, val_files, val_emo, val_text, model_type):



    batch_size_train = 12  #
    batch_size_test = 5  #

    # define how image transformed
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    root = '/home/purna/PycharmProjects/EmotionClassification'
    path1 = os.path.join(root, 'images_final')

    if model_type == 'Custom' or model_type == 'Roberta' or model_type == 'LP':
        max_len = 128
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        train_dataset = CustomDataset(path1, train_files, train_text, train_emo, tokenizer, max_len, image_transform)
        val_dataset = CustomDataset(path1, val_files, val_text, val_emo, tokenizer, max_len, image_transform)
    else:
        train_dataset = Hist(path1, train_files, train_emo, train_text, transforms=image_transform)
        val_dataset = Hist(path1, val_files, val_emo, val_text, transforms=image_transform)

    # data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size_train,
                                               shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size_test,
                                             shuffle=False, num_workers=2)

    return train_loader, val_loader