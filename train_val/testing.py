from train_val.train_val import *
from utils.utils import *
from transformers import RobertaTokenizer, RobertaModel
import torch
import torchvision
from utils.DataLoaders import *
import clip
import torch.nn as nn
import torch.optim as optim
from Models.LateFusion import *
from Models.EarlyFusion import *
from Models.Ensemble import *
from Models.ImageOnly import *
from Models.TextOnly import *
from Models.Roberta import *
from Models.LP import *


def testing(model_type, test_files, test_text, test_emo, trial):
    file_n=model_type+'.pth'

    file_loc =  file_n
    rb = 'Roberta.pth'
    rb2 = 'Roberta2.pth'
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    root = '/home/purna/PycharmProjects/EmotionClassification'
    path1=os.path.join(root, 'images_final')
    if model_type == 'Custom' or model_type == 'Roberta' or model_type == 'LP':
        max_len = 128
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        test_dataset = CustomDataset(path1, test_files, test_text, test_emo, tokenizer, max_len, test_transform)
    else:
        test_dataset = Hist(path1, test_files, test_emo, test_text,
                        transforms=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=8,
                                              shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    if model_type == 'Custom':
        input_dim = 768
        output_dim = 8
        model2 = RoBERTaClassifier(num_labels=output_dim)
        decoder = Decoder4(input_dim, output_dim).to(device)
        model2.load_state_dict(torch.load(rb, map_location=device))
    elif model_type == 'LP':
        input_dim = 768
        output_dim = 20
        model2 = CustomRobertaCLIPModel(num_classes=output_dim)
        model2.load_state_dict(torch.load('LP.pth', map_location=device))
    elif model_type == 'Roberta':
        output_dim = 20
        model2 = RoBERTaClassifier(num_labels=output_dim)
        model2.load_state_dict(torch.load(rb2, map_location=device))
    elif model_type == 'ImageOnly':
        input_dim = 768
        output_dim = 20
        decoder = Decoder4(input_dim, output_dim).to(device)
        model2='None'
    elif model_type == 'TextOnly':
        input_dim = 768
        output_dim = 20
        decoder = Decoder5(input_dim, output_dim).to(device)
        model2 = 'None'
    elif model_type == "LateFusion":
        input_dim1 = np.shape(test_text)[1]
        input_dim2 = 768
        output_dim = 20
        hidden_dim = 256
        decoder = Decoder1(input_dim1, input_dim2, hidden_dim, output_dim).to(device)
        model2 = 'None'
    elif model_type == "EarlyFusion":
        input_dim = 768*2
        output_dim = 20
        decoder = Decoder2(input_dim, output_dim).to(device)
        model2 = 'None'
    else:
        input_dim1 = np.shape(test_text)[1]
        input_dim2 = 768
        output_dim = 20
        decoder = Decoder3(input_dim1, input_dim2, output_dim).to(device)
        model2 = 'None'

    if model_type == 'Roberta':
        decoder='None'
        model='None'
    elif model_type == 'LP':
        decoder='None'
    else:
        decoder.load_state_dict(torch.load(file_loc, map_location=device))
        decoder = decoder.to(device)


    y_pred, y , acc, y_score= test(model, model2, decoder, device, test_loader, model_type)

    return y_pred, y, acc, y_score