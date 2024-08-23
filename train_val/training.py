import torch
import numpy as np
import clip
from utils.utils import *
from Models.LateFusion import *
from Models.EarlyFusion import *
from Models.Ensemble import *
from Models.ImageOnly import *
from Models.TextOnly import *
from utils.DataLoaders import *
from Models.Roberta import *
from Models.LP import *
from torch.nn import functional as F
import torch.optim as optim
from train_val.train_val import *


def train(model_type, train_files, val_files, train_text, val_text, train_emo, val_emo, Acc, trial):
    Acc=0
    file_n=model_type+'.pth'

    train_loader, val_loader= get_dataloader(train_files, train_emo, train_text, val_files, val_emo, val_text, model_type)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    criterion = nn.BCELoss()
    if model_type == 'Custom':
        input_dim = 768
        output_dim = 8
        model2 = RoBERTaClassifier(num_labels=output_dim)
        for param in model2.parameters():
            param.requires_grad = True
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=2e-5)
        decoder = Decoder4(input_dim, output_dim).to(device)
    elif model_type == 'LP':
        input_dim = 768
        output_dim = 8
        model2 = CustomRobertaCLIPModel(num_classes=output_dim)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=2e-5)
        decoder = 'None'
    elif model_type == 'Roberta':

        output_dim = 8
        model2 = RoBERTaClassifier(num_labels=output_dim)
        for param in model2.parameters():
            param.requires_grad = True
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=2e-5)
        decoder = 'None'

    elif model_type == 'ImageOnly':
        input_dim = 768
        output_dim = 8
        decoder = Decoder4(input_dim, output_dim).to(device)
        model2='None'
        optimizer2='None'
    elif model_type == 'TextOnly':
        input_dim = 768
        output_dim = 8
        decoder = Decoder5(input_dim, output_dim).to(device)
        model2 = 'None'
        optimizer2 = 'None'
    elif model_type == "LateFusion":
        input_dim1 = np.shape(train_text)[1]
        input_dim2 = 768
        output_dim = 8
        hidden_dim = 256
        decoder = Decoder1(input_dim1, input_dim2, hidden_dim, output_dim).to(device)
        model2 = 'None'
        optimizer2 = 'None'
    elif model_type == "EarlyFusion":
        input_dim = 768*2
        output_dim = 8
        decoder = Decoder2(input_dim, output_dim).to(device)
        model2 = 'None'
        optimizer2 = 'None'
    else:
        input_dim1 = np.shape(train_text)[1]
        input_dim2 = 768
        output_dim = 8
        decoder = Decoder3(input_dim1, input_dim2, output_dim).to(device)
        model2 = 'None'
        optimizer2 = 'None'
    if model_type == 'Roberta' or model_type == 'LP':
        optimizer = 'None'
        scheduler = 'None'
    else:
        for param in decoder.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(decoder.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30000, 60000, 90000], gamma=0.97)

    num_epochs = 50


    for epoch in range(num_epochs):
        total_loss = train_model(model, model2, decoder, device, train_loader, optimizer, optimizer2, scheduler, criterion, model_type)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}")

        total_val_loss, hamming = validation(model, model2, decoder, device, val_loader, criterion, model_type)

        print(
            f"Validation Loss: {total_val_loss / len(val_loader):.4f}, Validation F1 score (weighted): {hamming * 100:.4f}%")
        file_loc=file_n
        rb='Roberta.pth'
        rb2 = 'Roberta2.pth'
        if Acc < hamming:
            if model_type == 'Roberta' or model_type =='LP':
                dummy =0
            else:
                torch.save(decoder.state_dict(), file_loc)

            if model_type == 'Custom':
                torch.save(model2.state_dict(), rb)
            elif model_type == 'Roberta':
                torch.save(model2.state_dict(), rb2)
            elif model_type == 'LP':
                torch.save(model2.state_dict(), 'LP.pth')
            Acc = hamming



    return Acc
