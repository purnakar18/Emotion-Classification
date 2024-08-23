from sklearn.metrics import f1_score
import torchvision.transforms as T
import numpy as np
import torch


def train_model(model, model2, decoder, device, train_loader, optimizer, optimizer2, scheduler, criterion, model_type):
    if model_type == 'Roberta':
        model = 'None'
        decoder = 'None'
    elif model_type == 'LP':
        decoder = 'None'
    else:
        model = model.to(device)
        decoder = decoder.to(device)
        decoder.train()

    total_loss = 0.0
    if model_type =='Custom' or model_type == 'Roberta' or model_type == 'LP':

        model2 = model2.to(device)
        model2.train()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            images = batch['images'].to(device)
            if model_type == 'Custom':
                optimizer.zero_grad()
            optimizer2.zero_grad()
            if model_type != 'LP':
                outputs1 = model2(input_ids, attention_mask)
            if model_type == 'Custom':
                with torch.no_grad():
                    image_features = model.encode_image(images)
                image_features = image_features.to(torch.float32)
                outputs2 = decoder(image_features)

                outputs = (3 * outputs1 + 1*outputs2) / 4
            elif model_type == 'LP':
                with torch.no_grad():
                    image_features = model.encode_image(images)
                image_features = image_features.to(torch.float32)
                outputs= model2(input_ids, attention_mask, image_features)
            else:
                outputs = outputs1
            loss = criterion(outputs, labels)
            loss.backward()
            if model_type == 'Custom':
                optimizer.step()
            optimizer2.step()

            total_loss += loss.item()


    else:
        decoder.train()
        for images, targets, text in train_loader:
            images, targets, text = images.to(device), targets.float().to(device), text.float().to(device)
            with torch.no_grad():
                image_features = model.encode_image(images)
            image_features = image_features.to(torch.float32)
            if model_type == 'ImageOnly':
                outputs = decoder(image_features)
            elif model_type == 'TextOnly':
                outputs = decoder(text)
            elif model_type == 'EarlyFusion':
                image_features = image_features.cpu()
                text = text.cpu()
                features = np.concatenate((image_features, text), axis=1)
                features = torch.from_numpy(features).float().to(device)
                outputs = decoder(features)
            else:
                outputs = decoder(text, image_features)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

    return total_loss


def validation(model, model2, decoder, device, val_loader, criterion, model_type):
    if model_type == 'Roberta':
        model = 'None'
        decoder = 'None'
    elif model_type == 'LP':
        decoder = 'None'
    else:
        model = model.to(device)
        decoder = decoder.to(device)
        decoder.eval()
    y_true_val = []
    y_pred_val = []
    y = []
    y_pred = []

    total_loss = 0
    if model_type == 'Custom' or model_type == 'Roberta' or model_type == 'LP':
        model2 = model2.to(device)
        model2.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                images = batch['images'].to(device)
                if model_type != 'LP':
                    outputs1 = model2(input_ids, attention_mask)
                if model_type == 'Custom':
                    with torch.no_grad():
                        image_features = model.encode_image(images)
                    image_features = image_features.to(torch.float32)
                    outputs2 = decoder(image_features)

                    outputs = (3 * outputs1 + 1*outputs2) / 4
                elif model_type == 'LP':
                    with torch.no_grad():
                        image_features = model.encode_image(images)
                    image_features = image_features.to(torch.float32)
                    outputs = model2(input_ids, attention_mask, image_features)
                else:
                    outputs = outputs1

                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = (outputs > 0.5).float()
                y_pred_val.extend(preds.cpu().numpy())
                y_true_val.extend(labels.cpu().numpy())

            y = np.array(y_true_val)
            y = np.reshape(y, (-1, 8))

            y_pred = np.array(y_pred_val)
            y_pred = np.reshape(y_pred, (-1, 8))

        hamming = f1_score(y, y_pred, average='weighted')
    else:

        with torch.no_grad():
            for images, targets, text in val_loader:
                images = images.to(device)
                targets = targets.float().to(device)
                text = text.float().to(device)

                with torch.no_grad():
                    image_features = model.encode_image(images)
                image_features = image_features.to(torch.float32)
                if model_type == 'ImageOnly':
                    outputs = decoder(image_features)
                elif model_type == 'TextOnly':
                    outputs = decoder(text)
                elif model_type == 'EarlyFusion':
                    image_features = image_features.cpu()
                    text = text.cpu()
                    features = np.concatenate((image_features, text), axis=1)
                    features = torch.from_numpy(features).float().to(device)
                    outputs = decoder(features)
                else:
                    outputs = decoder(text, image_features)

                loss = criterion(outputs, targets)

                pred = (outputs > 0.5).float()
                total_loss += loss.item()

                y.extend(targets.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())


            y = np.array(y)
            y = np.reshape(y, (-1, 8))

            y_pred = np.array(y_pred)
            y_pred = np.reshape(y_pred, (-1, 8))

        hamming = f1_score(y, y_pred, average='weighted')

    return total_loss, hamming


def test(model, model2, decoder, device, test_loader, model_type):
    if model_type == 'Roberta':
        model = 'None'
        decoder = 'None'
    elif model_type == 'LP':
        decoder = 'None'
    else:
        model = model.to(device)
        decoder = decoder.to(device)
        decoder.eval()

    y_true_val = []
    y_pred_val = []
    y = []
    y_pred = []
    y_s=[]
    y_score=[]
    total_loss = 0
    if model_type == 'Custom' or model_type == 'Roberta' or model_type == 'LP':
        model2 = model2.to(device)
        model2.eval()
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                images = batch['images'].to(device)
                if model_type != 'LP':
                    outputs1 = model2(input_ids, attention_mask)
                if model_type == 'Custom':
                    with torch.no_grad():
                        image_features = model.encode_image(images)
                    image_features = image_features.to(torch.float32)
                    outputs2 = decoder(image_features)

                    outputs = (3 * outputs1 + 1*outputs2) / 4
                elif model_type == 'LP':
                    with torch.no_grad():
                        image_features = model.encode_image(images)
                    image_features = image_features.to(torch.float32)
                    outputs = model2(input_ids, attention_mask, image_features)
                else:
                    outputs = outputs1

                preds = (outputs > 0.5).float()
                y_pred_val.extend(preds.cpu().numpy())
                y_true_val.extend(labels.cpu().numpy())
                y_s.extend(outputs.cpu().numpy())

            y = np.array(y_true_val)
            y = np.reshape(y, (-1, 8))

            y_pred = np.array(y_pred_val)
            y_pred = np.reshape(y_pred, (-1, 8))
            y_score = np.reshape(y_s, (-1, 8))

        hamming = f1_score(y, y_pred, average='weighted')
    else:
        with torch.no_grad():
            for images, targets, text in test_loader:
                images = images.to(device)
                targets = targets.float().to(device)
                text = text.float().to(device) # Convert targets to float and move to device

                with torch.no_grad():
                    image_features = model.encode_image(images)
                image_features = image_features.to(torch.float32)
                if model_type == 'ImageOnly':
                    outputs = decoder(image_features)
                elif model_type == 'TextOnly':
                    outputs = decoder(text)
                elif model_type == 'EarlyFusion':
                    image_features = image_features.cpu()
                    text = text.cpu()
                    features = np.concatenate((image_features, text), axis=1)
                    features = torch.from_numpy(features).float().to(device)
                    outputs = decoder(features)
                else:
                    outputs = decoder(text, image_features)




                pred = (outputs > 0.5).float()


                y.extend(targets.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())


            y = np.array(y)
            y = np.reshape(y, (-1, 8))
            y_pred = np.array(y_pred)
            y_pred = np.reshape(y_pred, (-1, 8))

        hamming = f1_score(y, y_pred, average='weighted')

    return y_pred, y, hamming, y_score


