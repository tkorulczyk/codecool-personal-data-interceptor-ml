import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Scikit-learn metrics imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,jaccard_score, roc_auc_score, matthews_corrcoef


def compute_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate and return various evaluation metrics for given ground truth labels and predicted labels.

    Parameters:
    y_true (array-like): Ground truth labels.
    y_pred (array-like): Predicted labels.
    average (str, optional): The averaging method to be used for calculating precision, recall, and F1-score.
                              Defaults to 'weighted'.

    Returns:
    tuple: A tuple containing the following metrics (in order): accuracy, precision, recall, F1-score,
           Jaccard score, ROC AUC score, and Matthews correlation coefficient.
    """

    # Convert one-hot encoded format to class labels if necessary
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred, average=average), 3)
    recall = round(recall_score(y_true, y_pred, average=average), 3)
    f1 = round(f1_score(y_true, y_pred, average=average), 3)
    jaccard = round(jaccard_score(y_true, y_pred, average=average), 3)

    if len(np.unique(y_true)) > 1:
        roc_auc = round(roc_auc_score(y_true, y_pred), 3)
    else:
        roc_auc = "Undefined"

    matthews_corr = round(matthews_corrcoef(y_true, y_pred), 3)
    return accuracy, precision, recall, f1, jaccard, roc_auc, matthews_corr

class NLP_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(100, 64)
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, 3)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x


class Vect_Word_Dataset(Dataset):
    def __init__(self, words_vect, labels):
        self.words = words_vect
        self.labels = labels

    def __getitem__(self, idx):
        vect = self.words[idx]
        label = self.labels[idx]

        return vect, label

    def __len__(self):
        return len(self.words)


def get_dataloader(vect_words,
                   labels,
                   batch_size=32,
                   shuffle=True,
                   num_workers=2,
                   generator=None):
    dataset = Vect_Word_Dataset(vect_words, labels)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            generator=generator)
    return dataloader


def train_test_pt_model(model,
                        train_loader,
                        test_loader,
                        criterion,
                        optimizer,
                        epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    train_accs = []
    epochs_loop = []

    test_losses = []
    test_epoch_acc = []
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, leave=False)
        batch_losses = []
        test_batch_losses = []
        epochs_loop.append(epoch + 1)
        acc, count, test_acc, test_count = 0, 0, 0, 0
        model.train()
        for inputs, target in progress_bar:
            inputs.view(inputs.shape[0], -1)
            inputs = inputs.to(device)
            target = target.to(device)

            model.zero_grad()
            output = model(inputs)
            loss = criterion(output, target.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            progress_bar.set_description(f'Epoch: {epoch + 1} | Loss: {loss.item():.3f}')
            batch_losses.append(loss.item())

            pred = (torch.sigmoid(output) > 0.5).float()
            acc += (pred == target).sum()
            count += len(target)

        model.eval()
        with torch.inference_mode():
            for test_input, test_target in test_loader:
                inputs.view(inputs.shape[0], -1)
                test_input = test_input.to(device)
                test_target = test_target.to(device)

                test_out = model(test_input).squeeze()
                test_pred = torch.round(torch.sigmoid(test_out))
                test_loss = criterion(test_out, test_target.float())
                test_batch_losses.append(test_loss.item())
                test_acc += (test_pred == test_target).sum()
                test_count += len(test_target)

        epoch_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(epoch_loss)
        epoch_acc = (acc / count) * 100
        train_accs.append(epoch_acc.item())

        test_epoch_loss = sum(test_batch_losses) / len(test_batch_losses)
        test_losses.append(test_epoch_loss)
        epoch_test_acc = (test_acc / test_count) * 100
        test_epoch_acc.append(epoch_test_acc.item())

        if not epoch % 10:
            train_metrics = compute_metrics(target.cpu().numpy(), pred.cpu().numpy())
            test_metrics = compute_metrics(test_target.cpu().numpy(), test_pred.cpu().numpy())

            tqdm.write(f'Epoch #{epoch + 1}\tTrain Loss: {epoch_loss:.5f} \tTrain ACC: {epoch_acc:.5f}')
            tqdm.write(f'Epoch #{epoch + 1}\tTest Loss: {test_epoch_loss:.5f} \tTest ACC: {epoch_test_acc:.5f}\n')

            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Jaccard', 'ROC AUC', 'Matthews Corr']
            metrics_dict = {'Metric': metric_names, 'Train': train_metrics, 'Test': test_metrics}

            metrics_df = pd.DataFrame(metrics_dict)
            print(metrics_df)

            tqdm.write('\n')

        return train_losses, train_accs, test_losses, test_epoch_acc, epochs_loop


# 5. Encoding of categorical variables
def one_hot_encoding(df, columns):
    df_encoded = pd.get_dummies(df, columns=columns, prefix=columns, prefix_sep='_')
    df_encoded = df_encoded.astype(int)
    return df_encoded

# 5. Encoding of categorical variables
def one_hot_encoding(df, columns):
    df_encoded = pd.get_dummies(df, columns=columns, prefix=columns, prefix_sep='_')
    for column in columns:
        for col in df_encoded.columns:
            if col.startswith(column):
                df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded


# def val_pt_model(testing_pic):
#     if testing_pic is None:
#         return "Nie znaleziono obrazka"
#     if not testing_pic.any():
#         return "Obrazek jest pusty"
#     else:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         loaded_model.to(device)
#         loaded_model.eval()
#         with torch.inference_mode():
#             inputs = pic_transformation(testing_pic)
#             inputs = np.expand_dims(inputs, axis=(0,1))
#             inputs = torch.Tensor(inputs)
#             inputs = inputs.to(device=device)
#             out = loaded_model(inputs)
#             pred_proba = torch.sigmoid(out)
#             pred = torch.argmax(pred_proba, dim=1)
#             if pred:
#                 pred_label = 'Predator'
#                 pred_proba = float(pred_proba.squeeze())
#             else:
#                 pred_label = 'Alien'
#                 pred_proba = 1- float(pred_proba.squeeze())
#             return pred_label, pred_proba*100

def main():
    df = pd.read_csv('names.csv', sep=';')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    df = df.drop(columns='płeć')

    # One hot encoding
    df = one_hot_encoding(df, ['rodzaj'])
    X = df['dana'].str.lower()
    y = df.drop(columns='dana')

    tfidf = TfidfVectorizer()
    X = X.fillna('')
    X_tfidf = tfidf.fit_transform(X)
    # print(X_tfidf)

    svd = TruncatedSVD(n_components=100)
    X_tfidf_svd = svd.fit_transform(X_tfidf)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf_svd, y, test_size=0.2, random_state=42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        generator = torch.Generator(device).manual_seed(42)
    else:
        generator = torch.default_generator.manual_seed(42)

    train_tensor_label = torch.Tensor(np.array(y_train)).float()
    train_X_tensor = torch.Tensor(np.array(X_train))
    dataloader = get_dataloader(train_X_tensor,
                                train_tensor_label,
                                batch_size=32,
                                shuffle=True,
                                num_workers=4,
                                generator=generator)

    test_tensor_label = torch.Tensor(np.array(y_test)).float()
    test_tensor_vect = torch.Tensor(np.array(X_test))
    test_loader = get_dataloader(test_tensor_vect,
                                 test_tensor_label,
                                 batch_size=32,
                                 shuffle=True,
                                 num_workers=4,
                                 generator=generator)

    pt_model = NLP_Net()
    pt_model.to(device) # Przenieś model na GPU, jeśli dostępne
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(pt_model.parameters(), lr=0.001)

    train_losses, train_accs, test_losses, test_epoch_acc, _ = train_test_pt_model(pt_model,
                                                                                   dataloader,
                                                                                   test_loader,
                                                                                   criterion,
                                                                                   optimizer,
                                                                                   epochs=30)

if __name__ == '__main__':
    main()