import pandas as pd
import torch
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    median_absolute_error,
)

from regession_model import RegressionModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels.float())
            # print("output", outputs)
            # print("label", labels.float())
            # print("loss", loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(total_loss)


# Testing function
def test_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            outputs = model(embeddings).squeeze()
            predictions.extend(outputs.tolist())
            true_labels.extend(labels.tolist())

    mse = mean_squared_error(true_labels, predictions)
    print(f"Mean Squared Error: {mse:.4f}")
    return predictions, true_labels


def get_data_for_training(data, field, label_field="label"):
    embeddings = list()
    labels = list()
    completion_ids = list()
    for idx, row in data.iterrows():
        labels.append(row[label_field])
        embeddings.append(row[field])
        completion_ids.append(row["completion_id"])
    np_emb = np.array(embeddings)
    embeddings = torch.tensor(np_emb, device=device).to(torch.float)
    labels = torch.tensor(labels, device=device)
    print(embeddings.shape, labels.shape)
    return embeddings, labels, completion_ids


def training_classify(train_df, test_df, field, label_field, lang="CPP"):
    X_train, y_train, _ = get_data_for_training(train_df, field, label_field)
    X_test, y_test, completion_ids = get_data_for_training(test_df, field, label_field)
    embedding_dim = X_train.shape[-1]
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.01

    # Split data into train and test sets

    # Data loaders
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    mses = list()
    maes = list()
    rmses = list()
    r2s = list()

    y_pred, y_true = list(), list()

    y_pred_out, y_true_out = list(), list()
    min_mse = 999999

    for i in range(5):
        model = RegressionModel(embedding_dim)
        model = model.to(device)
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_model(model, train_loader, criterion, optimizer, num_epochs)
        y_pred, y_true = test_model(model, test_loader)
        tmp_mae = median_absolute_error(y_true, y_pred)
        tmp_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        tmp_mse = mean_squared_error(y_true, y_pred)
        tmp_r2 = r2_score(y_true, y_pred)

        maes.append(tmp_mae)
        rmses.append(tmp_rmse)
        mses.append(tmp_mse)
        r2s.append(tmp_r2)

        if tmp_mse < min_mse:
            min_mse = tmp_mse
            y_pred_out, y_true_out = y_pred, y_true
            break  # CHANGE ME

    max_index = mses.index(min(mses))
    print(f"mean: {sum(mses)/5}\t{sum(rmses)/5}\t{sum(maes)/5}\t{sum(r2s)/5}")
    print(
        f"max: {mses[max_index]}\t{rmses[max_index]}\t{maes[max_index]}\t{r2s[max_index]}\t"
    )

    result = list()
    test_df = test_df.reset_index()
    mapping_code = dict()
    mapping_generation = dict()
    for idx, row in test_df.iterrows():
        mapping_code[row["completion_id"]] = row["extracted_code"]
        mapping_generation[row["completion_id"]] = row["generation"]

    print(len(y_true), len(y_pred), len(completion_ids))
    for idx, id in enumerate(completion_ids):
        result.append(
            {
                "ref": y_true_out[idx],
                "predict": y_pred_out[idx],
                "id": id,
                "extracted_code": mapping_code[id],
                "generation": mapping_generation[id],
            }
        )
    import time

    dfs = pd.DataFrame(result)
    dfs.to_csv(f"result_base_{lang}_{field}_{time.time()}.csv", index=False)
