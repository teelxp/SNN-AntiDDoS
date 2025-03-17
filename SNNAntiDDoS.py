import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.amp as amp
from tqdm import tqdm
import snntorch as snn

# Фиксируем seed для воспроизводимости
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
torch.backends.cudnn.benchmark = True

# Гиперпараметры
config = {
    "file_path": "02-14-2018.csv",  # Проверьте корректность пути к файлу
    "batch_size": 4096,
    "epochs": 5,
    "lr": 0.0005,
    "T_max": 100,
    "hidden_dim1": 150,
    "hidden_dim2": 75,
    "seed": 42,
    "num_workers": 4,  # уменьшено число воркеров для совместимости
}

def load_and_preprocess_data(file_path: str):
    # Загрузка датасета
    data = pd.read_csv(file_path)
    # Очистка заголовков: удаляем пробелы в начале и конце названий столбцов
    data.columns = data.columns.str.strip()
    print("Столбцы:", data.columns.tolist())

    # Проверка наличия необходимых колонок
    if "Label" not in data.columns:
        raise ValueError("Отсутствует колонка Label (целевая метка)")

    # Выводим распределение классов (до обработки)
    print("Распределение классов (до обработки):")
    print(data["Label"].value_counts())

    # Перекодировка меток: Benign → 0, остальные → 1
    data["Label"] = data["Label"].apply(lambda x: 0 if x == "Benign" else 1)

    # Сохраняем оригинальные значения для последующей денормализации iptables правил
    original_ports = data["Dst Port"].copy() if "Dst Port" in data.columns else None
    original_protocols = data["Protocol"].copy() if "Protocol" in data.columns else None

    # Если в датасете присутствует столбец Timestamp, его можно отбросить или использовать для создания дополнительных признаков
    if "Timestamp" in data.columns:
        data = data.drop(columns=["Timestamp"])

    # Кодирование категориальных признаков: например, "Protocol"
    if "Protocol" in data.columns:
        if not pd.api.types.is_numeric_dtype(data["Protocol"]):
            data["Protocol"] = data["Protocol"].astype("category").cat.codes

    # Замена бесконечных значений на NaN и удаление строк с пропущенными значениями
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    # Сброс индексов, чтобы индексы были последовательными
    data.reset_index(drop=True, inplace=True)
    if original_ports is not None:
        original_ports = original_ports.loc[data.index]
    if original_protocols is not None:
        original_protocols = original_protocols.loc[data.index]

    # Определяем список признаков: исключаем целевую метку
    features = [col for col in data.columns if col != "Label"]

    # Приведение числовых признаков к корректному типу (если необходимо)
    data[features] = data[features].apply(pd.to_numeric, errors='coerce')

    # Нормализация числовых признаков
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    # Формирование входных данных и меток
    X = torch.tensor(data[features].values, dtype=torch.float32)
    y = torch.tensor(data["Label"].values, dtype=torch.float32)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, data.index, test_size=0.2, random_state=config["seed"]
    )
    # Также возвращаем оригинальные значения для тестового набора
    if original_ports is not None:
        original_ports = original_ports.iloc[test_idx]
    if original_protocols is not None:
        original_protocols = original_protocols.iloc[test_idx]

    return X_train, X_test, y_train, y_test, data, features, train_idx, test_idx, original_ports, original_protocols

# Оптимизированная модель с использованием SNN
class ModifiedTrafficSNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.2)
        self.lif1 = snn.Leaky(beta=0.85, init_hidden=False)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.2)
        self.lif2 = snn.Leaky(beta=0.85, init_hidden=False)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='linear')

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        mem1 = torch.zeros_like(x)
        spk1, _ = self.lif1(x, mem1)

        x = self.fc2(spk1)
        x = self.bn2(x)
        x = self.dropout2(x)
        mem2 = torch.zeros_like(x)
        spk2, _ = self.lif2(x, mem2)

        out = self.fc3(spk2)
        return out

def generate_iptables_rules(model, X_data, orig_ports, orig_protocols):
    """
    Генерация iptables правил на основе предсказаний модели.
    Для каждого примера используются денормализованные значения 'Dst Port' и 'Protocol'.
    """
    model.eval()
    rules = []
    with torch.no_grad():
        out = model(X_data)
        preds = (torch.sigmoid(out).squeeze() > 0.5).float().cpu().numpy()
    for i, pred in enumerate(preds):
        # Берём оригинальные значения для iptables правила
        dst_port = orig_ports.iloc[i] if orig_ports is not None else "N/A"
        protocol = orig_protocols.iloc[i] if orig_protocols is not None else "N/A"
        action = "ACCEPT" if pred == 1 else "DROP"
        rule = f"iptables -A INPUT -p {protocol} --dport {dst_port} -j {action}"
        rules.append(rule)
    return rules

# Warm-up scheduler
class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]

def train_snn_fn(model, loader, optimizer, scheduler, warmup_scheduler, criterion, scaler, device, epochs,
                 warmup_epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                out = model(X_batch)
                loss = criterion(out.squeeze(), y_batch)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / len(loader):.4f}")

def main():
    X_train, X_test, y_train, y_test, data, features, train_idx, test_idx, orig_ports, orig_protocols = load_and_preprocess_data(
        config["file_path"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используемое устройство:", device)

    # Подготовка DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)

    input_dim = len(features)
    model = ModifiedTrafficSNN(input_dim, config["hidden_dim1"], config["hidden_dim2"], output_dim=1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    warmup_epochs = 5
    warmup_scheduler = WarmUpLR(optimizer, warmup_epochs=warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["T_max"])

    criterion = nn.BCEWithLogitsLoss()
    scaler = amp.GradScaler()

    train_snn_fn(model, train_loader, optimizer, scheduler, warmup_scheduler, criterion, scaler, device,
                 epochs=config["epochs"], warmup_epochs=warmup_epochs)

    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device, non_blocking=True)
        y_test = y_test.to(device, non_blocking=True)
        test_out = model(X_test)
        preds = (torch.sigmoid(test_out).squeeze() > 0.5).float()
        accuracy = (preds == y_test).sum().item() / y_test.size(0)
        f1 = f1_score(y_test.cpu().numpy(), preds.cpu().numpy(), average="binary")
        print(f"Test Accuracy: {accuracy:.2f}")
        print(f"Test F1-score: {f1:.2f}")

    # Генерация правил iptables с денормализованными значениями
    rules = generate_iptables_rules(model, X_test, orig_ports, orig_protocols)
    print("\nПример правил iptables:")
    for r in rules[:5]:
        print(r)

if __name__ == '__main__':
    main()
