import os
import random
import torch
import pandas as pd
import networkx as nx
import snntorch as snn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.amp as amp
from tqdm import tqdm

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
    "file_path": "DDoS_dataset.csv",  # Убедитесь, что путь к файлу корректный
    "batch_size": 4096,
    "epochs": 100,
    "lr": 0.0005,
    "T_max": 100,
    "hidden_dim1": 150,
    "hidden_dim2": 75,
    "seed": 42,
    "num_workers": 16,
}

def normalize_ip(ip) -> float:
    """
    Если ip является строкой и содержит точку, нормализуем его.
    Иначе пытаемся привести к float.
    """
    if isinstance(ip, str) and '.' in ip:
        parts = list(map(int, ip.split('.')))
        return sum([p / (256 ** (i + 1)) for i, p in enumerate(parts)])
    try:
        return float(ip)
    except Exception:
        return 0.0

def denormalize_ip(normalized_ip) -> str:
    """
    Если значение меньше 1, считаем, что оно нормализовано и преобразуем обратно в IP-адрес.
    Если значение уже строковое, возвращаем его как есть.
    Иначе – возвращаем строковое представление.
    """
    if isinstance(normalized_ip, str):
        return normalized_ip
    if normalized_ip < 1:
        ip_float = normalized_ip * (256 ** 4)
        parts = []
        for i in range(4):
            part = int(ip_float // (256 ** (3 - i)))
            parts.append(str(part))
            ip_float -= part * (256 ** (3 - i))
        return ".".join(parts)
    return str(normalized_ip)

def load_and_preprocess_data(file_path: str):
    # Загрузка датасета
    data = pd.read_csv(file_path)

    # Вывод списка столбцов для проверки
    # print(data.columns.tolist())

    # Проверяем наличие необходимых колонок
    required_columns = ["Highest Layer", "Transport Layer", "Source IP", "Dest IP",
                        "Source Port", "Dest Port", "Packet Length", "Packets/Time", "target"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Отсутствует колонка {col}")

    # Кодирование категориальных признаков
    for col in ["Highest Layer", "Transport Layer"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Приведение IP-адресов к числовому виду
    data["Source IP"] = data["Source IP"].apply(normalize_ip)
    data["Dest IP"] = data["Dest IP"].apply(normalize_ip)

    # Нормализация числовых признаков
    scaler = MinMaxScaler()
    num_columns = ["Source Port", "Dest Port", "Packet Length", "Packets/Time"]
    data[num_columns] = scaler.fit_transform(data[num_columns])

    # Построение графа для вычисления степеней узлов по IP-адресам
    G = nx.DiGraph()
    unique_ips = pd.concat([data["Source IP"], data["Dest IP"]]).unique()
    G.add_nodes_from(unique_ips)
    for idx, row in data.iterrows():
        G.add_edge(row["Source IP"], row["Dest IP"])
    in_degs = dict(G.in_degree())
    out_degs = dict(G.out_degree())
    data["in_degree"] = data["Source IP"].apply(lambda ip: in_degs.get(ip, 0))
    data["out_degree"] = data["Source IP"].apply(lambda ip: out_degs.get(ip, 0))
    deg_scaler = MinMaxScaler()
    data[["in_degree", "out_degree"]] = deg_scaler.fit_transform(data[["in_degree", "out_degree"]])

    # Формируем список признаков для модели
    features = [
        "Highest Layer", "Transport Layer", "Source IP", "Dest IP",
        "Source Port", "Dest Port", "Packet Length", "Packets/Time",
        "in_degree", "out_degree"
    ]

    X = torch.tensor(data[features].values, dtype=torch.float32)
    y = torch.tensor(data["target"].values, dtype=torch.float32)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, data.index, test_size=0.2, random_state=config["seed"]
    )
    return X_train, X_test, y_train, y_test, data, features, train_idx, test_idx


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

def generate_iptables_rules(model, X_data, original_data):
    model.eval()
    rules = []
    with torch.no_grad():
        out = model(X_data)
        preds = (torch.sigmoid(out).squeeze() > 0.5).float().cpu().numpy()
    for i, pred in enumerate(preds):
        idx = original_data.index[i]
        src_ip = denormalize_ip(original_data.loc[idx, "Source IP"])
        dst_ip = denormalize_ip(original_data.loc[idx, "Dest IP"])
        action = "ACCEPT" if pred == 1 else "DROP"
        rule = f"iptables -A INPUT -s {src_ip} -d {dst_ip} -j {action}"
        rules.append(rule)
    return rules

# Warm-up scheduler
class WarmUpLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]

def train_snn_fn(model, loader, optimizer, scheduler, warmup_scheduler, criterion, scaler, device, epochs, warmup_epochs):
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
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

def main():
    X_train, X_test, y_train, y_test, data, features, train_idx, test_idx = load_and_preprocess_data(config["file_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используемое устройство:", device)

    # Подготовка DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True, persistent_workers=True)

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
        print(f"Test Accuracy: {accuracy:.2f}")

    rules = generate_iptables_rules(model, X_test, data.iloc[test_idx])
    print("\nПример правил iptables:")
    for r in rules[:5]:
        print(r)

if __name__ == '__main__':
    main()
