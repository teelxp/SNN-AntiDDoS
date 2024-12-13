import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import snntorch as snn

# Путь до файла с данными
file_path = 'network_traffic_dataset.csv'

# Загрузка датасета
traffic_data = pd.read_csv(file_path)

# Предобработка данных
le_protocol = LabelEncoder()
traffic_data["protocol"] = le_protocol.fit_transform(traffic_data["protocol"])
le_action = LabelEncoder()
traffic_data["action"] = le_action.fit_transform(traffic_data["action"])

# Нормализация данных
scaler = MinMaxScaler()
numerical_columns = ["src_port", "dest_port", "packet_size"]
traffic_data[numerical_columns] = scaler.fit_transform(traffic_data[numerical_columns])


# Нормализация IP-адресов
def normalize_ip(ip):
    parts = list(map(int, ip.split('.')))
    return sum([p / (256 ** (i + 1)) for i, p in enumerate(parts)])


traffic_data["src_ip"] = traffic_data["src_ip"].apply(normalize_ip)
traffic_data["dest_ip"] = traffic_data["dest_ip"].apply(normalize_ip)

# Формирование входов и выходов
features = ["src_ip", "dest_ip", "src_port", "dest_port", "protocol", "packet_size"]
X = torch.tensor(traffic_data[features].values, dtype=torch.float32)
y = torch.tensor(traffic_data["action"].values, dtype=torch.float32)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, traffic_data.index, test_size=0.2, random_state=42
)


# Определение нейронной сети
class TrafficSNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=0.9)  # Первый LIF-слой
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        mem1, spk1 = self.lif1(self.fc1(x))  # Первый слой LIF
        mem2 = self.fc2(spk1)  # Второй линейный слой
        return torch.sigmoid(mem2)  # Применяем sigmoid для вывода


# Создание модели
input_dim = len(features)
hidden_dim = 10
output_dim = 1
model = TrafficSNN(input_dim, hidden_dim, output_dim)


# Функция обучения
def train_snn(model, X_train, y_train, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        spikes = model(X_train)  # Прямой проход
        loss = criterion(spikes.squeeze(), y_train)  # Вычисление потерь
        loss.backward(retain_graph=True)  # Указываем retain_graph=True
        optimizer.step()  # Обновление параметров

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


# Обучение модели
train_snn(model, X_train, y_train)

# Оценка на тестовой выборке
with torch.no_grad():
    spikes = model(X_test)  # Прямой проход
    predictions = (spikes.squeeze() > 0.5).float()
    accuracy = (predictions == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy:.2f}")


# Функция для преобразования чисел обратно в формат IPv4
def denormalize_ip(normalized_ip):
    parts = []
    ip_float = normalized_ip * (256 ** 4)  # Восстанавливаем значение из нормализованного формата
    for i in range(4):
        part = int(ip_float // (256 ** (3 - i)))
        parts.append(str(part))
        ip_float -= part * (256 ** (3 - i))
    return ".".join(parts)

def generate_iptables_rules(model, X_data, original_data):
    with torch.no_grad():
        spikes = model(X_data)  # Прямой проход
        predictions = (spikes.squeeze() > 0.5).float()
        rules = []
        for i, pred in enumerate(predictions.numpy()):
            src_ip = denormalize_ip(original_data.iloc[i]["src_ip"])
            dest_ip = denormalize_ip(original_data.iloc[i]["dest_ip"])
            action = "ACCEPT" if pred == 1 else "DROP"
            rule = f"iptables -A INPUT -s {src_ip} -d {dest_ip} -j {action}"
            rules.append(rule)
    return rules


# Генерация правил для тестовой выборки
rules = generate_iptables_rules(model, X_test, traffic_data.iloc[test_idx])
print(rules[:1])  # Показать первые 10 правил
