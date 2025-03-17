import numpy as np
import pandas as pd

num_rows = 30000

def random_ip():
    """Генерирует случайный IP-адрес в формате A.B.C.D"""
    return ".".join(str(np.random.randint(0, 256)) for _ in range(4))

# Генерируем случайные данные для каждого столбца
src_ips = [random_ip() for _ in range(num_rows)]
dest_ips = [random_ip() for _ in range(num_rows)]
src_ports = np.random.randint(1, 65536, size=num_rows)
dest_ports = np.random.randint(1, 65536, size=num_rows)
protocols = np.random.choice(["TCP", "UDP", "ICMP"], size=num_rows)
packet_sizes = np.random.randint(64, 1501, size=num_rows)  # Например, от 64 до 1500 байт

# Половина строк ALLOW, половина BLOCK
actions = np.array(["ALLOW"] * (num_rows // 2) + ["BLOCK"] * (num_rows - num_rows // 2))
# Перемешиваем массив, чтобы ALLOW/BLOCK шли вперемешку
np.random.shuffle(actions)

# Формируем DataFrame
df = pd.DataFrame({
    "src_ip": src_ips,
    "dest_ip": dest_ips,
    "src_port": src_ports,
    "dest_port": dest_ports,
    "protocol": protocols,
    "packet_size": packet_sizes,
    "action": actions
})

# Чтобы индекс начинался с 1, как на скриншоте
df.index += 1

# Сохраняем в CSV
df.to_csv("network_traffic_dataset.csv", index_label="id")

print("Файл 'network_traffic_dataset.csv' создан со структурой, аналогичной скриншоту, и содержит 30 000 строк.")
