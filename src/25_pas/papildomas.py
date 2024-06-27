import pandas as pd
import numpy as np

# Duomenų generavimas
np.random.seed(42)
data = {
    'Vairuotuoju skaicius': np.random.randint(1, 3, size=100),
    'Atstumas (km)': np.random.uniform(100, 3500, size=100).round(2),
    'Kaina ($)': np.random.uniform(100, 5000, size=100)
}

# Sukuriame DataFrame
df = pd.DataFrame(data)
# Išsaugome duomenis į CSV failą
csv_path = ('uzdstreamlit.csv')
df.to_csv(csv_path, index=False)