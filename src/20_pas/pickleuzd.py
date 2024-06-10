

import pickle

class DataSerializer:
    @staticmethod
    def save_to_pkl(obj, filename):
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def read_pkl(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

# Pavyzdžiai kaip naudoti klasę DataSerializer
# Objekto serializavimas į .pkl failą
data_to_save = {'key': 'value'}
DataSerializer.save_to_pkl(data_to_save, 'data.pkl')

# Objekto nuskaitymas iš .pkl failo
loaded_data = DataSerializer.read_pkl('data.pkl')
print(loaded_data)  # Rezultatas: {'key': 'value'}