#3.1 Įdiekite pandas paketą.
#3.2 Sukurkite klasę pavadinimu DataFrameHandler.
#3.3 Implementuokite metodą load_dataframe(file_path), kad įkeltumėte CSV failą į pandas DataFrame.
#3.4 Implementuokite metodus filter_data(condition) ir group_data(column), kad filtruotumėte ir sugrupuotumėte DataFrame.
import pandas as pd

class DataFrameHandler:
    def __init__(self):
        self.data_frame = None

    def load_dataframe(self, file_path):
        self.data_frame = pd.read_csv(file_path)

    def filter_data(self, condition):
        if self.data_frame is not None:
            self.data_frame = self.data_frame[condition]

    def group_data(self, column):
        if self.data_frame is not None:
            self.data_frame = self.data_frame.groupby(column)

df_handler = DataFrameHandler()
df_handler.load_dataframe('example.csv')


df_handler.filter_data(df_handler.data_frame['Column'] > 5)
df_handler.group_data('AnotherColumn')