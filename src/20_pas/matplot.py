#1.1 Įdiekite paketą matplotlib.
#1.2 Sukurkite klasę pavadinimu DataVisualizer.
#1.3 Panaudokite metodą plot_data(x, y).
#1.4 Įdiekite metodą save_plot(failo pavadinimas), kad išsaugotumėte brėžinį faile.

import matplotlib.pyplot as plt


class DataVisualizer:
    def plot_data(self, x, y):
        plt.plot(x, y)
        plt.xlabel('Dydis label')
        plt.ylabel('Kaina label')
        plt.title('Data Visualization')
        plt.show()

    def save_plot(self, filename):
        plt.savefig(filename)



dv = DataVisualizer()
x_data = [1, 2, 3, 4, 5]
y_data = [9, 3, 1, 3, 9]
dv.plot_data(x_data, y_data)
dv.save_plot('plot.png')
