#2. Sukurkite bazinę klasę Vehicle su atributais marke ir model
# ir metodu description(), kuris juos atspausdina.
# Sukurkite išvestinę klasę Car, kuri priima atributą year
# ir naudoja super(), kad iškviestų bazinės klasės metodą savo aprašymo metode.

class Vehicle:

    def __init__(self,marke: str,model: str):
        self.marke = marke
        self.model = model

    def description(self):
        print(f'Marke: {self.marke}')
        print(f'Modelis: {self.model}')

class Car(Vehicle):

    def __init__(self, marke: str, model: str,year: int):
        super().__init__(marke, model)
        self.year = year

    def description(self):
        super().description()
        print(f'Metai: {self.year}')

car = Car('Volvo','V90',year=2015)
car.description()