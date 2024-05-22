class Car:
    def __init__(self,marke: str,spalva: str,modelis: str):
        self.marke = marke
        self.spalva = spalva
        self.modelis = modelis

    def info(self) -> str:
        return f"Automobilis {self.marke} {self.modelis} yra {self.spalva}"

    def start_engine(self) -> str:
        return f"Dabar veikia {self.marke} {self.modelis} variklis"


car1 = Car(marke='Toyota',spalva='baltas',modelis='Corola')
car2 = Car(marke='Mazda',spalva='raudonas',modelis='MX-5')

print(car1.info())
print(car2.info())
print(car1.start_engine())
print(car2.start_engine())

