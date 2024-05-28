class Animal:
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species

    def describe(self):
        print(f"{self.name} gyvunas priklauso {self.species} rusiai")


cat = Animal(name="Jokis", species="Katinas")
dog = Animal(name="Dokis", species="Suo")

cat.describe()
dog.describe()
