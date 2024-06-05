#2.1 Sukurkite abstrakčią klasę Employee su šiais abstrakčiais metodais:
		#calculate_pay(): Turėtų grąžinti darbuotojo darbo užmokestį.
		#get_role(): Turėtų grąžinti darbuotojo vaidmenį.
	#2.2 Sukurkite tris vaikines klases: Vadybininkas, Inžinierius ir Stažuotojas.
	#2.3 Implementuokite metodus calculate_pay() ir get_role() kiekvienoje vaikynėje klasėje.
	#2.4 Sukurkite objektus ir iškvieskite metodus calculate_pay() ir get_role().

from abc import ABC, abstractmethod

class Employee(ABC):

    @abstractmethod
    def calculate_pay(self):
        pass

    @abstractmethod
    def get_role(self):
        pass

class Vadybininkas(Employee):
    def __init__(self, alga):
        self.alga = alga

    def calculate_pay(self):
        print(f"Alga: {self.alga}")

    def get_role(self):
        print("Vadybininkas")


class Inzinierius(Employee):
    def __init__(self, val: int, tarifas: float):
        self.val = val
        self.tarifas = tarifas

    def calculate_pay(self):
        alga = self.val * self.tarifas
        print(f"Alga: {alga:.2f}")

    def get_role(self):
        print("Inzinierius")


class Stazuotojas(Employee):
    def __init__(self, stipendija: int):
        self.stipendija = stipendija

    def calculate_pay(self):
        alga = self.stipendija * 4
        print(f"Menesio alga: {alga}")

    def get_role(self):
        print("Stazuotojas")


vadybininkas = Vadybininkas(alga=5000)
inzinierius = Inzinierius(240, 12.25)
stazuotojas = Stazuotojas(stipendija=300)

vadybininkas.get_role()
vadybininkas.calculate_pay()
inzinierius.get_role()
inzinierius.calculate_pay()
stazuotojas.get_role()
stazuotojas.calculate_pay()
