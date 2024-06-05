#1.1 Sukurkite abstrakčią klasę Shape su šiais abstrakčiais metodais:
		#plotas(): Turėtų grąžinti figūros plotą.
		#perimetras(): Turėtų grąžinti figūros perimetrą.
	#1.2 Sukurkite tris vaikines klases: Apskritimas, Stačiakampis ir Trikampis.
	#1.3 Implementuokite metodus plotas() ir perimetras() kiekvienoje vaikynėje klasėje.
	#1.4 Sukurkite kiekvienos klasės objektus ir iškvieskite ploto() ir perimetro() metodus.

from abc import ABC, abstractmethod
import math

class Shape(ABC):

    @abstractmethod
    def plotas(self):
        pass

    @abstractmethod
    def perimetras(self):
        pass

class Apskritimas(Shape):
    def __init__(self,r: float):
        self.r = r

    def plotas(self):
        plotas = math.pi * self.r **2
        print(f"Apskritimo plotas lygus {plotas:.2f}")

    def perimetras(self):
        perimetras = math.pi * 2 * self.r
        print(f"Apskritimo perimetras lygus {perimetras:.2f}")

class Staciakampis(Shape):

    def __init__(self,a: float, b: float):
        self.a = a
        self.b = b

    def plotas(self):
        plotas = self.a * self.b
        print(f"Staciakampio plotas lygus {plotas:.2f}")

    def perimetras(self):
        perimetras = 2 * (self.a + self.b)
        print(f"Staciakampio perimetras lygus {perimetras:.2f}")

class Trikampis(Shape):

    def __init__(self,a: float, b: float, c:float):
        self.a = a
        self.b = b
        self.c = c

    def plotas(self):
        s = (self.a + self.b + self.c) / 2
        plotas = math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))
        print(f"Trikampio plotas lygus {plotas:.2f}")

    def perimetras(self):
        perimetras = self.a + self.b + self.c
        print(f"Trikampio perimetras lygus {perimetras:.2f}")

apskritimas = Apskritimas(r=1)
staciakampis = Staciakampis(a=3, b=2)
trikampis = Trikampis(3, 4, 5)

apskritimas.plotas()
apskritimas.perimetras()
staciakampis.plotas()
staciakampis.perimetras()
trikampis.plotas()
trikampis.perimetras()
