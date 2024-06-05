	#1.1 Sukurkite abstrakčią klasę Shape su šiais abstrakčiais metodais:
		#plotas(): Turėtų grąžinti figūros plotą.
		#perimetras(): Turėtų grąžinti figūros perimetrą.
	#1.2 Sukurkite tris vaikines klases: Apskritimas, Stačiakampis ir Trikampis.
	#1.3 Implementuokite metodus plotas() ir perimetras() kiekvienoje vaikynėje klasėje.
	#1.4 Sukurkite kiekvienos klasės objektus ir iškvieskite ploto() ir perimetro() metodus.

from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
        def plotas(self,):
