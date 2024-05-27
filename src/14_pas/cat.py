#3 užduotis
	#3.1 Sukurkite Cat klasę.
	#3.2 Pridėkite klasės atributą total_cats, kuris fiksuoja sukurtų katės egzempliorių skaičių.
	#3.3 Inicializatoriuje padidinkite total_cats.
	#3.4 Sukūrę keletą egzempliorių, išspausdinkite bendrą kačių skaičių.
class Cat:
    total_cats: int = 0
    def __init__(self,name: str, age: int):
        self.name = name
        self.age = age
        Cat.total_cats += 1

cat1 = Cat(name="Barsikas",age=2)
cat2 = Cat(name="Varsikas",age=5)
cat3 = Cat(name="Marsikas",age=4)
cat4 = Cat(name="Tarsikas",age=3)
cat5 = Cat(name="Larsikas",age=2)

print(f"Siuo metu zinome apie {Cat.total_cats} kates")
