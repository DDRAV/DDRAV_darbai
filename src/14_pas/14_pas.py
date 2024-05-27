class Dog:
    ##jei clasei naudojama konstanta irasom:
    legs: int = 4

    ## jei norim kad konstanta keistusi:
    total_dogs: int = 0


    #ir poto rasom konstruktyva ir funkcijas

    def __init__(self, species: str, name: str, age: int):
        self.species = species
        self.name = name
        self.age = age
        ##pakeiciam konstanta jei irasom irasa
        Dog.total_dogs += 1

    def description(self) -> str:
        return f"Dog: {self.species}, {self.age}, {self.name}"

    def speak(self, sound: str) -> str:
        return f"Dog named {self.name} says {sound}"

dog = Dog(species='labradoras', name='rex', age=5)
desc = dog.description()
print(desc)
print(dog.legs)
print(dog.total_dogs)