#paveldejimas - inheritence
#base, inherited

class Animals:

    def __init__(self,name: str,age: int):
        self.name = name
        self.age = age

    def walk(self):
        return f'{self.name} started walking'

    def speak(self, sound:str):
        return f'{self.name} is saying: {sound}'

#jei norime paveldes is Animals parasom taip:
class Cat(Animals):

    def __init__(self, name:str, age: int):
        super().__init__(name,age)

    def doing(self):
        return f'{self.name} started to do something'

cat = Cat(name='kitty', age=3)
print(cat.walk())
print(cat.speak('meow'))
print(cat.age)
print(cat.doing())

#visi paveldetojai gali naudotis paldetos clases funkcijom,
# bet paveldetoju funkcijom gali nautois tik jie
# jei paveldimoji klase turi tokia pacia funkcija iskarto nauskaitoma jos funkcija on top
# jei norime grazint ir tevines klases funkcija ja galime iskviest su
# super().speak(sound)