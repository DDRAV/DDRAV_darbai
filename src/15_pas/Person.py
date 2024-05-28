#1. Sukurkite bazinę klasę Person su atributais name ir age.
# Apibrėžkite metodą introduce, kuris spausdina pasveikinimą.
# Sukurkite išvestinę klasę Student, kuri paveldi iš Person ir prideda atributą school_name.
# Pridėkite metodą print_info(), kur atspausdinkite visą žinomą informaciją apie žmogų.

class Person:

    def __init__(self,name: str,age: int):
        self.name = name
        self.age = age

    def introduce(self):
        print(f'Labas {self.name}\nMalonu susipazinti')

    def print_info(self):
        print(f'Vardas:{self.name}')
        print(f'Amzius:{self.age}')

class Student(Person):
    def __init__(self, name: str, age: int, school_name: str):
        self.school_name = school_name
        super().__init__(name,age)

    def print_info(self):
        super().print_info()
        print(f'Mokymosi istaiga: {self.school_name}')
person = Person(name='Darius', age=27)
student = Student(name='Dariukas', age=17, school_name='Vilnius Coding School')
person.introduce()
person.print_info()
student.introduce()
student.print_info()
