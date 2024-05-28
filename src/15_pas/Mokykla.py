# Parašykite programą, kuri imituotų mokyklos systemą.
# Implementuokite klases: Person, Teacher, Student, Cource.
# Programoje turi būti galimybė priskirti studentą prie kurso,
# atspausdinti informaciją apie kursą: kas dėsto, kiek valandų, koks pažymys.

class Person:
    def __init__(self, vardas: str, amzius: int):
        self.vardas = vardas
        self.amzius = amzius

    def __str__(self):
        return f"{self.vardas} Amzius: {self.amzius}"


class Teacher(Person):
    def __init__(self, vardas: str, amzius: int, paskaita: str):
        super().__init__(vardas, amzius)
        self.paskaita = paskaita

    def __str__(self):
        return f"{self.vardas} Amzius: {self.amzius}\n Destomas dalykas: {self.paskaita}"


class Student(Person):
    def __init__(self, vardas: str, amzius: int, pazymis: int):
        super().__init__(vardas, amzius)
        self.pazymis = pazymis

    def __str__(self):
        return f"{self.vardas} Amzius: {self.amzius}\n Pazymis: {self.pazymis}"


class Course:
    def __init__(self, name: str, destytojas: str, valandu_kiekis: int):
        self.name = name
        self.destytojas = destytojas
        self.valandu_kiekis = valandu_kiekis
        self.students = []

    def add_students(self, student):
        self.students.append(student)

    def print_info(self):
        print(f"Kursas: {self.name}")
        print(f"Desto: {self.destytojas}")
        print(f"Valandos: {self.valandu_kiekis}")
        print("Mokinantys studentai:")
        for student in self.students:
            print(f"- {student}")


stud1 = Student(vardas="Mykolas", amzius=20, pazymis=9)
stud2 = Student(vardas="Aurimas", amzius=22, pazymis=8)
stud3 = Student(vardas="Tautvydas", amzius=21, pazymis=6)
stud4 = Student(vardas="Elena", amzius=19, pazymis=10)

dest1 = Teacher(vardas="Vygintas", amzius=45, paskaita="Fizika")
dest2 = Teacher(vardas="Marija", amzius=37, paskaita="Matematika")

kurs1 = Course(name="Aerodinamika", destytojas=dest1, valandu_kiekis=20)
kurs2 = Course(name="Geometrija", destytojas=dest2, valandu_kiekis=80)

kurs1.add_students(stud1)
kurs1.add_students(stud3)

kurs2.add_students(stud2)
kurs2.add_students(stud4)

kurs1.print_info()
kurs2.print_info()


