#5 užduotis
	#5.1 Apibrėžkite klasę pavadinimu Student.
	#5.2 Klasė turi turėti inicializatorių, kuris nustato du atributus: name, class ir grade.
	#5.3 Pridėkite metodą get_grade, kad grąžintumėte įvertinimą.
	#5.4 Pridėkite metodą set_grade, kad pakeistumėte įvertinimą (užtikrinkite, kad įvertinimas būtų nuo 0 iki 100).
	#5.5 Pridėkite metodą print_studen_info, kuris gražintų informaciją apie studentą.
	#5.6 Sukurkite egzempliorių ir išbandykite metodus.

class Student:
    def __init__(self,name: str, stud_class: int, grade: int):
        self.name = name
        self.stud_class = stud_class
        self.grade = grade

    def get_grade(self):
        return print(f"{self.name} ivertinimas {self.grade}")

    def set_grade(self,new_grade: int):
        if new_grade <= 100 and new_grade >= 0:
            self.grade = new_grade
        else:
            print("Ivestas neteisinga iverinimas")

    def print_info(self):
        return print(f"Studentas {self.name} mokosi {self.stud_class} ir turi {self.grade} ivertinima")

stud1 = Student(name="Darius",stud_class= 10, grade= 90)

stud1.print_info()
stud1.set_grade(79)
stud1.get_grade()
stud1.set_grade(101)
stud1.print_info()