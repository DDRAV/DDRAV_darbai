class Teacher:
    def __init__(self, name: str, subject: str):
        self.name = name
        self.subject = subject

    def __str__(self):
        return f"{self.name} desto {self.subject}"


class Classroom:
    def __init__(self, teacher, students=None):
        self.teacher = teacher
        if students is None:
            self.students = []
        else:
            self.students = students

    def add_student(self, stud_name: str):
        self.students.append(stud_name)
        print(f"Studentas {stud_name} pridetas prie klases")

    def remove_student(self, stud_name: str):
        if stud_name in self.students:
            self.students.remove(stud_name)
            print(f"Studentas {stud_name} pasalintas is klases")
        else:
            print(f"Studentas {stud_name} nerastas klaseje")

    def show_class(self):
        print(f"Mokytojas {self.teacher}:")
        if self.students:
            print("Studentai:")
            for stud in self.students:
                print(f"- {stud}")
        else:
            print(f"Sioje klaseje nera studentu")


mokytojas1 = Teacher(name="Jonas", subject="Matematika")
mokytojas2 = Teacher(name="Ona", subject="Daile")

studentai1 = ["Darius", "Marius", "Karolina"]
studentai2 = ["Romualdas", "Adele", "Kotryna"]

klase1 = Classroom(mokytojas1, studentai2)
klase2 = Classroom(mokytojas2, studentai1)

klase1.show_class()

klase1.add_student("Karolis")
klase1.remove_student("Marius")
klase1.remove_student("Adele")

klase1.show_class()

klase2.show_class()
klase2.add_student("Adele")
klase2.show_class()
