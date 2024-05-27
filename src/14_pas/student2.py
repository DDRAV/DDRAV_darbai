#10 užduotis
	#10.1 Apibrėžkite klasę Student su atributais name ir courses (objektų Course sąrašas).
	#10.2 Apibrėžkite klasę Kursas su atributais title (pavadinimas) ir students (objektų Student sąrašas).
	#10.3 Pridėkite Studentui metodus: register - registruotis į kursą ir drop - nutraukti kursą.
	#10.4 Į kursą pridėkite metodus: add_student ir remove_student.
	#10.5 Sukurkite Student ir Course objektus ir imituokite kursų registravimą ir nutraukimą.

class Student:
    def __init__(self,name: str):
        self.name = name
        self.courses = []

    def register(self, course):
        if course not in self.courses:
            self.courses.append(course)
            course.add_student(self)

    def drop(self, course):
        if course in self.courses:
            self.courses.remove(course)
            course.remove_student(self)



class Kursas:
    def __init__(self,title: str):
        self.title = title
        self.students = []

    def add_student(self,student):
        if student not in self.students:
            self.students.append(student)
            student.register(self)


    def remove_student(self,student):
        if student in self.students:
            self.students.remove(student)
            student.drop(self)


student1 = Student("Onute")
student2 = Student("Jonas")

kursas1 = Kursas("Matematika")
kursas2 = Kursas("Chemija")

kursas2.add_student(student2)
kursas1.add_student(student1)
kursas1.add_student(student2)

print(f"Studentai {kursas1.title} kurse: {[student.name for student in kursas1.students]}")
print(f"Studentai {kursas2.title} kurse: {[student.name for student in kursas2.students]}")

