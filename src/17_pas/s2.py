#Sukurkite statefull klasę Skaitiklis, kuri saugoja skaičių.
#Joje turėtų būti metodai, skirti skaičiui padidinti, sumažinti, atstatyti ir dabartiniam skaičiui gauti.

class Skaitiklis:
    value = 0

    @staticmethod
    def padidinti(amount=1):
        Skaitiklis.value += amount

    @staticmethod
    def sumazinti(amount=1):
        Skaitiklis.value -= amount

    @staticmethod
    def atstatyti():
        Skaitiklis.value = 0

    @staticmethod
    def gauti():
        return print(f"Dabartinis saugomas skaicius: {Skaitiklis.value}")


Skaitiklis.padidinti(12)
Skaitiklis.gauti()
Skaitiklis.sumazinti(5)
Skaitiklis.gauti()
Skaitiklis.atstatyti()
Skaitiklis.gauti()

