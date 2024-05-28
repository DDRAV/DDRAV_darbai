
import math


class Circle:
    def __init__(self, radius: float):
        self.radius = radius

    def area(self):
        arearez = math.pi * self.radius ** 2
        print(f"Apskritimo plotas lygus {arearez:.2f}")

    def circumference(self):
        perimrez = 2 * math.pi * self.radius
        print(f"Apskritimo perimetras lygus {perimrez:.2f}")


apskritimas1 = Circle(radius=3)
apskritimas2 = Circle(radius=5)

apskritimas1.area()
apskritimas1.circumference()
apskritimas2.area()
apskritimas2.circumference()
