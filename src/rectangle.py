#4 užduotis
	#4.1 Apibrėžkite Rectangle klasę su atributais ilgis ir plotis.
	#4.2 Pridėkite metodus get_area ir get_perimeter, kad apskaičiuotumėte ir grąžintumėte stačiakampio plotą ir perimetrą.
	#4.3 Sukurkite Rectangle objektą ir išspausdinkite plotą bei perimetrą.


class Rectangle:
    def __init__(self,ilgis: int, plotis: int):
        self.ilgis = ilgis
        self.plotis = plotis

    def get_area(self):
        plotas = self.ilgis * self.plotis
        print(f"Staciakampio plotis lygus {plotas}")

    def get_perimeter(self):
        perimetras = (self.ilgis + self.plotis) * 2
        print(f"Staciakampio perimetras lygus {perimetras}")

staciak = Rectangle(ilgis= 2,plotis= 5)

staciak.get_area()
staciak.get_perimeter()