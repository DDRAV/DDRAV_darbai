#7 užduotis
	#7.1 Apibrėžkite klasę ShoppingCart su atributu items (prekių kainų sąrašas, arba galite naudoti žodyną).
	#7.2 Pridėkite privatų metodą _calculate_discount, kuris taikys 10 % nuolaidą, jei bendra kaina viršija 100 Eur.
	#7.3 Pridėkite viešuosius metodus add_item, skirtus elementui pridėti į krepšelį,
# total_price, skirtus apskaičiuoti bendrą kainą, įskaitant visas nuolaidas,
# ir checkout, skirtus atspausdinti galutinę kainą pritaikius nuolaidą.
	#7.4 Sukurkite "ShoppingCart" objektą, pridėkite prekių ir išbandykite checkout metodą.

class ShoppingCart:
    def __init__(self):
        self.kainos = []

    def _calculate_discount(self, total: float) -> float:
        if total > 100:
            return total*0.1
        else:
            return 0 #jei kaina mazesne nei 100 nuolaidos nera

    def add_item(self, kaina: float):
        self.kainos.append(kaina)

    def total_price(self) ->float:
        total = sum(self.kainos)
        discount = self._calculate_discount(total)
        return total - discount

    def checkout(self):
        galutine_kaina=self.total_price()
        print(f"Galutine kaina su nuolaida: {galutine_kaina:.2f} Eur")

apsipirkimas = ShoppingCart()

apsipirkimas.add_item(45)
apsipirkimas.checkout()
apsipirkimas.add_item(45)
apsipirkimas.checkout()
apsipirkimas.add_item(30)
apsipirkimas.checkout()
