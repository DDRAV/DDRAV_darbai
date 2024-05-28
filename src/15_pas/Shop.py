#Sukurkite el. parduotuves sistemą. Joje turi būti bent dvieju tipu klientai: preminium (premium gauna nuolaidą) ir paprastas.
# Be to parduotivėje turi būti bent dvieju kategorijų produktai. Suteikite klientui galimybę suformuoti užsakymą.

class Product:
    def __init__(self, pavadinimas: str, id: str, kiekis: int, kaina: float):
        self.pavadinimas = pavadinimas
        self.id = id
        self.kiekis = kiekis
        self.kaina = kaina

    def __str__(self):
        return f"{self.pavadinimas} prekes (id:{self.id}) kaina {self.kaina} - Dabartinis kiekis sandelyje {self.kiekis}"


class Elektronika(Product):
    def __init__(self, pavadinimas: str, id: str, kiekis: int, kaina: float):
        super().__init__(pavadinimas, id, kiekis, kaina)
    
    def __str__(self):
        return super().__str__()


class Irankiai(Product):
    def __init__(self, pavadinimas: str, id: str, kiekis: int, kaina: float):
        super().__init__(pavadinimas, id, kiekis, kaina)
    
    def __str__(self):
        return super().__str__()
    
class User:
    def __init__(self, name: str, id: str):
        self.name = name
        self.id = id
        self.cart = {}

    def __str__(self):
        return f"{self.name} ID: {self.id}"

    def add_to_cart(self, product: Product, kiekis: int):
        if product.id in self.cart:
            self.cart[product.id]['Kiekis'] += kiekis
        else:
            self.cart[product.id] = {'Preke': product, 'Kiekis': kiekis}
        return print(self.cart)

    def calculate_total(self) -> float:
        total = sum(item['Preke'].kaina * item['Kiekis'] for item in self.cart.values())
        return total
    

class PremiumUser(User):
    def __init__(self, name: str, id: str, discount: int):
        super().__init__(name, id)
        self.discount = discount

    def __str__(self):
        return f"{self.name} ID: {self.id}, Discount: {self.discount}%"

    def calculate_total(self) -> float:
        total = super().calculate_total()
        nuolaida = total * (1 - self.discount / 100)
        return nuolaida

phone = Elektronika("Phone", "E01", 10, 699.99)
drill = Irankiai("Drill", "I01", 5, 199.99)

user1 = User("John Doe", "U01")
premium_user1 = PremiumUser("Jane Doe", "U02", 15)

sandelys = [phone,drill]

print(f"Sveiki, sandelyje turime tokias prekes:")
for product in sandelys:
    print(product)
pasirinkimas = str(input("Iveskite 'taip' jei norite kazka pirkti\nIveskite 'ne' jei norite baigti apsipirkima").lower())

if pasirinkimas == "taip":
