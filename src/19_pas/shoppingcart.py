#5. Sukurkite klasę ShoppingCart, kuriame būtų prekių sąrašas.
# Joje turėtų būti metodai: pridėti prekes, pašalinti prekes, peržiūrėti krepšelį ir apskaičiuoti bendrą kainą.

class Item:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def __repr__(self):
        print(f"Item({self.name}, {self.price:.2f})")


class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)
        print(f"{item.name} pridėta į krepšelį. Kaina: {item.price:.2f}")

    def remove_item(self, item_name):
        for i in range(len(self.items)):
            if self.items[i].name == item_name:
                removed_item = self.items.pop(i)
                print(f"{removed_item.name} pašalinta iš krepšelio.")
                return
        print(f"{item_name} nerasta krepšelyje.")

    def view_cart(self):
        if not self.items:
            print("Krepšelis tuščias.")
        else:
            print("Prekės krepšelyje:")
            for item in self.items:
                print(f"- {item.name}: {item.price:.2f}")

    def calculate_total(self):
        total = sum(item.price for item in self.items)
        print(f"Bendra kaina: {total:.2f}")

cart = ShoppingCart()
item1 = Item("Obuolys", 0.99)
item2 = Item("Bananai", 1.49)
item3 = Item(name="Mango", price= 2.39)

cart.add_item(item1)
cart.add_item(item2)
cart.view_cart()
cart.calculate_total()
cart.remove_item("Obuolys")
cart.add_item(item3)
cart.view_cart()
cart.calculate_total()
