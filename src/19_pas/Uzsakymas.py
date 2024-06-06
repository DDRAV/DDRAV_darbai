class Item:
    def __init__(self, name, quantity, price):
        self.name = name
        self.quantity = quantity
        self.price = price

    def __repr__(self):
        return f"Preke({self.name}, Kiekis: {self.quantity}, Prekes kaina: {self.price:.2f})"


class Order:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)
        print(f"Prekė {item.name} (kiekis: {item.quantity}, kaina: {item.price:.2f} už vienetą) pridėta prie užsakymo.")

    def remove_item(self, item_name):
        for item in self.items:
            if item.name == item_name:
                self.items.remove(item)
                print(f"Prekė {item.name} pašalinta iš užsakymo.")
                break
        print(f"Prekė {item_name} nerasta užsakyme.")

    def calculate_total(self):
        total = sum(item.quantity * item.price for item in self.items)
        print(f"Bendra užsakymo kaina: {total:.2f}")
        return total

    def apply_discount(self, nuolaida):
        if not 0 <= nuolaida <= 100:
            print("Nuolaida turi būti tarp 0 ir 100 procentų.")
            return None
        pries_nuolaida = self.calculate_total()
        nuolaidos_suma = pries_nuolaida * (nuolaida / 100)
        po_nuolaidos = pries_nuolaida - nuolaidos_suma
        print(f"Nuolaida: {nuolaida}%, nuolaidos suma: {nuolaidos_suma:.2f}")
        print(f"Bendra kaina po nuolaidos: {po_nuolaidos:.2f}")
        return po_nuolaidos

    def view_order(self):
        if not self.items:
            print("Užsakymas tuščias.")
        else:
            print("Prekės užsakyme:")
            for item in self.items:
                print(f"- {item.name}: kiekis: {item.quantity}, kaina: {item.price:.2f} už vienetą")


order = Order()
item1 = Item("Obuolys", 5, 0.99)
item2 = Item("Bananai", 3, 1.49)
item3 = Item("Mangai", 7, 2.39)

order.add_item(item1)
order.add_item(item2)
order.view_order()
order.calculate_total() ## kazkodel po sios funkcijos 2 kartus isspausdina bendra uzsakymo kaina, kaip to isvengt?
order.apply_discount(5)
order.remove_item("Obuolys")
order.add_item(item3)
order.view_order()
order.calculate_total()
order.apply_discount(10)
