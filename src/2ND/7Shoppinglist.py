class ShoppingCart:
    def __init__(self):
        self._items = {}

    def add_item(self, item_name, quantity):
        if item_name in self._items:
            self._items[item_name] += quantity
        else:
            self._items[item_name] = quantity
        self._calculate_total()

    def remove_item(self, item_name, quantity):
        if item_name in self._items:
            self._items[item_name] -= quantity
            if self._items[item_name] <= 0:
                del self._items[item_name]
        self._calculate_total()

    def _calculate_total(self):
        self._total_items = sum(self._items.values())

    def get_total_items(self):
        return self._items


cart = ShoppingCart()
cart.add_item("obuolys", 3)
cart.add_item("bananas", 2)
print(cart.get_total_items())

cart.add_item("obuolys", 2)
print(cart.get_total_items())

cart.remove_item("obuolys", 5)
print(cart.get_total_items())
