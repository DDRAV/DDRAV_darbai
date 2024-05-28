class Item:
	def __init__(self, name: str, price: float):
		self.name = name
		self.price = price

	def __str__(self):
		return f"Preke: {self.name}\nKaina: {self.price:.2f}"


class Order:
	def __init__(self, order_id: int, items=None):
		self.order_id = order_id
		if items is None:
			self.items = []
		else:
			self.items = items

	def add_item(self, item: Item):
		self.items.append(item)
		print(f"Preke {item.name} prideta prie uzsakymo nr: {self.order_id}")

	def remove_item(self, item_name: str):
		for item in self.items:
			if item.name == item_name:
				self.items.remove(item)
				print(f"Preke {item_name} pasalinta is uzsakymo nr: {self.order_id}")
				return
		print(f"Preke {item_name} nerasta uzsakyme nr: {self.order_id}")

	def display_order(self):
		print(f"Uzsakymo nr: {self.order_id}")
		if self.items:
			print("Prekes dabartiniame uzsakyme:")
			for item in self.items:
				print(f"- {item}")
		else:
			print(f"Uzsakyme {self.order_id} nera prekiu")

	def checkout(self):
		suma = 0
		for item in self.items:
			suma += item.price
		print(f"Uzsakymo {self.order_id} galutine kaina: {suma:.2f}")


item1 = Item(name="Laptopas", price=999.99)
item2 = Item(name="Pelyte", price=25.50)
item3 = Item(name="Klaviatura", price=45.30)

order1 = Order(order_id=1)

order1.display_order()

order1.add_item(item1)
order1.add_item(item2)
order1.display_order()

order1.remove_item("Laptopas")
order1.display_order()

order1.add_item(item3)
order1.display_order()
order1.checkout()
