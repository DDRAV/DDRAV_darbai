#9 užduotis
	#9.1 Apibrėžkite klasę Book su atributais title, author ir available (numatytoji reikšmė yra True).
	#9.2 Apibrėžkite klasę Library su atributu books (Book objektų sąrašas).
	#9.3 Į Library pridėkite metodus: add_book - pridėti knygą,
# borrow_book - pasiskolinti knygą (nustatykite, kad available yra False) ir
# return_book - grąžinti knygą (nustatykite, kad available yra True).
	#9.4 Sukurkite Library ir Book objektus ir imituokite knygų skolinimąsi bei grąžinimą.


class Book:
    def __init__(self,title: str,author: str, available: bool = True):
        self.title = title
        self.author = author
        self.available = available

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book: Book):
        self.books.append(book)

    def borrow_book(self, book: Book):
        for b in self.books:
            if b.title == book.title and b.available:
                b.available = False
                return f"{b.title} Knyga paskolinta."
        return f"{book.title} knyga jau kazkam paskolinta."

    def return_book(self, book: Book):
        for b in self.books:
            if b.title == book.title and not b.available:
                b.available = True
                return f"{b.title} knyga grazinta."
        return f"{book.title} knyga negali but grazinta nes nera paskolinta."

    def actual_lib(self):
        library_info = "Knygu sarasas ir uzimtumas:\n"
        for book in self.books:
            availability = 'Prieinama' if book.available \
                else 'Neprienama'
            book_info = f"{book.title} ({book.author}) - {availability}\n"
            library_info += book_info
        return library_info


book1 = Book(title="To Kill a Mockingbird", author="Harper Lee")
book2 = Book(title="1984", author="George Orwell")

library = Library()
print(library.actual_lib())

library.add_book(book1)
library.add_book(book2)
print(library.actual_lib())

print(library.borrow_book(book1))
print(library.actual_lib())

print(library.borrow_book(book1))
print(library.actual_lib())

print(library.return_book(book2))
print(library.actual_lib())

print(library.return_book(book1))
print(library.actual_lib())



