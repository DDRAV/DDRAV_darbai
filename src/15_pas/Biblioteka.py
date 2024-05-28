#4. Sukurkite paprastą bibliotekos sistemą su knygomis, nariais ir skolintomis knygomis.
	#4.1 Sukurkite bazinę klasę Book su atributais title, author ir isbn (knygis id).
	#4.2 Sukurkite bazinę klasę Member su atributais name ir member_id.
	#4.3 Sukurkite išvestinę klasę BorrowedBook, kuri paveldi iš Book ir prideda atributus borrower_name (member vardas kuris pasiėmė knygą) ir due_date (iki kada pasiėmė).
	#4.4. Sukurkite Member ir imituokite bibliotekos veiklą.


class Book:
    def __init__(self,title: str,author: str,isbn: int):
        self.title = title
        self.author = author
        self.isbn = isbn

    def __str__(self):
        return f"{self.title} knyga parasyta {self.author} (ISBN: {self.isbn})"

class Member:
    def __init__(self,name: str,member_id: str):
        self.name = name
        self.member_id = member_id

    def __str__(self):
        return f"Vartotojas {self.name} (ID: {self.member_id})"

class BorrowedBook(Book):
    def __init__(self,title: str,author: str,isbn: int,borrower_name: str, term: int):
        super().__init__(title,author,isbn)
        self.borrower_name = borrower_name
        self.term = term

    def __str__(self):
        return (f"{self.title} knyga parasyta {self.author} (ISBN: {self.isbn})\n"
                f"Paskolinta: {self.borrower_name}\n"
                f"Tokiam dienu skaiciui: {self.term}")

user1 = Member(name="Darius",member_id="001")
print(user1)
user2 = Member(name="Marius",member_id="002")
print(user2)
book1 = Book("1984", "George Orwell", "1234567890")
print(book1)
book2 = Book("To Kill a Mockingbird", "Harper Lee", "2345678901")
print(book2)
borrow1 = BorrowedBook("1984", "George Orwell", "1234567890",user1.name,7)
print(borrow1)
borrow2 = BorrowedBook("To Kill a Mockingbird", "Harper Lee", "2345678901",user2.name,14)
print(borrow2)



