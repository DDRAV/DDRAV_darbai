class BankAccount:
    def __init__(self,savininkas: str,likutis: int):
        self.savininkas = savininkas
        self.likutis = likutis

    def deposit(self, inesta: int) -> None:
        self.likutis += inesta

    def withdraw(self, isimta: int) -> None:
        self.likutis -= isimta

    def display_balance(self) -> str:
        return f"{self.savininkas} dabartinis balanso likutis: {self.likutis}"

user1 = BankAccount(savininkas="Darius",likutis="100")

print(user1.display_balance())
user1.deposit(50)
print(user1.display_balance())