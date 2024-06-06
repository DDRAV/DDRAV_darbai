#6. Sukurkite klasę UserSession, kuri valdo vartotojo prisijungimo būseną.
# Joje turėtų būti būdai prisijungti, atsijungti, patikrinti, ar vartotojas yra prisijungęs, ir gauti prisijungusio vartotojo duomenis.

class UserSession:
    def __init__(self):
        self.user = None

    def login(self, username, password):
        self.username = username
        self.password = password
        self.user = {'Username': self.username, 'Password': self.password}
        print(f"Vartotojas {self.username} prisijungė.")

    def logout(self):
        if self.user:
            print(f"Vartotojas {self.username} atsijungė.")
            self.user = None
        else:
            print("Nėra prisijungusio vartotojo.")

    def is_logged_in(self):
        if self.user is None:
            return False
        else:
            return True

    def get_user_data(self):
        if self.user:
            return self.user
        else:
            print("Nėra prisijungusio vartotojo.")
            return None


session = UserSession()
session.login("Jonas", "slaptazodis123")
print("Ar vartotojas prisijungęs?", session.is_logged_in())
print("Prisijungusio vartotojo duomenys:", session.get_user_data())
session.logout()
print("Ar vartotojas prisijungęs?", session.is_logged_in())
print("Prisijungusio vartotojo duomenys:", session.get_user_data())