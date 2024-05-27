#6 užduotis
	#6.1 Apibrėžkite klasę PasswordValidator su atributu password.
	#6.2 Pridėkite privačius metodus _has_uppercase, _has_lowercase ir _has_digit kurie grąžina True,
# jei slaptažodyje yra bent viena didžioji raidė, mažoji raidė ar skaitmuo.
	#6.3 Pridėkite viešąjį (public) metodą is_valid, kuris, naudodamasis šiais privačiaisiais metodais,
# tikrina, ar slaptažodis yra galiojantis.
	#6.4 Sukurkite PasswordValidator objektą ir išbandykite metodą is_valid.

class PasswordValidator:
    def __init__(self,password: str):
        self.password = password

    def _has_uppercase(self) -> bool:
        return any(char.isupper() for char in self.password)

    def _has_lowercase(self) -> bool:
        return any(char.islower() for char in self.password)

    def _has_digit(self) -> bool:
        return any(char.isdigit() for char in self.password)

    def is_valid(self) -> bool:
        return self._has_uppercase() and self._has_lowercase() and self._has_digit()

pass1 = PasswordValidator("Example123")
print(pass1.is_valid())

pass2 = PasswordValidator("example123")
print(pass2.is_valid())

pass3 = PasswordValidator("EXAMPLE123")
print(pass3.is_valid())

pass4 = PasswordValidator("Example")
print(pass4.is_valid())