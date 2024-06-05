#Sukurkite stateless klasę StringUtils su statiniais metodais
# įprastoms eilutės operacijoms atlikti: to_uppercase, to_lowercase ir reverse_string.
class StringUtils:
    @staticmethod
    def uppercase(s):
        return s.upper()

    @staticmethod
    def lowercase(s):
        return s.lower()

    @staticmethod
    def reverse(s):
        return s[::-1]

string = str(input("Iveskite fraze"))

upper = StringUtils.uppercase(string)
lower = StringUtils.lowercase(string)
revers = StringUtils.reverse(string)

print(f"Ivesta fraze {string}")
print(f"Fraze didelem raidem: {upper}")
print(f"Fraze mazom raidem: {lower}")
print(f"Apversta fraze: {revers}")
