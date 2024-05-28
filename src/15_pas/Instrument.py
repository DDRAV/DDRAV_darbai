#3. Sukurkite bazinę klasę Instrument su metodu play, kuris atspausdina pranešimą kad instrumentas groja.
# Sukurkite dvi išvestines klases Guitar ir Drum,
# taip pat implementuokite metodą play() ir atspausdina info kuris butent instrumentas groja.
# Parašykite funkciją play_instrument, kuri priima objektą Instrument ir iškviečia jo metodą play.

class Instrument:
    def play(self):
        print(f'Instrument is playing')

class Guitar(Instrument):

    def play(self):
        print(f'Guitar is playing')

class Drums(Instrument):

    def play(self):
        print(f'Drums are plaing')

def play_instrument(instrument):
    instrument.play()

gitara = Guitar()
ukulele = Guitar()
bugnai = Drums()

play_instrument(gitara)
play_instrument(ukulele)
play_instrument(bugnai)