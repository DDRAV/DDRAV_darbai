import numpy as np

masyvas_1d = np.arange(1, 11)
print(masyvas_1d)

masyvas_2d = np.full((3, 3), 7)
print(masyvas_2d)

# Sukuriame du 1D masyvus su reikšmėmis nuo 1 iki 5
masyvas1 = np.arange(1, 6)
masyvas2 = np.arange(1, 6)

# Sudėtis
sudetis = masyvas1 + masyvas2
print("Sudėtis:", sudetis)

# Atimtis
atimtis = masyvas1 - masyvas2
print("Atimtis:", atimtis)

# Daugyba
daugyba = masyvas1 * masyvas2
print("Daugyba:", daugyba)

# Dalyba
dalyba = masyvas1 / masyvas2
print("Dalyba:", dalyba)

# Sukuriame 1D masyvą su reikšmėmis nuo 10 iki 20
masyvas_1d_10_20 = np.arange(10, 21)
print("Visas masyvas:", masyvas_1d_10_20)

# Pirmieji 5 elementai
pirmi_5 = masyvas_1d_10_20[:5]
print("Pirmieji 5 elementai:", pirmi_5)

# Paskutiniai 5 elementai
paskutiniai_5 = masyvas_1d_10_20[-5:]
print("Paskutiniai 5 elementai:", paskutiniai_5)

#44.1 Sukurkite 1D masyvą su 12 elementų ir pertvarkykite jį į 2D masyvą, kurio forma (3, 4).

masyvas12 = np.arange(1, 13)
print("Pradinis masyvas:", masyvas12)

masyvas3x4 = masyvas12.reshape(3, 4)
print("2D masyvas 3x4:")
print(masyvas3x4)

#5Masyvų generavimas naudojant integruotas funkcijas
# 5.1 Naudodami linspace sukurkite 10 tolygiai išdėstytų reikšmių nuo 0 iki 1 masyvą.
# 5.2 Sukurkite 1D masyvą su reikšmėmis nuo 1 iki 10. Apskaičiuokite ir išspausdinkite sumą, vidurkį ir standartinį nuokrypį.

# Sukuriame 10 tolygiai išdėstytų reikšmių nuo 0 iki 1 masyvą
masyvas5uzd = np.linspace(0, 1, 10)
print("10 tolygiai išdėstytų reikšmių nuo 0 iki 1 masyvas:")
print(masyvas5uzd)

# Apskaičiuojame sumą
suma = np.sum(masyvas5uzd)
print("Suma:", suma)

# Apskaičiuojame vidurkį
vidurkis = np.mean(masyvas5uzd)
print("Vidurkis:", vidurkis)

# Apskaičiuojame standartinį nuokrypį
std_nuokrypis = np.std(masyvas5uzd)
print("Standartinis nuokrypis:", std_nuokrypis)

#6. Operacijos su matricomis
# 6.1 Sukurkite dvi 2D formos (3, 3) matricas. Atlikite matricų sudėties, atimties ir matricų daugybos veiksmus.

matrica61 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrica62 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

print("Matrica 1:")
print(matrica61)
print("Matrica 2:")
print(matrica62)

# Matricų sudėtis
sudetis = matrica61 + matrica62
print("Matricų sudėtis:")
print(sudetis)

# Matricų atimtis
atimtis = matrica61 - matrica62
print("Matricų atimtis:")
print(atimtis)

# Matricų daugyba (elementų daugyba)
daugyba_elementais = matrica61 * matrica62
print("Matricų daugyba (elementų daugyba):")
print(daugyba_elementais)

# Matricų daugyba (matricų daugyba)
daugyba_matricu = np.dot(matrica61, matrica62)
print("Matricų daugyba (matricų daugyba):")
print(daugyba_matricu)

# 6.2 Sukurkite 1D matricą su reikšmėmis [0, π/2, π, 3π/2, 2π]. Apskaičiuokite šių reikšmių sinusą ir kosinusą.


# Sukuriame 1D masyvą su reikšmėmis [0, π/2, π, 3π/2, 2π]
reiksmes = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
print("Reikšmės:")
print(reiksmes)

# Apskaičiuojame sinusą
sinusai = np.sin(reiksmes)
print("Sinusai:")
print(sinusai)

# Apskaičiuojame kosinusą
kosinusai = np.cos(reiksmes)
print("Kosinusai:")
print(kosinusai)

#7. Statistinės funkcijos
# .1 Sukurkite 100 dydžio 1D masyvą su atsitiktinėmis reikšmėmis nuo 0 iki 1. Apskaičiuokite medianą ir dispersiją.

# Sukuriame 100 dydžio 1D masyvą su atsitiktinėmis reikšmėmis nuo 0 iki 1
masyvas7uzd = np.random.rand(100)
print("Atsitiktinis masyvas:")
print(masyvas7uzd)

mediana = np.median(masyvas7uzd)
print("Mediana:", mediana)

dispersija = np.var(masyvas7uzd)
print("Dispersija:", dispersija)

#8. Atsitiktinių skaičių generavimas
# 8.1 Sukurkite 3x3 masyvą su atsitiktiniais sveikaisiais skaičiais nuo 1 iki 10.

masyvas8uzd = np.random.randint(1, 11, size=(3, 3))
print("3x3 masyvas su atsitiktiniais sveikaisiais skaičiais nuo 1 iki 10:")
print(masyvas8uzd)

#9. Loginis indeksavimas
# 9.1 Sukurkite 1D masyvą su reikšmėmis nuo 1 iki 10. Išskirkite visus lyginius skaičius naudodami loginį indeksavimą.



masyvas9uzd = np.arange(1, 11)
print("1D masyvas su reikšmėmis nuo 1 iki 10:")
print(masyvas9uzd)

#isskyriame lyginius skaičius
lyginiai_skaiciai = masyvas9uzd[masyvas9uzd % 2 == 0]
print("Lyginiai skaičiai:")
print(lyginiai_skaiciai)

#10. Masyvų rūšiavimas
# 10.1 Sukurkite 1D masyvą su 10 atsitiktinių sveikųjų skaičių nuo 0 iki 50. Surūšiuokite masyvą didėjančia tvarka.
masyvas10uzd = np.random.randint(0, 51, size=10)
print("Atsitiktinis masyvas:")
print(masyvas10uzd)

rusiavimas = np.sort(masyvas10uzd)
print("Surūšiuotas masyvas didėjančia tvarka:")
print(rusiavimas)

#11. Tiesinės algebros operacijos
# 11.1 Sukurkite 2D formos (2, 2) masyvą A ir kitą 2D formos (2, 2) masyvą B. Apskaičiuokite A ir B sandaugą ir A matricos determinantą.

matrica11uzd1 = np.array([[1, 2], [3, 4]])
print("Matrica A:")
print(matrica11uzd1)

# Sukuriame 2D masyvą B (2, 2) formos
matrica11uzd2 = np.array([[5, 6], [7, 8]])
print("Matrica B:")
print(matrica11uzd2)

# Apskaičiuojame A ir B sandaugą
sandauga = np.dot(matrica11uzd1, matrica11uzd2)
print("A ir B sandauga:")
print(sandauga)

# Apskaičiuojame A matricos determinantą
determinantas = np.linalg.det(matrica11uzd1)
print("A matricos determinantas:")
print(determinantas)

#12. Išplėstinis indeksavimas
# 12.1 Sukurkite 4x4 masyvą ir naudodami išplėstinį indeksavimą įstrižainės elementams nustatykite reikšmę 1.

masyvas12uzd = np.random.randint(0, 21, size=(4, 4))
print("Masyvas:")
print(masyvas12uzd)

# Nustatome įstrižainėms elementams reikšmę 1
masyvas12uzd[np.arange(4), np.arange(4)] = 1
print("Masyvas su nustatytais įstrižainės elementais 1:")
print(masyvas12uzd)

#13. Tikimybių skaičiavimas
# 13.1 Naudodami "NumPy" imituokite dviejų šešiabriaunių kauliukų metimo procesą 1000 kartų. Apskaičiuokite tikimybę, kad metant bus gauta suma 7.


# Imituojame dviejų šešiabriaunių kauliukų metimą 1000 kartų
metymai = np.random.randint(1, 7, size=(1000, 2))

sumos = np.sum(metymai, axis=1)

# Apskaičiuojame tikimybę, kad suma bus lygi 7
tikimybe = np.mean(sumos == 7)
print("Tikimybė, kad suma bus lygi 7:", tikimybe)
