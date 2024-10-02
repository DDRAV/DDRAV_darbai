#Jums pateiktas duomenų rinkinys, kuriame yra namų dydžiai (kvadratiniais metrais) ir jų atitinkamos kainos (tūkstančiais eurų).
	#Dydis Kaina
	#50	    150
	#80	    220
	#120	320
	#160	410
	#200	480

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

Dydis = np.array([50, 80, 120, 160, 200]).flatten()

Kaina = np.array([150, 220, 320, 410, 480])

#1Nubraižykite duomenų taškus (namų dydis vs. kaina) grafike.
plt.scatter(Dydis, Kaina, color='blue', label='Duomenu taskai')

plt.xlabel('Namo dydis')

plt.ylabel('Namo kaina')

plt.title('Dydzio nuo kainos priklausomybe')

plt.legend()

plt.grid(True)

plt.show()

#Atlikite tiesinę regresiją, kad rastumėte geriausiai tinkančią tiesę, kuri prognozuotų namų kainas pagal jų dydį.
X = np.array(Dydis).reshape(-1, 1)

Y = np.array(Kaina)

model = LinearRegression()

model.fit(X, Y)

beta_1 = model.coef_[0]

beta_0 = model.intercept_

Y_pred = model.predict(X)

r_squared = model.score(X, Y)

plt.scatter(Dydis, Kaina, color='blue', label='Duomenu taskai')

plt.plot(X, Y_pred, color='red', linewidth=2, label=f'Regression line: Y = {beta_0:.2f} + {beta_1:.2f}X')

plt.xlabel('Namo dydis')

plt.ylabel('Namo kaina')

plt.title('Dydzio nuo kainos priklausomybe')

plt.legend()

plt.grid(True)

plt.show()

#2.3 Naudodamiesi modeliu, prognozuokite 150 kvadratinių metrų namo kainą.

namo_dydis = 150
prognozuota_kaina = model.predict(np.array(namo_dydis).reshape(1, -1))
print(f"Prognozuojama 150 kvadratinių metrų namo kaina: {prognozuota_kaina[0]:.2f} tūkst. eurų")


#4 3 variantas
mse2 = r_squared
print(f"Vidutinė kvadratinė klaida (MSE2): {mse2:.2f}")
