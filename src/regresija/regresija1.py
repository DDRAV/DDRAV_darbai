import numpy as np

X_arr = np.array([2, 3, 5, 7, 8]).flatten()

Y_arr = np.array([50, 55, 70, 80, 85])

#1 savarankiskai be sklearn parasom tapati kas pavyzdyje
def regresion_constants(X_arr, Y_arr, X):
    # Calculate means of Dydis and Kaina
    X_vid = np.mean(X_arr)
    Y_vid = np.mean(Y_arr)

    # Numerator: Σ(Xi - X_mean)(Yi - Y_mean)
    numerator = np.sum((X_arr - X_vid) * (Y_arr - Y_vid))

    # Denominator: Σ(Xi - X_mean)²
    denominator = np.sum((X_arr - X_vid) ** 2)

    # Slope (β1)
    beta_1 = numerator / denominator

    beta_0 = Y_vid - beta_1*X_vid

    Y_pred = beta_0 + beta_1 * X_arr

    ss_res = np.sum((Y_arr - Y_pred) ** 2)

    ss_tot = np.sum((Y_arr - Y_vid) ** 2)

    r_squared = 1 - (ss_res / ss_tot)

    Y_rez = beta_0 + beta_1 * X

    print(f"X vid: {X_vid}\n"
          f"Y vid: {Y_vid}\n"
          f"Beta 1: {beta_1}\n"
          f"Beta 0: {beta_0}\n"
          f"R2: {r_squared}\n"
          f"Jei praleidome {X} val mokinantis tiketina gausime {Y_rez} balu")

    return X_vid, Y_vid, beta_1, beta_0, r_squared, Y_rez


regresion_constants(X_arr, Y_arr, 4)


