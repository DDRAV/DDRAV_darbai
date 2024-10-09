#Užduotis *: Klasifikuoti naujienų straipsnius į vieną iš 20 kategorijų (pvz., sportas, mokslas, politika).
#Sukurkite sprendimo medį su Gini indeksu ir entropija.
#Palyginkite modelių tikslumą ir kryžminį patikrinimą.
#Vizualizuokite sprendimo medį ir aptarkite svarbiausius žodžius klasifikacijai.
#Duomenų rinkinys: Naudokite sklearn.datasets.fetch_20newsgroups().

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Fetch 20 newsgroups dataset (only train data)
categories = None  # You can specify categories like ['rec.sport.baseball', 'sci.med']
newsgroups_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Vektorizacija (galima naudoti ir CountVectorizer, ir TfidfVectorizer)
vectorizer = CountVectorizer(max_features=5000)  # Or use CountVectorizer()
X = vectorizer.fit_transform(newsgroups_data.data)
y = newsgroups_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


gini_tree = DecisionTreeClassifier(criterion='gini', random_state=42)
entropy_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Random Forest modelis
random_forest = RandomForestClassifier(random_state=42)

# 3. Nustatome parametrų rinkinį GridSearchCV tiek sprendimo medžiams, tiek random forest
param_grid_tree = {
    'max_depth': [50,100],  # Limit the depth of the tree
    'min_samples_split': [10, 20],  # Increase the minimum number of samples required to split
    'min_samples_leaf': [2, 5]  # Increase the minimum number of samples per leaf
}

param_grid_rf = {
    'n_estimators': [50, 100],  # Number of trees in the forest
    'max_depth': [50, 100],  # Maximum depth of the trees
    'min_samples_split': [10, 20],  # Minimum samples required to split
    'min_samples_leaf': [2, 5]  # Minimum samples required in leaf nodes
}

# Optimizuojame parametrus naudojant GridSearchCV su Gini indeksu
gini_grid_search = GridSearchCV(estimator=gini_tree, param_grid=param_grid_tree, cv=3, n_jobs=-1, verbose=1)
gini_grid_search.fit(X_train, y_train)

# Optimizuojame parametrus naudojant GridSearchCV su Entropija
entropy_grid_search = GridSearchCV(estimator=entropy_tree, param_grid=param_grid_tree, cv=3, n_jobs=-1, verbose=1)
entropy_grid_search.fit(X_train, y_train)

# Optimizuojame parametrus naudojant GridSearchCV su Random Forest
rf_grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)

# Išvedame geriausius parametrus abiem modeliams
print("Geriausi Gini parametrai:", gini_grid_search.best_params_)
print("Geriausi Entropy parametrai:", entropy_grid_search.best_params_)
print("Geriausi Random Forest parametrai:", rf_grid_search.best_params_)

# 4. Kryžminis patikrinimas su geriausiais modeliais
gini_best_tree = gini_grid_search.best_estimator_
entropy_best_tree = entropy_grid_search.best_estimator_
rf_best_model = rf_grid_search.best_estimator_

# Kryžminio patikrinimo rezultatai
gini_cv_scores = cross_val_score(gini_best_tree, X_train, y_train, cv=3)
entropy_cv_scores = cross_val_score(entropy_best_tree, X_train, y_train, cv=3)
rf_cv_scores = cross_val_score(rf_best_model, X_train, y_train, cv=3)

print("Geriausio Gini medžio kryžminio patikrinimo tikslumas:", np.mean(gini_cv_scores))
print("Geriausio Entropy medžio kryžminio patikrinimo tikslumas:", np.mean(entropy_cv_scores))
print("Geriausio Random Forest modelio kryžminio patikrinimo tikslumas:", np.mean(rf_cv_scores))

# 5. Vizualizacija (tik sprendimų medžiams, Random Forest sunkiau vizualizuoti dėl didelio medžių kiekio)
plt.figure(figsize=(20, 10))
plot_tree(gini_best_tree, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=newsgroups_data.target_names, max_depth=3)
plt.title("Optimizuotas sprendimų medis su Gini indeksu")
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(entropy_best_tree, filled=True, feature_names=vectorizer.get_feature_names_out(), class_names=newsgroups_data.target_names, max_depth=3)
plt.title("Optimizuotas sprendimų medis su Entropija")
plt.show()

# 6. Svarbių žodžių aptarimas
gini_importances = gini_best_tree.feature_importances_
entropy_importances = entropy_best_tree.feature_importances_
rf_importances = rf_best_model.feature_importances_

# Ištraukiame svarbiausius žodžius
important_gini_words = np.argsort(gini_importances)[-10:]
important_entropy_words = np.argsort(entropy_importances)[-10:]
important_rf_words = np.argsort(rf_importances)[-10:]

print("Svarbiausi žodžiai optimizuotam Gini medžiui:", [vectorizer.get_feature_names_out()[i] for i in important_gini_words])
print("Svarbiausi žodžiai optimizuotam Entropy medžiui:", [vectorizer.get_feature_names_out()[i] for i in important_entropy_words])
print("Svarbiausi žodžiai optimizuotam Random Forest modeliui:", [vectorizer.get_feature_names_out()[i] for i in important_rf_words])