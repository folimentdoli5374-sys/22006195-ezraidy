# Prédiction des Coûts d'Assurance Santé par Machine Learning
![WhatsApp Image 2025-12-11 à 11 43 53_c949579c](https://github.com/user-attachments/assets/66a98179-12f8-4c30-8c7d-d2e6b8dd62b9)


**Projet de Data Science & Machine Learning - A. Larhlimi**  
**Année Universitaire 2025-2026**
**Ezraidy soulaimane**
**22006195**

---

## 1. Contexte et Objectif

Le secteur de l'assurance santé nécessite des modèles prédictifs précis pour estimer les coûts médicaux annuels. Cette analyse vise à développer un modèle de régression capable de prédire les frais médicaux d'un patient selon ses caractéristiques démographiques et comportementales. L'objectif est double : optimiser la tarification des contrats et identifier les facteurs de risque majeurs. Ce projet s'inscrit dans la thématique "Assurance Santé" du cahier des charges, avec une problématique de régression supervisée.

---

## 2. Données et Nettoyage

| **Aspect** | **Description** |
|------------|----------------|
| **Source** | Kaggle - Medical Insurance Dataset |
| **Taille** | 1 338 observations, 7 variables |
| **Target** | `charges` (coûts médicaux en USD) |
| **Features** | age, sex, bmi, children, smoker, region |
| **Types** | 3 numériques (age, bmi, children), 3 catégorielles (sex, smoker, region) |
| **Valeurs manquantes** | Aucune |
| **Doublons** | 1 ligne retirée |
| **Encodage** | One-Hot Encoding pour sex et region, Label Encoding pour smoker |
| **Normalisation** | StandardScaler appliqué sur age, bmi, children |

### Code - Chargement et Nettoyage

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Chargement des données
df = pd.read_csv('insurance.csv')

# Vérification des valeurs manquantes
print(df.isnull().sum())

# Suppression des doublons
df.drop_duplicates(inplace=True)

# Encodage des variables catégorielles
le = LabelEncoder()
df['smoker'] = le.fit_transform(df['smoker'])
df['sex'] = le.fit_transform(df['sex'])

# One-Hot Encoding pour region
df = pd.get_dummies(df, columns=['region'], drop_first=True)

print(df.info())
print(df.describe())
```

---

## 3. Analyse Exploratoire des Données (EDA)

### Insights Majeurs

**Insight 1 : Impact du tabagisme**  
Les fumeurs présentent des coûts médicaux 3 fois supérieurs aux non-fumeurs (moyenne : 32 050 USD vs 8 434 USD). Le tabagisme est le prédicteur le plus discriminant.

**Insight 2 : Corrélation âge-BMI-coûts**  
L'âge et l'IMC montrent une corrélation positive modérée avec les charges (r=0.30 et r=0.20). Cependant, l'effet combiné "fumeur + IMC élevé" amplifie drastiquement les coûts.

**Insight 3 : Distribution bimodale**  
La distribution des charges révèle deux groupes distincts : un cluster bas coût (<15 000 USD, 75% des cas) et un cluster haut coût (>30 000 USD, fumeurs principalement).

### Code - Visualisations EDA

```python
# Configuration du style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Distribution des charges par statut fumeur
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='smoker', y='charges', palette='Set2')
plt.title('Distribution des Charges par Statut Fumeur', fontsize=14, weight='bold')
plt.xlabel('Fumeur (0=Non, 1=Oui)')
plt.ylabel('Charges (USD)')
plt.show()

# Statistiques descriptives
print(df.groupby('smoker')['charges'].describe())

# 2. Heatmap de corrélation
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Matrice de Corrélation des Variables', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# 3. Distribution des charges (histogramme)
plt.figure(figsize=(10, 6))
plt.hist(df['charges'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Charges (USD)')
plt.ylabel('Fréquence')
plt.title('Distribution des Charges Médicales', fontsize=14, weight='bold')
plt.axvline(df['charges'].mean(), color='red', linestyle='--', label='Moyenne')
plt.axvline(df['charges'].median(), color='green', linestyle='--', label='Médiane')
plt.legend()
plt.show()

# 4. Relation Age-BMI-Charges avec smoker
plt.figure(figsize=(12, 6))
scatter = plt.scatter(df['age'], df['bmi'], c=df['charges'], 
                     s=50, alpha=0.6, cmap='viridis')
plt.colorbar(scatter, label='Charges (USD)')
plt.xlabel('Âge')
plt.ylabel('BMI')
plt.title('Relation Âge-BMI-Charges', fontsize=14, weight='bold')
plt.show()

# 5. Feature Engineering - Création de variables d'interaction
df['smoker_bmi'] = df['smoker'] * df['bmi']
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'senior'])
df['age_group'] = le.fit_transform(df['age_group'])
```

### Graphique Récapitulatif

```
Corrélations avec 'charges' :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
smoker     : 0.79 ██████████████████
age        : 0.30 ███████
bmi        : 0.20 █████
children   : 0.07 ██
sex        : 0.06 █
```

---

## 4. Méthodologie et Modèles

**Split de données**  
- Training set : 80% (1 070 obs.)  
- Test set : 20% (268 obs.)  
- Validation : Cross-Validation k=5 folds

### Code - Préparation et Modélisation

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Séparation features / target
X = df.drop('charges', axis=1)
y = df['charges']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation
scaler = StandardScaler()
numeric_cols = ['age', 'bmi', 'children', 'smoker_bmi']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ===== MODÈLE 1 : Linear Regression =====
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}, R²: {r2_lr:.3f}")

# ===== MODÈLE 2 : Ridge Regression avec GridSearch =====
ridge = Ridge()
param_grid_ridge = {'alpha': [0.1, 1, 10, 50, 100]}
grid_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, 
                          scoring='neg_mean_squared_error')
grid_ridge.fit(X_train, y_train)

best_ridge = grid_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge (alpha={grid_ridge.best_params_['alpha']}) - RMSE: {rmse_ridge:.2f}, MAE: {mae_ridge:.2f}, R²: {r2_ridge:.3f}")

# ===== MODÈLE 3 : Random Forest avec optimisation =====
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, 
                       scoring='neg_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R²: {r2_rf:.3f}")

# Cross-Validation sur le meilleur modèle
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, 
                            scoring='r2')
print(f"CV R² scores: {cv_scores}")
print(f"CV R² mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
```

---

## 5. Résultats et Interprétations

| Métrique | Linear Reg. | Ridge | Random Forest |
|----------|-------------|-------|---------------|
| **RMSE** | 6 045 USD | 6 012 USD | 4 789 USD |
| **MAE** | 4 234 USD | 4 201 USD | 2 567 USD |
| **R²** | 0.751 | 0.754 | 0.856 |
| **CV R² (mean)** | - | - | 0.848 ± 0.021 |

**Modèle retenu : Random Forest (R²=0.856)**

### Code - Analyse des Résultats

```python
# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(8), x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - Random Forest', fontsize=14, weight='bold')
plt.xlabel('Importance')
plt.show()

print(feature_importance)

# Graphique Prédictions vs Réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Prédiction parfaite')
plt.xlabel('Charges Réelles (USD)')
plt.ylabel('Charges Prédites (USD)')
plt.title('Prédictions vs Valeurs Réelles - Random Forest', fontsize=14, weight='bold')
plt.legend()
plt.show()

# Analyse des résidus
residuals = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.6, color='coral')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Charges Prédites (USD)')
plt.ylabel('Résidus (USD)')
plt.title('Analyse des Résidus', fontsize=14, weight='bold')
plt.show()

# Distribution des erreurs
print(f"Erreur moyenne absolue: {mae_rf:.2f} USD")
print(f"Erreur quadratique: {rmse_rf:.2f} USD")
print(f"Erreurs > 10000 USD: {(np.abs(residuals) > 10000).sum()} cas ({(np.abs(residuals) > 10000).sum()/len(residuals)*100:.1f}%)")
```

**Interprétation métier**  
Le modèle confirme que le tabagisme est le facteur de risque dominant (47% d'importance). Pour un fumeur de 40 ans avec IMC de 30, le surcoût prédit est de +23 000 USD/an comparé à un profil identique non-fumeur. Ces prédictions permettent une tarification différenciée et des campagnes de prévention ciblées.

---

## 6. Lien avec le Sujet du Projet

Ce projet répond strictement à la thématique **"Assurance Santé"** du cahier des charges (section 3, page 1) qui demande d'estimer le montant annuel des frais médicaux par patient via une approche de régression.

**Arguments concrets de conformité :**

1. **Type de ML** : Régression supervisée avec variable cible continue (charges médicales)
2. **Problématique métier** : Optimisation de la tarification et identification des risques, cas d'usage standard en actuariat
3. **Cycle complet** : Preprocessing rigoureux, EDA approfondi, comparaison de 3+ modèles, validation croisée, optimisation hyperparamètres (GridSearchCV)
4. **Valeur ajoutée** : Modèle déployable avec R²=0.856, permettant une estimation fiable à ±4 800 USD en moyenne, exploitable pour les décisions tarifaires

---

## 7. Limites et Recommandations

**Limites identifiées :**

1. **Taille du dataset** : 1 338 observations limitent la généralisation. Risque de biais géographique (données US uniquement).
2. **Variables absentes** : Absence d'historique médical (maladies chroniques, hospitalisations) qui sont des prédicteurs majeurs.
3. **Outliers** : 8% de valeurs extrêmes mal prédites (charges >50k USD), nécessitant un traitement spécifique.
4. **Temporalité** : Pas de dimension temporelle, impossible de prédire l'évolution des coûts sur plusieurs années.

**Recommandations :**

1. **Enrichissement** : Intégrer des données de parcours de soins (ICD codes, fréquence consultations).
2. **Techniques avancées** : Tester XGBoost avec gestion native des outliers et LightGBM pour gains de performance.
3. **Explicabilité** : Implémenter SHAP values pour justifier les prédictions auprès des régulateurs.
4. **Déploiement** : Créer une API REST avec monitoring continu des dérives de modèle (concept drift).

---

## 8. Code Complet - Requirements.txt

```txt
pandas==2.1.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
jupyter==1.0.0
```

---

**Conclusion**  
Ce projet démontre la faisabilité d'un modèle prédictif robuste (R²=0.856) pour l'estimation des coûts d'assurance santé. Le Random Forest offre le meilleur compromis précision-interprétabilité. Les axes d'amélioration identifiés ouvrent la voie à un MVP déployable en environnement production.

---

**Livrables disponibles sur GitHub :**  
`README.md` | `notebook.ipynb` | `requirements.txt` | `rapport.pdf` | `video_storytelling.mp4`
