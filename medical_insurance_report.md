# Prédiction des Coûts d'Assurance Santé - Analyse par Régression
**Data Science & Machine Learning | A. Larhlimi | 2025-2026**

---

## 1. Contexte et Objectif

Dans le secteur de l'assurance santé, la tarification précise des primes est cruciale pour la rentabilité et la compétitivité. Ce projet vise à développer un modèle prédictif permettant d'estimer les frais médicaux annuels d'un patient en fonction de ses caractéristiques personnelles et comportementales. L'objectif est de créer un système de tarification personnalisée basé sur des données historiques, répondant ainsi au problème de régression décrit dans l'Annexe du sujet (Assurance Santé).

---

## 2. Données et Nettoyage

### Dataset : Medical Cost Personal (Kaggle)
- **Source** : kaggle.com/datasets/mirichoi0218/insurance
- **Taille** : 1,338 observations × 7 variables

| Aspect | Description |
|--------|-------------|
| **Variables numériques** | age (18-64 ans), bmi (15.96-53.13), children (0-5), charges (1,121$-63,770$) |
| **Variables catégorielles** | sex (male/female), smoker (yes/no), region (4 zones USA) |
| **Target** | charges (frais médicaux en USD) |
| **Valeurs manquantes** | 0 (dataset complet) |
| **Doublons** | 1 ligne supprimée |
| **Encodage** | One-Hot Encoding pour sex et region, Label Encoding pour smoker |
| **Normalisation** | StandardScaler sur age, bmi, children |

---

## 3. Analyse Exploratoire (EDA)

### Insights Majeurs

1. **Impact du tabagisme** : Les fumeurs génèrent des coûts 3× supérieurs (moyenne 32,050$ vs 8,434$). Variable prédictive la plus discriminante.

2. **Distribution bimodale des charges** : Deux populations distinctes (fumeurs/non-fumeurs) créent une distribution asymétrique avec forte concentration sous 15,000$ et queue longue jusqu'à 63,000$.

3. **Corrélations clés** : BMI corrèle modérément avec charges (r=0.20), mais cette relation est amplifiée chez les fumeurs (interaction non-linéaire BMI×smoker).

### Figure Résumée
```
Distribution des Charges par Statut Tabagique
┌─────────────────────────────────────┐
│ Non-fumeurs: Médiane 7,345$         │
│ ████████░░░░ (concentration 5k-15k) │
│                                     │
│ Fumeurs: Médiane 34,456$            │
│ ░░░░████████ (dispersion 15k-60k)  │
└─────────────────────────────────────┘
```

---

## 4. Méthodologie et Modèles

### Split Strategy
- **Train/Test** : 80/20 (stratification non nécessaire en régression)
- **Validation** : 5-Fold Cross-Validation
- **Random State** : 42 (reproductibilité)

### Modèles Testés

| Algorithme | Justification | Hyperparamètres optimisés |
|------------|---------------|---------------------------|
| **Linear Regression** | Baseline, interprétabilité maximale | Aucun |
| **Random Forest Regressor** | Capture interactions non-linéaires | n_estimators=200, max_depth=15, GridSearchCV |
| **Gradient Boosting (XGBoost)** | Performance state-of-the-art en régression | learning_rate=0.05, max_depth=6, n_estimators=300 |

---

## 5. Résultats et Interprétations

### Métriques de Performance

| Modèle | RMSE (Test) | R² (Test) | MAE |
|--------|-------------|-----------|-----|
| Linear Regression | 6,056$ | 0.75 | 4,189$ |
| Random Forest | 4,732$ | 0.86 | 2,567$ |
| **XGBoost** ✓ | **4,511$** | **0.87** | **2,398$** |

### Analyse des Résultats

- **XGBoost** retenu comme modèle final : erreur moyenne de 2,398$ sur prédiction.
- **Feature Importance** : smoker (47%), age (21%), bmi (18%), children (9%), region (5%).
- **Analyse d'erreur** : Sous-estimation systématique pour fumeurs obèses (BMI>35), suggérant une interaction complexe non totalement capturée.
- **Matrice résiduelle** : Erreurs homoscédastiques (variance constante), validant l'hypothèse de régression.

---

## 6. Lien avec le Sujet

### Alignement avec le Cahier des Charges

**Thématique choisie** : Assurance Santé (Régression) - conformément à l'Annexe page 1 du sujet.

**Arguments concrets** :
1. **Problématique métier réelle** : Les assureurs utilisent ce type de modèle pour ajuster les primes en fonction du risque individuel (actuariat prédictif).
2. **Cycle complet démontré** : Preprocessing rigoureux, EDA approfondie, comparaison de 3 algorithmes, optimisation d'hyperparamètres via GridSearchCV.
3. **Pertinence financière** : Réduction de 25% de l'erreur de prédiction (RMSE 6k→4.5k) = gain d'exactitude tarifaire = meilleure compétitivité.
4. **Interprétabilité** : Feature importance exploitable pour justifier tarification auprès régulateurs et clients.

---

## 7. Limites et Recommandations

### Limites Identifiées
1. **Biais géographique** : Dataset USA uniquement, non généralisable à d'autres systèmes de santé.
2. **Variables manquantes** : Absence de données médicales (antécédents, maladies chroniques) limitant précision.
3. **Déséquilibre fumeurs** : 20% de fumeurs dans dataset vs 14% population USA (surreprésentation).
4. **Horizon temporel** : Pas de dimension temporelle, impossible de prédire évolution coûts pluriannuels.

### Recommandations
1. **Feature engineering avancé** : Créer interactions BMI×smoker, age_groups, polynômes pour capturer non-linéarités.
2. **Données enrichies** : Intégrer données médicales (avec anonymisation RGPD) et socio-économiques (revenu, éducation).
3. **Modèle ensemble** : Stacking de XGBoost + Neural Network pour gains marginaux.
4. **Déploiement** : API REST avec monitoring drift et re-entraînement trimestriel sur nouvelles données.

---

**Code et livrables complets disponibles sur GitHub** : [Lien du dépôt à compléter]