# fct pour importer le data
def import_data(data_name):
    import pandas as pd
    
    return pd.read_csv(f"../data/{data_name}")

# fct qui vérifie si les colonnes d’un DataFrame ont le bon type de valeurs
def check_column_types(df):
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'float64':
            if not all(isinstance(x, float) for x in df[col].dropna()):
                print(f"La colonne '{col}' devrait contenir des float, mais contient d'autres types.")
        
        elif col_type == 'int64':
            if not all(isinstance(x, (int, float)) for x in df[col].dropna()):
                print(f"La colonne '{col}' devrait contenir des int, mais contient d'autres types.")
        
        elif col_type == 'object':
            if not all(isinstance(x, str) for x in df[col].dropna()):
                print(f"La colonne '{col}' devrait contenir des chaînes, mais contient d'autres types.")


# fct pour compter les doublons
def count_duplicated(df):
    return int(df.duplicated().sum())


# fct pour afficher les infos du dataframe
def affichage_info(dataframe_name):
    dataframe_name.info()


# fct pour afficher les statistiques descriptives
def affichage_description(dataframe_name):
    return dataframe_name.describe()

# fct pour calculer le nombre manquante dans mon dataframe
def isempty_count(dataframe_name):
    return dataframe_name.isnull().sum()

# fct pour compter les doublons
def count_dublicated(dataframe_name):
    return int(dataframe_name.duplicated().sum())

# fct pour la visualisation avec count_plot
def count_plot_affichage(dataframe_name,column_name):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8,4))
    sns.countplot(x=column_name, data=dataframe_name, hue='Churn')
    plt.title(f"Churn by {column_name}")
    plt.xticks(rotation=45)  
    plt.show()
    
    
# fct pour encode toutes les colonnes catégorielles spécifiées en valeurs numeriques
def encode_categorical(dataframe_name):
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    
    dataframe_encoded = dataframe_name.copy()
    categorical_cols = dataframe_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        dataframe_encoded[col] = le.fit_transform(dataframe_encoded[col])
    return dataframe_encoded

# fct qui entraîne un modele, predit et calcule toutes les métriques cles
def evaluate_model(model, X_train, X_test, y_train, y_test,scale = True):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
    
    # Normalisation si demandé
    if scale:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # ROC-AUC si possible
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = None
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc
    }
    
    return metrics


def plot_roc(models, X_test, y_test):
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    plt.figure(figsize=(8,6))
    colors = ['blue', 'green', 'orange', 'purple', 'brown']

    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, color=colors[i % len(colors)], label=f"{name} (AUC = {auc:.2f})")

    # Ligne de hasard
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Hasard (AUC = 0.5)')

    plt.title('Courbes ROC comparatives')
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

