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
def encode_categorical(dataframe_name, columns):
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    
    dataframe_encoded = dataframe_name.copy()  
    for col in columns:
        le = LabelEncoder()
        dataframe_encoded[col] = le.fit_transform(dataframe_encoded[col])
    return dataframe_encoded

